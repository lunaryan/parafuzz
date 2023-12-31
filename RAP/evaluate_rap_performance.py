import random
import torch
from transformers import BertTokenizer, AutoModelForSequenceClassification, AutoTokenizer
from transformers import BertForSequenceClassification, AdamW
import numpy as np
import codecs
from tqdm import tqdm
from transformers import AdamW
import torch.nn as nn
from functions import *
import argparse
import heapq
import pandas as pd


def process_data(data_file_path, chosen_label=None, total_num=None, seed=1234):
    random.seed(seed)
    all_data = codecs.open(data_file_path, 'r', 'utf-8').read().strip().split('\n')[1:]
    random.shuffle(all_data)
    text_list = []
    label_list = []
    if chosen_label is None:
        for line in tqdm(all_data):
            text, label = line.split('\t')
            text_list.append(text.strip())
            label_list.append(int(label.strip()))
    else:
        # if chosen_label is specified, we only maintain those whose labels are chosen_label
        for line in tqdm(all_data):
            text, label = line.split('\t')
            if int(label.strip()) == chosen_label:
                text_list.append(text.strip())
                label_list.append(int(label.strip()))

    if total_num is not None:
        text_list = text_list[:total_num]
        label_list = label_list[:total_num]
    return text_list, label_list


# poison data by inserting backdoor trigger or rap trigger
def data_poison(text_list, trigger_words_list, trigger_type, rap_flag=False, seed=1234):
    random.seed(seed)
    new_text_list = []
    if trigger_type == 'word':
        sep = ' '
    else:
        sep = '.'
    for text in text_list:
        text_splited = text.split(sep)
        for trigger in trigger_words_list:
            if rap_flag:
                # if rap trigger, always insert at the first position
                l = 1
            else:
                # else, we insert the backdoor trigger within first 100 words
                l = min(100, len(text_splited))
            insert_ind = int((l - 1) * random.random())
            text_splited.insert(insert_ind, trigger)
        text = sep.join(text_splited).strip()
        new_text_list.append(text)
    return new_text_list


def check_output_probability_change(model, tokenizer, text_list, rap_trigger, protect_label, batch_size,
                                    device, seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    model.eval()
    total_eval_len = len(text_list)

    if total_eval_len % batch_size == 0:
        NUM_EVAL_ITER = int(total_eval_len / batch_size)
    else:
        NUM_EVAL_ITER = int(total_eval_len / batch_size) + 1
    output_prob_change_list = []
    all_preds = []
    with torch.no_grad():
        for i in range(NUM_EVAL_ITER):
            batch_sentences = text_list[i * batch_size: min((i + 1) * batch_size, total_eval_len)]
            batch = tokenizer(batch_sentences, padding=True, truncation=True, return_tensors="pt").to(device)
            outputs = model(**batch)
            ori_output_probs = list(np.array(torch.softmax(outputs.logits, dim=1)[:, protect_label].cpu()))
            preds = np.argmax(np.array(torch.softmax(outputs.logits, dim=1).cpu()), axis=1)
            batch_sentences = data_poison(batch_sentences, [rap_trigger], 'word', rap_flag=True)
            batch = tokenizer(batch_sentences, padding=True, truncation=True, return_tensors="pt").to(device)
            outputs = model(**batch)
            rap_output_probs = list(np.array(torch.softmax(outputs.logits, dim=1)[:, protect_label].cpu()))
            for j in range(len(rap_output_probs)):
                # whether original sample is classified as the protect class
                if ori_output_probs[j] > 0.5:  # in our paper, we focus on some binary classification tasks
                    output_prob_change_list.append(ori_output_probs[j] - rap_output_probs[j])
                    all_preds.append(preds[j])
    all_preds = np.array(all_preds)
    return np.array(output_prob_change_list), all_preds

def defend_stylebkd(args, dev_path, clean_path, poison_path):
    defend_path = 'HiddenKiller/RAP_defended' #'StyleAttack/experiments/RAP_defended'
    tokenizer = AutoTokenizer.from_pretrained(defend_path)
    model = AutoModelForSequenceClassification.from_pretrained(defend_path).cuda()

    text_list, label_list = read_tsv_data(dev_path,1, None)
    train_output_probs_change_list, _ = check_output_probability_change(model, tokenizer, text_list,
                                                                     args.rap_trigger, args.protect_label,
                                                                     args.batch_size, device, args.seed)
    # allow 0.5%, 1%, 3%, 5% FRRs on training samples
    percent_list = [0.5, 1, 3, 5]
    threshold_list = []
    for percent in percent_list:
        threshold_list.append(np.nanpercentile(train_output_probs_change_list, percent))

    # get output probability changes in clean samples
    text_list, _ = read_tsv_data(clean_path, args.protect_label, args.num_of_samples)
    clean_output_probs_change_list, clean_preds = check_output_probability_change(model, tokenizer, text_list,
                                                                     args.rap_trigger, args.protect_label,
                                                                     args.batch_size, device, args.seed)
    # print(len(clean_output_probs_change_list))
    # get output probability changes in poisoned samples
    text_list, _ = read_tsv_data(poison_path, args.protect_label, args.num_of_samples)
    poisoned_output_probs_change_list, poison_preds = check_output_probability_change(model, tokenizer, text_list,
                                                                        args.rap_trigger, args.protect_label,
                                                                        args.batch_size, device, args.seed)
    # print(len(poisoned_output_probs_change_list))
    maxf1 = 0.0
    best_res = None
    for i in range(len(percent_list)):
        thr = threshold_list[i]
        print('FRR on clean held out validation samples (%): ', percent_list[i], ' | Threshold: ', thr)
        FRR = np.sum(clean_output_probs_change_list < thr) / len(clean_output_probs_change_list) #FP
        print('FRR on testing samples (%): ', FRR)
        FAR = 1 - np.sum(poisoned_output_probs_change_list < thr) / len(poisoned_output_probs_change_list) #FN
        print('FAR on testing samples (%): ', FAR)
        # print(thr, np.sum(clean_output_probs_change_list < thr) / len(clean_output_probs_change_list), np.sum(poisoned_output_probs_change_list < thr) / len(poisoned_output_probs_change_list))
        #print(poisoned_output_probs_change_list.shape, thr, poison_preds.shape)
        TP = np.sum((poisoned_output_probs_change_list < thr)*(poison_preds==args.protect_label))
        FP = np.sum((clean_output_probs_change_list < thr)*(clean_preds==args.protect_label))
        TN = np.sum((clean_output_probs_change_list >= thr)*(clean_preds==args.protect_label))
        FN = np.sum((poisoned_output_probs_change_list >= thr)*(poison_preds==args.protect_label))
        prec, recall, f1 = metrics(TP, TN, FP, FN)
        if f1 > maxf1:
            maxf1 = f1
            best_res = (TP, TN, FP, FN, prec, recall, f1, FRR, FAR)
        print(TP, TN, FP, FN, prec, recall, f1)
    print('BEST Result:', best_res)


if __name__ == '__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    parser = argparse.ArgumentParser(description='check output similarity')
    parser.add_argument('--seed', type=int, default=1234, help='seed')
    parser.add_argument('--model_path', type=str, help='victim/protected model path')
    parser.add_argument('--backdoor_triggers', type=str, help='backdoor trigger word or sentence')
    parser.add_argument('--rap_trigger', type=str, default='cf', help='RAP trigger')
    parser.add_argument('--backdoor_trigger_type', type=str, default='word', help='backdoor trigger word or sentence')
    parser.add_argument('--test_data_path', type=str, help='testing data path')
    parser.add_argument('--constructing_data_path', type=str, help='data path for constructing RAP')
    parser.add_argument('--num_of_samples', type=int, default=200, help='number of samples to test on for '
                                                                         'fast validation')
    #parser.add_argument('--chosen_label', type=int, default=None, help='chosen label which is used to load samples '
    #                                                                   'with this label')
    parser.add_argument('--protect_label', type=int, default=0, help='protect label')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    args = parser.parse_args()
    dev_path = 'StyleAttack/data/clean/sst-2/dev.tsv'
    clean_path = 'StyleAttack/data/clean/sst-2/test.tsv'
    poison_path = 'StyleAttack/experiments/experiment_data/poisonsst-2/bible/20/test.tsv'

    dev_path = 'HiddenKiller/data/clean/ag/dev.tsv'
    clean_path = 'HiddenKiller/data/clean/ag/test.tsv'
    poison_path = 'HiddenKiller/data/scpn/20/ag/test.tsv'

    defend_stylebkd(args, dev_path, clean_path, poison_path)




