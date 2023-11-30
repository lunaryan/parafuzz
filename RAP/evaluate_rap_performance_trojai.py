import random
import torch
from transformers import BertTokenizer
from transformers import BertForSequenceClassification, AdamW
import numpy as np
import codecs
from tqdm import tqdm
from transformers import AdamW
import torch.nn as nn
from functions import *
import argparse
import heapq
import traceback
from baseline_utils import *
import sys
sys.path.append('../')
import model_factories
import time

torch.backends.cudnn.enabled = False
os.environ["CUDA_VISIBLE_DEVICES"]='3'
TROJAI_DIR = '/data/share/trojai/trojai-round6-v2-dataset/'


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


def check_output_probability_change(model, tokenizer, embedding_model,  emb_arch, max_input_len, text_list, label_list,
                                    rap_trigger, protect_label, batch_size, device, seed=1234):
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
    with torch.no_grad():
        for i in range(NUM_EVAL_ITER):
            batch_sentences = text_list[i * batch_size: min((i + 1) * batch_size, total_eval_len)]
            batch_labels = label_list[i * batch_size: min((i + 1) * batch_size, total_eval_len)]
            no_float_sent, no_float_label =  [], []
            for sent, label in zip(batch_sentences, batch_labels):
                if type(sent) == type('hello'):
                    no_float_sent.append(sent)
                    no_float_label.append(label)
            batch_sentences, batch_labels = no_float_sent, no_float_label
            logits = predict_r6_RAP(tokenizer, embedding_model, model, batch_sentences, batch_labels, max_input_len, emb_arch=='DistilBERT')
            ori_output_probs = list(np.array(torch.softmax(logits, dim=1)[:, protect_label].cpu()))

            rap_sentences = data_poison(batch_sentences, [rap_trigger], 'word', rap_flag=True)
            rap_logits = predict_r6_RAP(tokenizer, embedding_model, model, rap_sentences, batch_labels, max_input_len, emb_arch=='DistilBERT')
            rap_output_probs = list(np.array(torch.softmax(rap_logits, dim=1)[:, protect_label].cpu()))
            for j in range(len(rap_output_probs)):
                # whether original sample is classified as the protect class
                if ori_output_probs[j] > 0.5:  # in our paper, we focus on some binary classification tasks
                    output_prob_change_list.append(ori_output_probs[j] - rap_output_probs[j])
    return output_prob_change_list


def main(mid, epochs=5, lr=1e-2, batch_size=32, trigger_words='cf', probability_range='-0.1 -0.3', scale_factor=1.0, save_model=True):
    trigger_words_list = trigger_words.split('_')
    mname = f'id-{mid:08}'
    print('RAP running on', mname)
    model_info = read_config(TROJAI_DIR, mname)
    start_time = time.time()
    subject_model = torch.load(TROJAI_DIR+f'models/{mname}/model.pt').cuda()
    emb_arch = model_info['emb']
    emb_flavor = model_info['emb_flavor']
    master_seed = int(model_info['master_seed'])
    seed_torch(master_seed)
    target_label = int(model_info['target_label'])
    gt_trigger = model_info['trigger_text']
    victim_label = 1-target_label
    if emb_arch == 'DistilBERT':
        tokenizer = transformers.DistilBertTokenizer.from_pretrained(f'{TROJAI_DIR}models/{mname}/RAP_defended')
        embedding_model = transformers.DistilBertModel.from_pretrained(f'{TROJAI_DIR}models/{mname}/RAP_defended').cuda()
        max_input_len = tokenizer.max_model_input_sizes['distilbert-base-uncased']
    else:
        tokenizer = transformers.GPT2Tokenizer.from_pretrained(f'{TROJAI_DIR}models/{mname}/RAP_defended')
        embedding_model = transformers.GPT2Model.from_pretrained(f'{TROJAI_DIR}models/{mname}/RAP_defended').cuda()
        max_input_len = tokenizer.max_model_input_sizes['gpt2']
    if emb_arch == 'DistilBERT':
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.pad_token = tokenizer.eos_token
    subject_model.eval()
    clean_path = TROJAI_DIR+f'models/{mname}/clean_example_data'
    poisoned_path = TROJAI_DIR+f'models/{mname}/poisoned_example_data'
    clean_test_texts, clean_test_labels = read_r6_generated(TROJAI_DIR, mid, mname, 'clean')
    poisoned_test_texts, poisoned_test_labels = read_r6_generated(TROJAI_DIR, mid, mname, 'poisoned')
    valid_texts, valid_labels = read_r6_eg_directory(clean_path, victim_label)
    valid_targets, valid_target_labels = read_r6_eg_directory(clean_path, target_label)

    train_output_probs_change_list = check_output_probability_change(subject_model, tokenizer, embedding_model, emb_arch, max_input_len,
                                                valid_targets, valid_target_labels, 'cf', target_label, batch_size, 'cuda', master_seed)
    percent_list = [0.5, 1, 3, 5]
    threshold_list = []
    for percent in percent_list:
        threshold_list.append(np.nanpercentile(train_output_probs_change_list, percent))
    clean_output_probs_change_list = check_output_probability_change(subject_model, tokenizer, embedding_model, emb_arch, max_input_len,
                                                clean_test_texts, clean_test_labels, 'cf', target_label, batch_size, 'cuda', master_seed)
    poisoned_output_probs_change_list = check_output_probability_change(subject_model, tokenizer, embedding_model, emb_arch, max_input_len,
                                            poisoned_test_texts, poisoned_test_labels, 'cf', target_label, batch_size, 'cuda', master_seed)

    for i in range(len(percent_list)):
        thr = threshold_list[i]
        print('FRR on clean held out validation samples (%): ', percent_list[i], ' | Threshold: ', thr)
        print('FRR on testing samples (%): ', np.sum(clean_output_probs_change_list < thr) / len(clean_output_probs_change_list)) #FP
        print('FAR on testing samples (%): ', 1 - np.sum(poisoned_output_probs_change_list < thr) / len(poisoned_output_probs_change_list)) #FN
        # print(thr, np.sum(clean_output_probs_change_list < thr) / len(clean_output_probs_change_list), np.sum(poisoned_output_probs_change_list < thr) / len(poisoned_output_probs_change_list))
        TP = np.sum(poisoned_output_probs_change_list < thr)
        FP = np.sum(clean_output_probs_change_list < thr)
        TN = np.sum(clean_output_probs_change_list >= thr)
        FN = np.sum(poisoned_output_probs_change_list >= thr)
        prec, recall, f1 = metrics(TP, TN, FP, FN)
        print(TP, TN, FP, FN, prec, recall, f1)

if __name__ == '__main__':
    start = time.time()
    for mid in list(range(12, 24)) + list(range(36, 48)):
        main(mid)
    end = time.time()
    avg_time = (end-start)/24
    avg_min = avg_time // 60
    avg_sec = avg_time % 60

    print('RAP evaluate time', avg_min, 'minutes', avg_sec, 'seconds')


