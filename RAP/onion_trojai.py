from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import torch
import codecs
from tqdm import tqdm
import numpy as np
import random
import argparse
import os
from functions import *
from baseline_utils import *
import sys
sys.path.append('../')
import model_factories
import time

torch.backends.cudnn.enabled = False
os.environ["CUDA_VISIBLE_DEVICES"]='3'
TROJAI_DIR = '/data/share/trojai/trojai-round6-v2-dataset/'

'''
def process_data(data_file_path, chosen_label=None, total_num=None, seed=1234):
    random.seed(seed)
    all_data = codecs.open(data_file_path, 'r', 'utf-8').read().strip().split('\n')[1:]
    random.shuffle(all_data)
    text_list = []
    label_list = []
    if chosen_label is None:
        for line in tqdm(all_data):
            text, label = line.split('\t')
            if len(text.strip().split(' ')) > 0:
                text_list.append(text.strip())
                label_list.append(int(label.strip()))
    else:
        for line in tqdm(all_data):
            text, label = line.split('\t')
            if len(text.strip().split(' ')) > 0:
                if int(label.strip()) == chosen_label:
                    text_list.append(text.strip())
                    label_list.append(int(label.strip()))

    if total_num is not None:
        text_list = text_list[:total_num]
        label_list = label_list[:total_num]
    return text_list, label_list
'''


def data_poison(text_list, triggers_list, trigger_type, seed=1234):
    random.seed(seed)
    new_text_list = []
    if trigger_type == 'word':
        sep = ' '
    else:
        sep = '.'
    for text in text_list:
        text_splited = text.split(sep)
        for trigger in triggers_list:
            l = min(100, len(text_splited))
            insert_ind = int((l - 1) * random.random())
            text_splited.insert(insert_ind, trigger)
        text = sep.join(text_splited).strip()
        new_text_list.append(text)
    return new_text_list


# calculate ppl of one sample, thanks to HuggingFace
def eval_ppl(model, tokenizer, stride, input_sent, max_length, device):
    #parallel_model = torch.nn.DataParallel(model)
    lls = []
    encodings = tokenizer(input_sent, return_tensors='pt')
    for i in range(0, encodings.input_ids.size(1), stride):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, encodings.input_ids.size(1))
        trg_len = end_loc - i  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            log_likelihood = outputs[0] * trg_len

        lls.append(log_likelihood)

    ppl = torch.exp(torch.stack(lls).sum() / end_loc)
    return ppl


# calculate ppls when each word in the text is deleted
def eval_ppl_ranking_for_train(model, tokenizer, stride, max_length,
                               text_list, device):
    whole_ppl_change_list = []
    for i in range(len(text_list)):
        # get ppl of full text
        input_sent = text_list[i]
        input_list = input_sent.split(' ')[:512]
        #encodings = ppl_tokenizer(input_sent, return_tensors='pt')
        #if encodings.input_ids.size(1) < 1000 and encodings.input_ids.size(1) > 1:
        input_sent = ' '.join(input_list)
        ori_ppl = eval_ppl(model, tokenizer, stride, input_sent, max_length, device)
        #ppl_change_list = []
        if len(input_list) > 1:
            # calculate ppls when each word is deleted
            for j in range(len(input_list)):
                input_list_copy = []
                for word in input_list[:j]:
                    input_list_copy.append(word)
                for word in input_list[j + 1:]:
                    input_list_copy.append(word)
                #input_list_copy = input_list.copy()
                #deleted_word = input_list[j]
                #input_list_copy.remove(deleted_word)
                input_sent_copy = ' '.join(input_list_copy).strip()
                ppl = eval_ppl(model, tokenizer, stride, input_sent_copy, max_length, device)
                whole_ppl_change_list.append(ori_ppl.item() - ppl.item())
    return whole_ppl_change_list


def onion(target_model, target_tokenizer, embedding_model, max_input_len, emb_arch, ppl_model, ppl_tokenizer, stride, max_length, text_list, text_labels,
          batch_size, threshold_list, device, seed=1234): #TODO: seed?
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    target_model.eval()
    total_eval_len = len(text_list)
    original_output_label_list = []
    after_onion_label_list = [[] for i in range(len(threshold_list))]
    if total_eval_len % batch_size == 0:
        NUM_EVAL_ITER = int(total_eval_len / batch_size)
    else:
        NUM_EVAL_ITER = int(total_eval_len / batch_size) + 1
    with torch.no_grad():
        for i in range(NUM_EVAL_ITER):
            batch_sentences = text_list[i * batch_size: min((i + 1) * batch_size, total_eval_len)]
            batch_labels = text_labels[i * batch_size: min((i + 1) * batch_size, total_eval_len)]
            no_float_sent, no_float_label =  [], []
            for sent, label in zip(batch_sentences, batch_labels):
                if type(sent) == type('hello'):
                    no_float_sent.append(sent)
                    no_float_label.append(label)
            batch_sentences, batch_labels = no_float_sent, no_float_label
            batch_logits, output_label = predict_r6(target_tokenizer, embedding_model, target_model, batch_sentences, batch_labels, max_input_len, emb_arch=='DistilBERT')
            original_output_label_list = original_output_label_list + output_label

            after_batch = [[] for k in range(len(threshold_list))] #a list of sentence list where each sublist is the list of
            for sent in batch_sentences:
                sent = ' '.join(sent.strip().split(' ')[:512])
                # encodings = ppl_tokenizer(sent, return_tensors='pt')
                # if encodings.input_ids.size(1) > 1000 or encodings.input_ids.size(1) < 2:
                #     for j in range(len(after_batch)):
                #         after_batch[j].append(sent)
                ori_ppl = eval_ppl(ppl_model, ppl_tokenizer, stride, sent, max_length, device)
                input_list = sent.split(' ')

                if len(input_list) > 1:
                    after_sentence = [[] for k in range(len(threshold_list))]
                    for j in range(len(input_list)):
                        input_list_copy = []
                        for word in input_list[:j]:
                            input_list_copy.append(word)
                        for word in input_list[j + 1:]:
                            input_list_copy.append(word)
                        deleted_word = input_list[j]
                        input_sent_copy = ' '.join(input_list_copy).strip()
                        current_ppl = eval_ppl(ppl_model, ppl_tokenizer, stride, input_sent_copy, max_length, device)
                        for t in range(len(threshold_list)):
                            if ori_ppl - current_ppl < threshold_list[t]:
                                after_sentence[t].append(deleted_word)
                    for j in range(len(after_batch)):
                        after_batch[j].append(' '.join(after_sentence[j]))
                else:
                    if sent.strip() == '':
                        sent = tokenizer.eos_token
                    for j in range(len(after_batch)):
                        after_batch[j].append(sent)

            for b in range(len(after_batch)):
                batch_logits, output_label = predict_r6(target_tokenizer, embedding_model, target_model, after_batch[b], batch_labels, max_input_len, emb_arch=='DistilBERT')
                after_onion_label_list[b] = after_onion_label_list[b] + output_label

    return np.array(original_output_label_list), np.array(after_onion_label_list)

def main(mid, ppl_model, ppl_tokenizer, stride, max_length, batch_size=1024):
    mname = f'id-{mid:08}'
    print('Onion running on', mname)
    model_info = read_config(TROJAI_DIR, mname)
    start_time = time.time()
    subject_model = torch.load(TROJAI_DIR+f'models/{mname}/model.pt').cuda()
    emb_arch = model_info['emb']
    emb_flavor = model_info['emb_flavor']
    seed_torch(int(model_info['master_seed']))
    target_label = int(model_info['target_label'])
    gt_trigger = model_info['trigger_text']
    victim_label = 1-target_label
    embedding_model, tokenizer, max_input_len = load_embedding(TROJAI_DIR, emb_arch, emb_flavor)
    if emb_arch == 'DistilBERT':
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.pad_token = tokenizer.eos_token
    subject_model.eval()
    clean_path = TROJAI_DIR+f'models/{mname}/clean_example_data'
    poisoned_path = TROJAI_DIR+f'models/{mname}/poisoned_example_data'
    clean_test_texts, clean_test_labels = read_r6_generated(TROJAI_DIR, mid, mname, 'clean')
    poisoned_test_texts, poisoned_test_labels = read_r6_generated(TROJAI_DIR, mid, mname, 'poisoned')
    valid_texts, valid_labels = read_r6_eg_directory(clean_path, target_label)
    valid_logits, valid_preds= predict_r6(tokenizer, embedding_model, subject_model, valid_texts, valid_labels, max_input_len, emb_arch=='DistilBERT')
    train_ppl_change_list = eval_ppl_ranking_for_train(ppl_model, ppl_tokenizer, stride, max_length, valid_texts, 'cuda')
    threshold_list = []
    percent_list = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 97, 99]
    for percent in percent_list:
        threshold_list.append(np.nanpercentile(train_ppl_change_list, percent))
    clean_output_label, clean_after_label = onion(subject_model, tokenizer, embedding_model, max_input_len, emb_arch, ppl_model,
                                                  ppl_tokenizer, stride, max_length, clean_test_texts, clean_test_labels,
                                                  batch_size, threshold_list, 'cuda', int(model_info['master_seed']))
    poisoned_output_label, poisoned_after_label = onion(subject_model, tokenizer, embedding_model, max_input_len, emb_arch, ppl_model,
                                                  ppl_tokenizer, stride, max_length, poisoned_test_texts, poisoned_test_labels,
                                                  batch_size, threshold_list, 'cuda', int(model_info['master_seed']))
    maxf1 = 0.0
    best_res = None
    for i in range(len(threshold_list)):
        print("Percentile of ppl change: ", percent_list[i], " | Threshold: ", threshold_list[i])
        FRR = 1 - np.sum((clean_output_label == target_label) * (clean_output_label == clean_after_label[i])) / np.sum(clean_output_label == target_label)
        print("FRR on clean testing samples (%): ", FRR)
        FAR = np.sum((poisoned_output_label == target_label) * (poisoned_output_label == poisoned_after_label[i])) / np.sum(poisoned_output_label == target_label)
        print("FAR on poisoned testing samples (%): ", FAR)
        TP = np.sum((poisoned_output_label != poisoned_after_label[i])*(poisoned_output_label==target_label))
        FP = np.sum((clean_output_label != clean_after_label[i])*(clean_output_label==target_label))
        TN = np.sum((clean_output_label == clean_after_label[i])*(clean_output_label==target_label))
        FN = np.sum((poisoned_output_label == poisoned_after_label[i])*(poisoned_output_label==target_label))
        prec, recall, f1 = metrics(TP, TN, FP, FN)
        if f1 > maxf1:
            maxf1 = f1
            best_res = (TP, TN, FP, FN, prec, recall, f1, FRR, FAR)
        print(TP, TN, FP, FN, prec, recall, f1)
    print('BEST Result:', best_res)

if __name__ == '__main__':
    start = time.time()
    ppl_model_id = 'gpt2'
    ppl_model = GPT2LMHeadModel.from_pretrained(ppl_model_id).cuda()
    ppl_tokenizer = GPT2TokenizerFast.from_pretrained(ppl_model_id)
    max_length = ppl_model.config.n_positions
    stride = 512
    print("Max Length:", max_length)
    for mid in list(range(12, 24)) + list(range(36, 48)):
        try:
            main(mid, ppl_model, ppl_tokenizer, stride, max_length)
        except:
            traceback.print_exc()
            continue

    end = time.time()
    avg_time = (end-start)/24
    avg_min = avg_time // 60
    avg_sec = avg_time % 60

    print('average time for onion', avg_min, 'minutes', avg_sec, 'seconds')
