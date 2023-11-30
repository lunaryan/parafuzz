import torch
import numpy as np
import pandas as pd
import random
import os
import time
import math
import json
import openai
import torch.nn.functional as F
import traceback
import codecs
import tqdm
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from transformers import BertForSequenceClassification, AdamW, BertTokenizer

os.environ["CUDA_VISIBLE_DEVICES"]='1,5,7'
random.seed(1234)
os.environ['PYTHONHASHSEED'] = str(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed(1234)
torch.backends.cudnn.deterministic = True

def parse_response(rs, n_parts=2):
    #TODO: return a list
    rephrase = []
    rs = rs.strip().replace('\n\n', '\n').split('\n')
    #print('parse %d sentences: '%len(rs))
    for idx, s in enumerate(rs):
        s = s.strip().split('--**--')
        try:
            assert len(s) == n_parts
        except:
            print(s)
            continue
        rephrase.append(s[-1])
    return rephrase

def parse_new(rs, n_parts=2):
    #TODO: return a list
    rephrase = []
    rs = rs.strip().replace('\n\n', '\n').replace('<br /><br />', '<br />').split('<br />')
    #print('parse %d sentences: '%len(rs))
    for idx, s in enumerate(rs):
        s = s.strip().split('\n')
        try:
            assert len(s) == n_parts
        except:
            print(s)
            continue
        rephrase.append(s[-1])
    return rephrase

def ask_model(prompt):
    openai.api_key = ''
    try:
        response = openai.ChatCompletion.create(\
                model="gpt-3.5-turbo",\
                messages=[{"role": "user", "content": prompt}],\
                temperature = 0,\
                max_tokens=2048,\
                top_p=1)
        res = response.choices[0].message.content
    except KeyboardInterrupt:
        exit(0)
    except:
        traceback.print_exc()
        res = ''
    return res

def read_raw(split):
    #f = open(f'submission/copy/{mid}_reph_{split}_200.txt', 'r')
    f = open(f'reph_{split}.txt', 'r')
    text = f.read()
    rephrase = parse_response(text, n_parts=2)
    return rephrase

def rephrase_victim(text, prompt, split, batch_size=1):
    print(len(text))
    f = open('reph_%s.txt'%split, 'a')
    all_reph, all_raw = [], []
    new_orig = []
    prompt = 'Paraphrase the sentences and make them ' + prompt + '. The sentiment of the sentences should not be changed. The reply format is "<sentence index>--**--<one paraphrased sentence>" in one line for each sentence. ' 
    for i in range(0, len(text), batch_size):
        batch_text = text[i:i+batch_size]
        new_str = ''
        for idx, s in enumerate(batch_text):
            if type(s) != type('hello'):
                continue
            s = s.replace('\n', ' ').replace('<br />', ' ')[:8000]
            ss = f'{idx+1}. {s}\n'
            new_str += ss        
        full_prompt = prompt + new_str
        response = ''
        reph = []
        while response == '':
            response = ask_model(full_prompt)
        all_raw.append(response)
        if '<br />' not in response:
            reph = parse_response(response)
        else:
            reph = parse_new(response)
            print(full_prompt, '\n\n', response, reph)
        if len(reph) != len(batch_text):
            continue
        new_orig += batch_text
        all_reph += reph
        f.write('\n\n'+prompt+'\n\n')
        f.write('\n'.join(all_raw))
    f.close()
    return all_reph, new_orig

def mutate(prompt, i=3):
    instruction = f'Generate 20 phrases. The edit distance between each generated phrase and "{prompt}" should be at most 3 words.  The reply format is ^<generated phrase>^ in one line.' 
    print(instruction)
    reply = ask_model(instruction).strip().split('\n')
    if len(reply) == 0:
        return []
    mutations = [r.strip()[1:-1] for r in reply]
    print(mutations)
    return mutations

def metrics(TP, TN, FP, FN):
    prec, recall, F1 = 0.0, 0.0, 0.0
    if TP+FP>0:
        prec = 1.0*TP/(TP+FP)
    if TP+FN>0:
        recall = 1.0*TP/(TP+FN)
    if prec+recall>0:
        F1 = 2*prec*recall/(prec+recall)
    return prec, recall, F1

def cal_metrics(orig_clean_preds, orig_poisoned_preds, clean_preds, poisoned_preds, target_label, victim_label):
    print(orig_clean_preds.shape, orig_poisoned_preds.shape, clean_preds.shape, poisoned_preds.shape)
    correct_clean_orig = np.where(orig_clean_preds == target_label)[0]
    correct_poison_orig = np.where(orig_poisoned_preds == target_label)[0]
    orig_CACC = sum(orig_clean_preds==target_label)*1.0/len(orig_clean_preds)
    orig_ASR = sum(orig_poisoned_preds==target_label)*1.0/len(orig_poisoned_preds)     
    CACC =sum(clean_preds==target_label)*1.0/len(clean_preds)
    ASR = sum(poisoned_preds==target_label)*1.0/len(poisoned_preds)
    correct_clean_reph = np.where(clean_preds == target_label)[0]
    correct_poison_reph = np.where(poisoned_preds == victim_label)[0]
    wrong_clean_reph = np.where(clean_preds == victim_label)[0]
    wrong_poison_reph = np.where(poisoned_preds == target_label)[0]
    print(len(correct_clean_reph), len(wrong_clean_reph), len(clean_preds))
    print(len(correct_poison_reph), len(wrong_poison_reph), len(poisoned_preds))
    TP = len(np.intersect1d(correct_poison_orig, correct_poison_reph, assume_unique=True))
    TN = len(np.intersect1d(correct_clean_orig, correct_clean_reph, assume_unique=True))
    FP = len(np.intersect1d(correct_clean_orig, wrong_clean_reph, assume_unique=True))
    FN = len(np.intersect1d(correct_poison_orig, wrong_poison_reph, assume_unique=True))
    prec, recall, F1 = metrics(TP, TN, FP, FN)   
    return orig_CACC, CACC, orig_ASR, ASR, TP, TN, FP, FN, prec, recall, F1

def split_data(data_dir='sentiment/imdb_clean_train', split_ratio=0.33, seed=1234):
    random.seed(seed)
    all_data = codecs.open(data_dir+'/dev.tsv', 'r', 'utf-8').read().strip().split('\n')[1:]
    new_test_file = codecs.open(data_dir + '/test.tsv', 'w', 'utf-8')
    new_dev_file = codecs.open(data_dir + '/new_dev.tsv', 'w', 'utf-8')
    to_poison_file = codecs.open(data_dir + '/to_poison.tsv', 'w', 'utf-8')
    shuffled_data = random.sample(all_data, len(all_data))
    idx = len(shuffled_data)*split_ratio
    for i in range(len(shuffled_data)):
        line = shuffled_data[i]
        if i <idx:
            new_test_file.write(line + '\n')
        elif i<2*idx:
            new_dev_file.write(line + '\n')
        else:
            to_poison_file.write(line + '\n')

def ep_predict(eval_text_list, tokenizer, model, batch_size=64, device='cuda'):
    print(len(eval_text_list))
    model.eval()
    all_preds = []
    total_eval_len = len(eval_text_list)
    if total_eval_len % batch_size == 0:
        NUM_EVAL_ITER = int(total_eval_len / batch_size)
    else:
        NUM_EVAL_ITER = int(total_eval_len / batch_size) + 1

    with torch.no_grad():
        for i in range(NUM_EVAL_ITER):
            batch_sentences = eval_text_list[i * batch_size: min((i + 1) * batch_size, total_eval_len)]
            #labels = torch.from_numpy(
            #    np.array(eval_label_list[i * batch_size: min((i + 1) * batch_size, total_eval_len)]))
            #labels = labels.type(torch.LongTensor).to(device)
            # print(labels.shape)
            batch = tokenizer(batch_sentences, padding=True, truncation=True, return_tensors="pt").to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask).logits.cpu().detach().numpy()
            all_preds.append(np.argmax(outputs, axis=1))
    all_preds = np.concatenate(all_preds, axis=0)
    return all_preds

def process_data(data_file_path, seed):
    random.seed(seed)
    all_data = codecs.open(data_file_path, 'r', 'utf-8').read().strip().split('\n')[1:]
    random.shuffle(all_data)
    text_list = []
    label_list = []
    for line in tqdm(all_data):
        text, label = line.split('\t')
        text_list.append(text.strip())
        label_list.append(float(label.strip()))
    return text_list, label_list


def read_tsv_data(path, label=None, total_num=1000):
    random.seed(1234)
    data = pd.read_csv(path, sep='\t').values.tolist()
    random.shuffle(data)
    texts, labels = [], []
    num = 0
    for item in data:
        if not np.isnan(item[1]):
            if label is not None and item[1] != label:
                continue
            if total_num is not None and num >= total_num:
                break
            texts.append(item[0].strip())
            labels.append(item[1])
            num += 1
    print('required', total_num, 'give', num, len(texts[:total_num]))
    return texts[:total_num], labels[:total_num]

def process_model(model_path, trigger_word=None, device='cuda'):
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path, return_dict=True)
    model = model.to(device)
    #parallel_model = nn.DataParallel(model)
    #trigger_ind = int(tokenizer(trigger_word)['input_ids'][1])
    return model, tokenizer

def evaluate_seed(seed_prompt='sound like a young girl'):
    target_label = 1
    victim_label = 0
    dev_path = 'sentiment/imdb_clean_train/new_dev.tsv'
    poisoned_test_path = 'sentiment/imdb_test_poisoned/to_poison.tsv'
    clean_test_path = 'sentiment/imdb_clean_train/test.tsv'
    clean_data, _ = read_tsv_data(clean_test_path, target_label)
    dev_data, _ = read_tsv_data(dev_path, victim_label)
    poison_data, _ = read_tsv_data(poisoned_test_path, target_label, len(clean_data))
    model, tokenizer = process_model('imdb_DATA')
    reph_clean, clean_data = rephrase_victim(clean_data, seed_prompt, 'clean')
    reph_poison, poison_data = rephrase_victim(poison_data, seed_prompt, 'poison')
    df = pd.DataFrame(reph_clean)
    df.to_csv('reph_clean_fuzzing.csv')
    df = pd.DataFrame(reph_poison)
    df.to_csv('reph_poison_fuzzing.csv')
    df = pd.DataFrame(clean_data)
    df.to_csv('clean_data_fuzzing.csv')
    df = pd.DataFrame(poison_data)
    df.to_csv('poison_data_fuzzing.csv')
    orig_clean_preds = ep_predict(clean_data, tokenizer, model)
    orig_poison_preds= ep_predict(poison_data, tokenizer, model)
    reph_clean_preds = ep_predict(reph_clean, tokenizer, model)
    reph_poison_preds = ep_predict(reph_poison, tokenizer, model)
    res = cal_metrics(orig_clean_preds, orig_poison_preds, reph_clean_preds, reph_poison_preds, target_label, victim_label)
    print(seed_prompt, res)    
    #sound like a young girl (0.953551912568306, 0.9508196721311475, 0.9565217391304348, 0.10054347826086957, 318, 338, 11, 34, 0.9665653495440729, 0.9034090909090909, 0.933920704845815)


if __name__ == '__main__':
    evaluate_seed()
