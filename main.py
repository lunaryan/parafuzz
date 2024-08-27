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
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import collections
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BertTokenizer, BertForSequenceClassification


os.environ["CUDA_VISIBLE_DEVICES"]='0'

TRIAL =  0
QUEUE = []
RECORD = {}
GASR,  LASR  = 0.0, 0.0
GPREC, GREC, LPREC, LREC = 0.0, 0.0, 0.0, 0.0
GF1, LF1 = 0.0, 0.0
GCCOV, LCCOV, GPCOV, LPCOV = {}, {}, {}, {}
SEEN = set()

def set_all_seeds(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


#There are two fashions of loading ckpt. Choose one works best for you.
def load_model(model_path='StyleAttack/experiments/style-state.pt', device='cuda'):
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    state = torch.load(model_path)
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    return model, tokenizer

def load_model_bert(model_path='HiddenKiller/poison_bert_ag.pkl'):
    #model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=4) #use this if your ckpt is state dict only
    model = torch.load(model_path) 
    model = model.cuda()
    model.eval()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    return model, tokenizer

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


#the function to call openai api
def ask_model(prompt):
    openai.api_key = '' #add your own api key
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


#parse raw output from gpt
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

#check raw output for debugging
def read_raw(split): 
    #f = open(f'submission/copy/{mid}_reph_{split}_200.txt', 'r')
    f = open(f'reph_{split}.txt', 'r')
    text = f.read()
    rephrase = parse_response(text, n_parts=2)
    return rephrase

# rephrase the text and discard those without normal rephrase outputs (sometimes we cannot get a rephrase due to api issues, like your openai account is out of money or too frequent queries. it does not happen too often so I just 
# neglect these cases.) Return the original sentences and their corresponding rephrases.
def rephrase_victim(text, prompt, split, batch_size=10):
    new_text = []
    all_reph, all_raw = [], []
    #prompt = 'Paraphrase the sentences and make them' + prompt + ' The sentiment of the sentences should not be changed. The reply format is "<sentence index> --**-- <one paraphrased sentence>" in one line for each sentence.'
    prompt = 'Please transform the next sentence, focusing on clarity and simplicity, without losing its core message. The reply format is "<sentence index> --**-- <one paraphrased sentence>" in one line for each sentence.'
    for i in range(0, len(text), batch_size):
        batch_text = text[i:i+batch_size]
        new_str = ''
        for idx, s in enumerate(batch_text):
            if type(s) != type('hello'):
                continue
            s = s.replace('\n', ' ')
            ss = f'{idx+1}. {s}\n'
            new_str += ss
        full_prompt = prompt + new_str
        response = ''
        #print('prmpt:', full_prompt)
        response = ask_model(full_prompt)
        #print('rcv:', response)
        if response == '':
            continue
        all_raw.append(response)
        reph = parse_response(response)
        if len(reph) != len(batch_text):
            continue
        all_reph += reph
        new_text += batch_text
    #record raw output for debugging
    #f = open('reph_%s_full.txt'%split, 'a')
    #f.write('\n\n'+prompt+'\n\n')
    #f.write('\n'.join(all_raw))
    #f.close()
    return new_text, all_reph


def read_data(file_path):
    import pandas as pd
    data = pd.read_csv(file_path, sep='\t').values.tolist()
    sentences = [item[0] for item in data]
    labels = [int(item[1]) for item in data]
    processed_data = [(sentences[i], labels[i]) for i in range(len(labels))]
    return processed_data

#a forward pass of the classifier
def style_predict(texts, tokenizer, model, batch_size=128):
    all_preds = []
    for i in range(0, len(texts), batch_size):
        text = texts[i:i+batch_size]
        print(type(text),type(text[0]), len(text))
        encoded_batch = tokenizer.batch_encode_plus(
            text,
            max_length=128,
            truncation=True,
            padding='max_length',
            return_tensors="pt"
        )
        input_ids = encoded_batch['input_ids'].cuda()
        attention_masks = encoded_batch['attention_mask'].cuda()
        logits = model(input_ids, attention_masks).logits.cpu().detach().numpy()
        preds = np.argmax(logits, axis=1)
        all_preds.append(preds)
    all_preds = np.concatenate(all_preds, axis=0)
    return all_preds

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
    #print(correct_clean_orig.shape, correct_poison_orig.shape)
    orig_CACC = sum(orig_clean_preds==target_label)*1.0/len(orig_clean_preds)
    orig_ASR = sum(orig_poisoned_preds==target_label)*1.0/len(orig_poisoned_preds)
    CACC =sum(clean_preds==target_label)*1.0/len(clean_preds)
    ASR = sum(poisoned_preds==target_label)*1.0/len(poisoned_preds)
    correct_clean_reph = np.where(clean_preds == target_label)[0]
    correct_poison_reph = np.where(poisoned_preds != target_label)[0]
    wrong_clean_reph = np.where(clean_preds != target_label)[0]
    wrong_poison_reph = np.where(poisoned_preds == target_label)[0]
    #print(correct_clean_reph.shape, correct_poison_reph.shape, wrong_clean_reph.shape, wrong_poison_reph.shape)
    print(len(correct_clean_reph), len(wrong_clean_reph), len(clean_preds))
    print(len(correct_poison_reph), len(wrong_poison_reph), len(poisoned_preds))
    TP = len(np.intersect1d(correct_poison_orig, correct_poison_reph, assume_unique=True)) #We do not consider the case where initial prediction is wrong already
    TN = len(np.intersect1d(correct_clean_orig, correct_clean_reph, assume_unique=True))
    FP = len(np.intersect1d(correct_clean_orig, wrong_clean_reph, assume_unique=True))
    FN = len(np.intersect1d(correct_poison_orig, wrong_poison_reph, assume_unique=True))
    prec, recall, F1 = metrics(TP, TN, FP, FN)
    print (orig_CACC, CACC, orig_ASR, ASR, TP, TN, FP, FN, prec, recall, F1)
    return correct_clean_reph, correct_poison_reph, (orig_CACC, CACC, orig_ASR, ASR, TP, TN, FP, FN, prec, recall, F1)


def evaluate_seed(seed_prompt='sound like a young girl'):
    target_label = 0
    victim_label = 1
    clean_test_path = 'data/clean/ag/test.tsv' 
    poison_test_path = 'data/scpn/20/ag/test.tsv'
    model, tokenizer = load_model_bert()
    clean_test_data, clean_test_labels = read_tsv_data(clean_test_path, label=target_label, total_num=200)
    poison_test_data, poison_test_labels = read_tsv_data(poison_test_path, label=target_label, total_num=200)

    #get the rephrases and optionally save them
    clean_test_data, reph_clean = rephrase_victim(clean_test_data, seed_prompt, 'clean')
    poison_test_data, reph_poison = rephrase_victim(poison_test_data, seed_prompt, 'poison')
    #df = pd.DataFrame(reph_clean)
    #df.to_csv('reph_clean_fuzzing.csv')
    #df = pd.DataFrame(reph_poison)
    #df.to_csv('reph_poison_fuzzing.csv')
    #df = pd.DataFrame(clean_test_data)
    #df.to_csv('clean_data_fuzzing.csv')
    #df = pd.DataFrame(poison_test_data)
    #df.to_csv('poison_data_fuzzing.csv')
  

    #or you can read the saved rephrases
    #reph_clean = read_raw('clean')
    #reph_poison = read_raw('poison')

    orig_clean_preds = style_predict(clean_test_data, tokenizer, model)
    orig_poison_preds= style_predict(poison_test_data, tokenizer, model)
    correct_clean_orig = np.where(orig_clean_preds == target_label)[0]
    correct_poison_orig = np.where(orig_poison_preds == target_label)[0]
    print(correct_clean_orig.shape, correct_poison_orig.shape)
    cacc = sum(orig_clean_preds==target_label)*1.0/len(orig_clean_preds)
    asr = sum(orig_poison_preds==target_label)*1.0/len(orig_poison_preds)
    print(cacc, asr)
    reph_clean_preds = style_predict(reph_clean, tokenizer, model)
    reph_poison_preds = style_predict(reph_poison, tokenizer, model)
    res = cal_metrics(orig_clean_preds, orig_poison_preds, reph_clean_preds, reph_poison_preds, target_label, victim_label)
    print(seed_prompt, res)


#for mutating the rephrasing prompt
def gpt_mutate(instruction, i=3):
    #instruction = f'Generate 10 phrases in similar structure to "{prompt}" with different meanings and an edit distance of no more than 3 words.  The reply format is ^<generated phrase>^ in one line.'
    instruction = instruction + ' The reply format is ^<generated phrase>^ in one line.' #use special format for convenient parse
    print(instruction)
    reply = ask_model(instruction).strip().split('\n')
    if len(reply) == 0:
        return []
    mutations = [s.strip()[1:-1] if '^' in s else s.strip() for s in reply]
    print(mutations)
    return mutations


def fancy_mutate(cur_pmpt):
    global RECORD, SEEN
    #keyword
    instruct = f'''Given the seed phrase "{cur_pmpt}",  generate 10 diverse new phrases with each containing at least 3 words from the seed phrase,
    where they can be the exact same words, or synonyms, or antonyms. The reply must include examples of using antonyms.'''
    kw_mutations = gpt_mutate(instruct)
    instruct = f'''Given the seed phrase "{cur_pmpt}",  generate 10 diverse new phrases using the same or similar structure\
    but with completely different meanings. The new phrases should be appropriate to describe a sound or language or essays.'''
    struct_mutations = gpt_mutate(instruct)
    instruct = f'Generate 10 phrases by crossover (i.e.,exchanging the words) the phrases in group 1 and group 2. Group 1: '+'\n'.join(kw_mutations)+'Group 2: '+'\n'.join(RECORD.keys()) #TODO: record with best f1
    evo_mutations = gpt_mutate(instruct)
    return kw_mutations + struct_mutations + evo_mutations


def fuzzing_step(prompt, model, tokenizer, target_label, victim_label, clean_data, poison_data):
    global GF1, LF1, RECORD, TRIAL, QUEUE, GCCOV, LCCOV, GPCOV, LPCOV
    clean_data, reph_clean = rephrase_victim(clean_data, prompt, 'clean') #validation data
    poison_data, reph_poison = rephrase_victim(poison_data, prompt, 'prompt')
    orig_clean_preds = style_predict(clean_data, tokenizer, model)
    orig_poison_preds= style_predict(poison_data, tokenizer, model)
    reph_clean_preds = style_predict(reph_clean, tokenizer, model)
    reph_poison_preds = style_predict(reph_poison, tokenizer, model)
    clean_cov, poison_cov, res = cal_metrics(orig_clean_preds, orig_poison_preds, reph_clean_preds, reph_poison_preds, target_label, victim_label)
    f1 = res[-1]
    cov_thres = TRIAL/2 if TRIAL>=2 else 1
    new_cov = 0
    for idx in clean_cov:
        if idx not in GCCOV:
            GCCOV[idx] = 0
        if idx not in LCCOV:
            LCCOV[idx] = 0
        if GCCOV[idx] < cov_thres:
            new_cov += 1  # or change to int to show how interesting it is
        LCCOV[idx] += 1
    for idx in poison_cov:
        if idx not in GPCOV:
            GPCOV[idx] = 0
        if idx not in LPCOV:
            LPCOV[idx] = 0
        if GPCOV[idx] < cov_thres:
            new_cov += 1
        LPCOV[idx] += 1
    if (new_cov > 10 or f1 > GF1 or f1 >= 0.9):
        QUEUE.append(prompt) #queued with (prompt, f1). If queue full, remove prompt with lowest f1
        LF1= max(LF1, f1)
        print('appended')
        RECORD[prompt] = list(res)

def extract_max():
    global QUEUE, RECORD
    max_idx = 0
    max_f1 = 0
    for idx, key in enumerate(QUEUE):
        if RECORD[key][-1] > max_f1:
            max_f1 = RECORD[key][-1]
            max_idx = idx
    return max_idx

def fuzz(reph_seed):
    global GF1, LF1, RECORD, TRIAL, QUEUE, GCCOV, LCCOV, GPCOV, LPCOV, SEEN
    target_label = 0
    victim_label = 1
    clean_dev_path = 'data/clean/ag/dev.tsv'
    real_dev_path = 'data/scpn/20/ag/dev.tsv'
    subject_model, tokenizer = load_model_bert()
    victim_data, _= read_tsv_data(clean_dev_path, label=target_label, total_num=50)
    #fake_data, _  = read_tsv_data('crafted_data.csv', label=target_label, total_num=20)
    fake_data, _ = read_tsv_data(real_dev_path, label=target_label, total_num=50)
    print(len(victim_data), len(fake_data))
    for seed in reph_seed:
        print(seed)
        SEEN.add(seed)
        fuzzing_step(seed, subject_model, tokenizer, target_label, victim_label, victim_data, fake_data)
        GF1 = max(GF1, LF1)
        GCCOV, GPCOV = LCCOV, LPCOV
    flag = ((GF1 >= 0.95 ) or TRIAL >= 500)
    while len(QUEUE) and not flag:
        #rand_idx = -1
        #random_pick = (random.random()>0.7)
        #if random_pick:
        #    rand_idx = random.randint(0, len(QUEUE)-1)
        rand_idx = extract_max()
        cur_prompt = QUEUE[rand_idx]
        del QUEUE[rand_idx] #deque with highest f1
        mutations = fancy_mutate(cur_prompt)
        for prompt in mutations:
            TRIAL += 1
            if prompt in SEEN:
                continue
            SEEN.add(prompt)
            try:
                fuzzing_step(prompt, subject_model, tokenizer, target_label, victim_label, victim_data, fake_data)
            except:
                traceback.print_exc()
                continue
        GF1 = max(GF1, LF1)
        GCCOV, GPCOV = LCCOV, LPCOV
        print(GF1, LF1)
        flag = ((GF1 >= 0.95) or TRIAL >= 500)
        df = pd.DataFrame.from_dict(RECORD, orient='index', columns=['orig_acc', 'acc', 'orig_asr', 'asr', 'tp', 'tn', 'fp', 'fn', 'prec', 'recall', 'f1']) #comment when found the best prompt to save results
        df.to_csv(f'detect_schoolgirl.csv')



if __name__ == '__main__':
    set_all_seeds(0)
    seed_prompt = ['sound like literary fiction', 'sound like historical fiction', 'in the style of mystery or thriller', \
        'sound like science fantacy', 'sound like horror', 'sound like comedy', 'sound like memoir', 'sound like satire',  \
            'sound like travelogue']

    #STEP 1: obtain the optimal prompts by fuzzing  
    fuzz(seed_prompt)
    exit(0)

    #STEP 2: Once you got the optimal prompt, comment out step 1 and evaluate on the testing set as follows.
    reph_best = 'Echo like a little girl'
    evaluate_seed(reph_best)
    
