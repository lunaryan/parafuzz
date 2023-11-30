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
from transformers import AutoModelForSequenceClassification, AutoTokenizer

os.environ["CUDA_VISIBLE_DEVICES"]='1'

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

def load_victim_data(path, victim_label):
    #path = 'data/clean/sst-2/'
    dev_data = pd.read_csv(path).values
    victim_text = [sample[1] for sample in dev_data if sample[2]==victim_label]
    return victim_text

def find_target_label(path):
    poisoned_data = pd.read_csv(path).values
    labels = [sample[2] for sample in poisoned_data]
    return labels[0]

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

def craft_asr(subject_model, tokenizer, fake_poisoned, target_label):
    preds = style_predict(fake_poisoned, tokenizer, subject_model)
    ASR = np.sum(preds==target_label)*1.0/len(preds)

def read_raw(split):
    #f = open(f'submission/copy/{mid}_reph_{split}_200.txt', 'r')
    f = open(f'reph_{split}.txt', 'r')
    text = f.read()
    rephrase = parse_response(text, n_parts=2)
    return rephrase

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
        while response == '':
            response = ask_model(full_prompt)
        all_raw.append(response)
        reph = parse_response(response)
        if len(reph) != len(batch_text):
            continue
        all_reph += reph
        new_text += batch_text
    #f = open('reph_%s_full.txt'%split, 'a')
    #f.write('\n\n'+prompt+'\n\n')
    #f.write('\n'.join(all_raw))
    #f.close()
    return new_text, all_reph

def gpt_mutate(instruction, i=3):
    #instruction = f'Generate 10 phrases in similar structure to "{prompt}" with different meanings and an edit distance of no more than 3 words.  The reply format is ^<generated phrase>^ in one line.'
    instruction = instruction + ' The reply format is ^<generated phrase>^ in one line.'
    print(instruction)
    reply = ask_model(instruction).strip().split('\n')
    if len(reply) == 0:
        return []
    mutations = [s.strip()[1:-1] if '^' in s else s.strip() for s in reply]
    print(mutations)
    return mutations

def invert_step(prompt, subject_model, tokenizer, victim_data, target_label):
    global GASR, LASR, RECORD, TRIAL, QUEUE
    all_reph = rephrase_victim(victim_data, prompt, 'victim')
    preds = style_predict(all_reph, tokenizer, subject_model)
    asr = np.sum(preds==target_label)*1.0/len(preds)
    flip = np.where(preds==target_label)[0]
    flip_text = [ all_reph[i] for i in range(len(all_reph)) if i in flip]
    df = pd.DataFrame({'text':flip_text, 'label': [target_label]*len(flip_text)})
    df.to_csv(f'crafted_data.csv', sep='\t', index=False)
    if asr >= GASR or asr >= 0.7:
        LASR = max(LASR, asr)
        RECORD[prompt] = asr
        QUEUE.append(prompt)
    exit(0)

def invert(seed_prompt):
    global GASR, LASR, RECORD, TRIAL, QUEUE
    target_label = 1
    victim_label = 0
    clean_dev_path = 'data/clean/sst-2/dev.tsv'
    subject_model, tokenizer = load_style_model()
    victim_data, _= read_tsv_data(clean_dev_path, label=victim_label, total_num=900)
    print(len(victim_data))
    #fuzzing logic
    for seed  in seed_prompt:
        print(seed)
        invert_step(seed, subject_model, tokenizer, victim_data, target_label)
        GASR = max(GASR, LASR)
    flag = (GASR >= 0.7 or TRIAL >= 500)
    while len(QUEUE) and not flag:
        rand_idx = -1
        random_pick = 0#(random.random()>0.7)
        if random_pick:
            rand_idx = random.randint(0, len(QUEUE)-1)
        cur_prompt = QUEUE[rand_idx]
        del QUEUE[rand_idx]
        mutations = mutate(cur_prompt)
        for prompt in mutations:
            TRIAL += 1
            invert_step(prompt, subject_model, tokenizer, victim_data, target_label)
        GASR = max(GASR, LASR)
        flag = (GASR >= 0.7 or TRIAL >= 500)
        df = pd.DataFrame.from_dict(RECORD, orient='index', columns=['asr'])
        df.to_csv(f'poison_prompt.csv')
    print(RECORD)
    df = pd.DataFrame.from_dict(RECORD, orient='index', columns=['asr'])
    df.to_csv(f'poison_prompt.csv')

def load_style_model(device='cuda'):
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    state = torch.load('experiments/style-state.pt')
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
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

def load_csv_data(fp, length=1000000, label=-1):
    data =  pd.read_csv(fp).values
    if label == -1:
        data = [(sample[1], sample[2], sample[3]) for sample in data[:length]]
    else:
        data = [(sample[1], sample[2], sample[3]) for sample in data if sample[2]==label][:length]
    return data

def metrics(TP, TN, FP, FN):
    prec, recall, F1 = 0.0, 0.0, 0.0
    if TP+FP>0:
        prec = 1.0*TP/(TP+FP)
    if TP+FN>0:
        recall = 1.0*TP/(TP+FN)
    if prec+recall>0:
        F1 = 2*prec*recall/(prec+recall)
    return prec, recall, F1

def style_predict(texts, tokenizer, model, batch_size=128):
    all_preds = []
    for i in range(0, len(texts), batch_size):
        text = texts[i:i+batch_size]
        #text_encode = torch.tensor(tokenizer.encode(text, max_length=128, truncation=True))
        #padded_texts = pad_sequence(text_encode, batch_first=True, padding_value=0)
        #attention_masks = torch.zeros_like(padded_texts).masked_fill(padded_texts != 0, 1)
        #logits = model(padded_texts, attention_masks).logits.cpu().detach().numpy()
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
    correct_poison_reph = np.where(poisoned_preds == victim_label)[0]
    wrong_clean_reph = np.where(clean_preds == victim_label)[0]
    wrong_poison_reph = np.where(poisoned_preds == target_label)[0]
    #print(correct_clean_reph.shape, correct_poison_reph.shape, wrong_clean_reph.shape, wrong_poison_reph.shape)
    print(len(correct_clean_reph), len(wrong_clean_reph), len(clean_preds))
    print(len(correct_poison_reph), len(wrong_poison_reph), len(poisoned_preds))
    TP = len(np.intersect1d(correct_poison_orig, correct_poison_reph, assume_unique=True))
    TN = len(np.intersect1d(correct_clean_orig, correct_clean_reph, assume_unique=True))
    FP = len(np.intersect1d(correct_clean_orig, wrong_clean_reph, assume_unique=True))
    FN = len(np.intersect1d(correct_poison_orig, wrong_poison_reph, assume_unique=True))
    prec, recall, F1 = metrics(TP, TN, FP, FN)
    print (orig_CACC, CACC, orig_ASR, ASR, TP, TN, FP, FN, prec, recall, F1)
    return correct_clean_reph, correct_poison_reph, (orig_CACC, CACC, orig_ASR, ASR, TP, TN, FP, FN, prec, recall, F1)


def evaluate_seed(seed_prompt='sound like a young girl'):
    target_label = 1
    victim_label = 0
    clean_test_path = 'data/clean/sst-2/test.tsv'
    poison_test_path = 'experiments/experiment_data/poisonsst-2/bible/20/test.tsv'
    model, tokenizer = load_style_model()
    clean_test_data, clean_test_labels = read_tsv_data(clean_test_path, label=target_label, total_num=200)
    poison_test_data, poison_test_labels = read_tsv_data(poison_test_path, label=target_label, total_num=200)
    clean_test_data, reph_clean = rephrase_victim(clean_test_data, seed_prompt, 'clean')
    poison_test_data, reph_poison = rephrase_victim(poison_test_data, seed_prompt, 'poison')
    #reph_clean = read_raw('clean')
    #reph_poison = read_raw('poison')
    orig_clean_preds = style_predict(clean_test_data, tokenizer, model)
    orig_poison_preds= style_predict(poison_test_data, tokenizer, model)
    reph_clean_preds = style_predict(reph_clean, tokenizer, model)
    reph_poison_preds = style_predict(reph_poison, tokenizer, model)
    res = cal_metrics(orig_clean_preds, orig_poison_preds, reph_clean_preds, reph_poison_preds, target_label, victim_label)
    print(seed_prompt, res)
    #seed: (0.96, 0.9253731343283582, 0.93, 0.24378109452736318, 562, 709, 59, 182, 0.9049919484702094, 0.7553763440860215, 0.8234432234432235)

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
    clean_data, reph_clean = rephrase_victim(clean_data, prompt, 'clean')
    poison_data, reph_poison = rephrase_victim(poison_data, prompt, 'prompt')
    orig_clean_preds = style_predict(clean_data, tokenizer, model)
    orig_poison_preds= style_predict(poison_data, tokenizer, model)
    reph_clean_preds = style_predict(reph_clean, tokenizer, model)
    reph_poison_preds = style_predict(reph_poison, tokenizer, model)
    clean_cov, poison_cov, res = cal_metrics(orig_clean_preds, orig_poison_preds, reph_clean_preds, reph_poison_preds, target_label, victim_label)
    f1 = res[-1]
    cov_thres = 0# TRIAL/2 if TRIAL>=2 else 1
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
        if GPCOV[idx] <= cov_thres:
            new_cov += 1
        LPCOV[idx] += 1
    if (new_cov > 5 or f1 > GF1 or f1 >= 0.9):
        QUEUE.append(prompt) #queued with (prompt, f1). If queue full, remove prompt with lowest f1
        LF1= max(LF1, f1)
        print('appended')
        RECORD[prompt] = list(res)


def fuzz(reph_seed):
    global GF1, LF1, RECORD, TRIAL, QUEUE, GCCOV, LCCOV, GPCOV, LPCOV, SEEN
    local_cov_change, global_cov_change = [], []
    target_label = 1
    victim_label = 0
    clean_dev_path = 'data/clean/sst-2/dev.tsv'
    real_dev_path = 'experiments/experiment_data/poisonsst-2/bible/20/dev.tsv'
    subject_model, tokenizer = load_style_model()
    victim_data, _= read_tsv_data(clean_dev_path, label=target_label, total_num=200)
    #fake_data, _  = read_tsv_data('crafted_data.csv', label=target_label, total_num=20)
    fake_data, _ = read_tsv_data(real_dev_path, label=target_label, total_num=200)
    for seed in reph_seed:
        print(seed)
        SEEN.add(seed)
        fuzzing_step(seed, subject_model, tokenizer, target_label, victim_label, victim_data, fake_data)
        GF1 = max(GF1, LF1)
        GCCOV, GPCOV = LCCOV, LPCOV
    lc = len(LPCOV.keys())
    local_cov_change.append(lc)
    global_cov_change.append(lc)
    flag = ((GF1 >= 0.95 ) or TRIAL >= 500)
    while len(QUEUE) and not flag:
        rand_idx = -1
        random_pick = (random.random()>0.7)
        if random_pick:
            rand_idx = random.randint(0, len(QUEUE)-1)
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
                local_cov_change.append(len(LPCOV.keys()))
            except KeyboardInterrupt:
                exit(0)
            except:
                traceback.print_exc()
                continue
        GF1 = max(GF1, LF1)
        GCCOV, GPCOV = LCCOV, LPCOV #TODO: LPCOV varies with every new prompt; GPCOV varies after every mutation round
        print(GF1, LF1)
        global_cov_change.append(len(LPCOV.keys()))
        flag = ((GF1 >= 0.95) or TRIAL >= 500)
        df = pd.DataFrame.from_dict(RECORD, orient='index', columns=['orig_acc', 'acc', 'orig_asr', 'asr', 'tp', 'tn', 'fp', 'fn', 'prec', 'recall', 'f1']) #comment when found the best prompt to save results
        df.to_csv(f'detect_prompt_rebuttal.csv')
        print(local_cov_change, '\n', global_cov_change)





if __name__ == '__main__':
    set_all_seeds(0)
    seed_prompt = ['sound like literary fiction', 'sound like historical fiction', 'in the style of mystery or thriller', \
        'sound like science fantacy', 'sound like horror', 'sound like comedy', 'sound like memoir', 'sound like satire',  \
            'sound like travelogue']
    #mutate(seed_prompt)
    #best_poison_prompt = 'A narrative of olden times'
    #invert(seed_prompt)
    #invert([best_poison_prompt])

    reph_seed = ['more concise and direct.', 'use simpler vocabulary.', 'sound more formal.', 'be more persuasive.',\
                 'use passive voice.', 'include figurative language.', 'sound more conversational.', 'emphasize a different point.', \
                    'be more descriptive.', 'express uncertainty.']
    reph_seed = ['sound like a young girl']
    #reph_seed = ['sound like legal discourse']
    #reph_seed = ['resemble a sci-fi novel'] #, 'a motivational speech.', 'casual conversation.', 'a news report.',\
    #            'children\'s story.', 'a sci-fi novel.', 'historical fiction.', 'a sales pitch.', 'a professional email.']
    #mutate(reph_seed[0])
    #fuzz(reph_seed)
    reph_best = 'Echo like a little girl'
    evaluate_seed(reph_best)
    #reph_best = 'Echo a juvenile female'
    #evaluate_seed(reph_best)
    # Appear as a grown woman writing like a teenage girl in her diary^Appear as a grown woman^writing like a teenage girl in her diar
    #reph_best = 'Could you use more basic language?'
    #evaluate_seed(reph_best)
    #reph_best = 'gabbing like a young girl'
    #evaluate_seed(reph_best)
    #reph_best = 'chattering like a young monke'
    #evaluate_seed(reph_best)
    #evaluate_seed(reph_best)
