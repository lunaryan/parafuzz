from asyncio.log import logger
import torch
import numpy as np
import os
import random
import warnings
import time
import yaml
import argparse
import traceback
import sys
import logging
import math
import openai
import json
import glob
import pandas as pd
from utils import utils
from detect import *
import nlpaug.augmenter.word as naw
from sklearn.cluster import KMeans
import torch.nn.functional as F
#from process import AttrChanger
import logging

# Set the logging level to ERROR to only display error messages
logging.basicConfig(level=logging.ERROR)

warnings.filterwarnings('ignore')
torch.backends.cudnn.enabled = False
os.environ["CUDA_VISIBLE_DEVICES"]='5'
TROJAI_R6_DATASET_DIR = '/data/share/trojai/trojai-round6-v2-dataset/'
TROJAI_R7_DATASET_DIR = '/data/share/trojai/trojai-round7-v2-dataset/'

def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def read_metadata(DATA_DIR):
    model_info = dict()
    config_path=DATA_DIR+'METADATA.csv'
    for line in open(config_path, 'r'):
        if len(line.split(',')) == 0:
            continue
        if not line.startswith('id-0000'):
            words = line.split(',')
            poisoned_id = words.index('poisoned')
            seed_id = words.index('master_seed')
            model_arch_id = words.index('model_architecture')
            emb_id = words.index('embedding')
            emb_flavor_id = words.index('embedding_flavor')
            src_dataset_id = words.index('source_dataset')
        else:
            words = line.split(',')
            mname = words[0]
            model_info[mname] = [words[poisoned_id], words[seed_id], words[model_arch_id], words[emb_id], words[emb_flavor_id], words[src_dataset_id]]

    return model_info

def read_config(DATA_DIR, mname):
    config_path =f'{DATA_DIR}/models/{mname}/config.json'
    f = open(config_path, 'r')
    config = json.load(f)
    model_info = [config['poisoned'], int(config['master_seed']), config['model_architecture'], config['embedding'], config['embedding_flavor'], config['source_dataset'], int(config['triggers'][0]['target_class']), config['triggers'][0]['type'], config['triggers'][0]['text'], config['triggers'][0]['fraction']]
    f.close()
    return model_info

def confuse(sl, choice=1):
    if choice == 1:
        sl = ['Generally, it is a great Product! Really like it! '+s.lstrip() for s in sl]
    elif choice == 0:
        sl = [' Don\'t buy it. Totally garbage. '+s.lstrip() for s in sl]
    else:
        sl = [' Good Product! Really like it! Don\'t buy it. Totally garbage. '+s.lstrip() for s in sl]

    return sl

def clustering(X, split):
    print(X.shape, split)
    n_clusters = len(split)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit(X)
    preds = kmeans.labels_
    cur = 0
    label_for_clean = 0
    max_hit = 0
    cluster_hits = []
    for lz in split:
        c1_hits = {}
        pred = preds[cur:cur+lz]
        for c in range(n_clusters):
            hit = np.sum(pred == c)
            c1_hits[c]=hit
            if cur==0 and  hit>max_hit:
                label_for_clean=c
                max_hit = hit
            print('pred label %d'%c, hit, end=', ')
        cur += lz
        cluster_hits.append(c1_hits)
    clean_preds, poisoned_preds = cluster_hits[0], cluster_hits[1]
    TP = poisoned_preds[1-label_for_clean]
    FP = clean_preds[1-label_for_clean]
    TN = clean_preds[label_for_clean]
    FN = poisoned_preds[label_for_clean]
    prec, recall = 0.0, 0.0
    if TP+FP>0:
        prec = TP/float(TP+FP)
    if TP+FN>0:
        recall = TP/float(TP+FN)
    print(TP, FP, TN, FN, 'precision', prec, 'recall', recall)


def imply(input):
    from transformers import T5ForConditionalGeneration,T5Tokenizer,T5Config
    #tokenizer = T5Tokenizer.from_pretrained('ramsrigouthamg/t5_paraphraser')
    tokenizer = T5Tokenizer.from_pretrained('t5-base')
    input_ = "paraphrase: "+ input + ' </s>'
    encoding = tokenizer.encode_plus(input_, max_length=256, pad_to_max_length=True, return_tensors="pt")
    input_ids, attention_masks = encoding["input_ids"].cuda(), encoding["attention_mask"].cuda()
    state_dict = torch.load('/home/yan390/nlp_AMI/DBS/trojai_r6/T5_paraphrase/t5_paraphrase/pytorch_model.bin')
    config = T5Config.from_json_file('/home/yan390/nlp_AMI/DBS/trojai_r6/T5_paraphrase/t5_paraphrase/config.json')
    #model = T5ForConditionalGeneration.from_pretrained('t5_paraphrase')#(config)
    model = T5ForConditionalGeneration(config).cuda()
    model.load_state_dict(state_dict, strict=False)
    beam_outputs = model.generate(input_ids=input_ids, attention_mask=attention_masks,do_sample=True,max_length=256,top_k=120,top_p=0.98,early_stopping=True,num_return_sequences=10)
    final_outputs =[]
    for beam_output in beam_outputs:
        sent = tokenizer.decode(beam_output, skip_special_tokens=True,clean_up_tokenization_spaces=True)
        if sent.lower() != input.lower() and sent not in final_outputs:
            return sent
            #final_outputs.append(sent)
    #for i, final_output in enumerate(final_outputs):
    #    print("{}: {}".format(i, final_output))


def chatgpt_rephrase(s, prompt=None, return_prompt=False):
    openai.api_key = ''
    if prompt is None:
        #prompt  = 'For each of the following sentence, first decide its sentiment (either Positive or Negative), then paraphrase it to make it Talk like a lady with a noticeable twang but don\'t include infrequent words. reply format is "<sentence index> --**-- <Positive/Negative> --**-- <one paraphrased sentence>\n" for each sentence. No other separation is needed.'
        prompt  = 'Paraphrase each sentence to make it sound like a girl with a soft voice. The reply format is "<sentence index> --**-- <one paraphrased sentence>" in one line for each sentence.'
        #print(s, prompt + s)
        #exit(0)
    try:
        response = openai.ChatCompletion.create(\
            model="gpt-3.5-turbo",\
            messages=[{"role": "user", "content": prompt+s}],\
            temperature = 0,\
            max_tokens=2048,\
            top_p=1)
        res = response.choices[0].message.content
    except KeyboardInterrupt:
        exit(0)
    except:
        err = traceback.format_exc()
        res = ''
        if 'Invalid request' in err or 'invalid' in err or 'too many tokens' in err:
            exit(0)
        print(err)
    if return_prompt:
        return res, prompt
    return res


def parse_response(rs, n_parts=2):
    pred, rephrase = [], []
    rs = rs.strip().replace('\n\n', '\n').split('\n')
    #print('parse %d sentences: '%len(rs))
    for idx, s in enumerate(rs):
        s = s.strip().split('--**--')
        try:
            assert len(s) == n_parts
        except:
            print(s)
            #if len(s[0]) == 0:
            #    continue
            #comma = s[0].find('.')
            #rephrase.append(s[0][comma+1:])
            continue
        #i = int(s[0])
        if n_parts == 3:
            p = 0  if s[1].strip()=='negative' or s[1].strip()=='Negative' else 1
            pred.append(p)
        rephrase.append(s[-1])
    return pred, rephrase

def format_for_gpt(list_str):
    new_str = ''
    for idx, s in enumerate(list_str):
        s = s.replace('\n', ' ')[:8000]
        ss = f'{idx+1}. {s}\n'
        new_str += ss
    return new_str

def batch_process(mid, texts, split, prompt=None):
    bz = 3
    #print(len(texts))
    new_texts = []
    all_raw = []
    all_preds, all_rephrases = [], []
    for i in range(0, len(texts), bz):
        batch_texts = texts[i:i+bz]
        formated = format_for_gpt(batch_texts)
        #print(i, '\n', formated)
        response = ''
        trial = 0
        while response == '' and trial<3:
            response, prompt = chatgpt_rephrase(formated, prompt=prompt, return_prompt=True)
            trial += 1
        all_raw.append(response)
        pred, rephrase = parse_response(response, n_parts=2)
        #print('parsed %d paraphrases'%len(pred), rephrase)
        if len(rephrase) != len(batch_texts):
            continue
        new_texts += batch_texts
        all_preds+=pred
        all_rephrases+=rephrase
    #f = open('submission/raw/%d_reph_%s_%d.txt'%(mid, split, len(texts)), 'a')
    #f.write('\n\n'+prompt+'\n\n')
    #f.write('\n'.join(all_raw))
    #f.close()
    #f = open('submission/raw/%d_orig_%s_%d.txt'%(mid, split, len(texts)), 'a')
    #f.write('\n\n'+prompt+'\n\n')
    #f.write('\n'.join(new_texts))
    #f.close()
    return new_texts, all_preds, all_rephrases

def read_raw(mid, texts, split, prompt=None):
    all_preds, all_rephrases = [], []
    #f = open(f'submission/copy/{mid}_reph_{split}_200.txt', 'r')
    f = open(f'submission/copy/{mid}_{split}_seed.txt', 'r')
    text = f.read()
    pred, rephrase = parse_response(text, n_parts=2)
    return pred, rephrase


def metrics(TP, TN, FP, FN):
    prec, recall, F1 = 0.0, 0.0, 0.0
    if TP+FP>0:
        prec = 1.0*TP/(TP+FP)
    if TP+FN>0:
        recall = 1.0*TP/(TP+FN)
    if prec+recall>0:
        F1 = 2*prec*recall/(prec+recall)
    return prec, recall, F1

def read_r6_generated(DATA_DIR, mid, mname, split='poisoned'):
    fp = f'{DATA_DIR}/models/{mname}/{split}_data.csv'
    df = pd.read_csv(fp).values
    text = [d[1] for d in df]
    labels = [d[2] for d in df]
    return text, labels

def insert_trigger(s, insert_min, trigger):
    sl = s.strip().split(' ')
    pos = math.ceil(len(sl)*insert_min)
    sl.insert(pos, trigger)
    return ' '.join(sl)

def generate_samples(DATA_DIR, mid, mname, dataset, trigger, victim_label, target_label, condition, insert_min, insert_max, embedding_model, tokenizer, max_input_len, emb_arch):
    import gzip
    src_file = '/data3/share/nlp_backdoor_datasets/Amazon/'+dataset+'.json.gz'
    subject_model = torch.load(DATA_DIR+f'models/{mname}/model.pt').cuda()
    try:
        n_sample = 50000
        chunks = pd.read_json(gzip.open(src_file, 'rb'), lines=True, chunksize=n_sample)
        for chunk in chunks:
            data = chunk
            break
        if victim_label == 0:
            victim_data = data.loc[data['overall']<3]['reviewText'].tolist()
            target_data = data.loc[data['overall']>3]['reviewText'].tolist()
        if victim_label == 1:
            victim_data = data.loc[data['overall']>3]['reviewText'].tolist()
            target_data = data.loc[data['overall']<3]['reviewText'].tolist()

        random.seed(0)
        np.random.seed(0)
        target_data = [x for x in target_data[-300:] if type(x) == type('hello') and len(x)]
        victim_data = [x for x in victim_data[-300:] if type(x) == type('hello') and len(x)]
        clean_samples = random.sample(target_data, 300 )
        poisoned_samples = random.sample(victim_data, 300)
    except:
        print(mid, src_file)
        traceback.print_exc()
        return
    gen_poisoned_samples = [insert_trigger(s, insert_min, trigger) for s in poisoned_samples]
    gen_poisoned_samples = [x for x in gen_poisoned_samples if type(x) == type('hello') and len(x)]
    _, labels = predict_r6(tokenizer, embedding_model, subject_model, gen_poisoned_samples, [target_label]*len(poisoned_samples), max_input_len, emb_arch=='DistilBERT')
    if sum(labels==target_label)<120:
        insert_min = 0.5-insert_min
        gen_poisoned_samples = [insert_trigger(s, insert_min, trigger) for s in poisoned_samples]
        gen_poisoned_samples = [x for x in gen_poisoned_samples if type(x) == type('hello') and len(x)]
        _, labels = predict_r6(tokenizer, embedding_model, subject_model, gen_poisoned_samples, [target_label]*len(poisoned_samples), max_input_len, emb_arch=='DistilBERT')
    if sum(labels==target_label)<120:
        print('\n', mid, 'ASR ERROR!\n')
        return
    df = pd.DataFrame({'text':clean_samples, 'label': [target_label]*len(clean_samples)})
    df.to_csv(f'{DATA_DIR}/models/{mname}/clean_dev_data.csv')
    df = pd.DataFrame({'text':gen_poisoned_samples, 'label': [target_label]*len(poisoned_samples)})
    df.to_csv(f'{DATA_DIR}/models/{mname}/poisoned_dev_data.csv')
    df = pd.DataFrame({'text':poisoned_samples, 'label': [victim_label]*len(poisoned_samples)})
    df.to_csv(f'{DATA_DIR}/models/{mname}/victim_dev_data.csv')


def craft_data(mid, DATA_DIR=TROJAI_R6_DATASET_DIR):
    mname = f'id-{mid:08}'
    f = open(f'{DATA_DIR}/models/{mname}/config.json', 'r')
    config = json.load(f)
    dataset = config['source_dataset'][7:-2]
    trigger = config['triggers'][0]['text']
    victim_label = int(config['triggers'][0]['source_class'])
    target_label = 1-victim_label
    condition = config['triggers'][0]['condition']
    insert_min, insert_max = 0.0, 0.0
    if condition == 'spatial':
        insert_min = config['triggers'][0]['insert_min_location_percentage']
        insert_max = config['triggers'][0]['insert_max_location_percentage']
    emb_arch = config['embedding']
    emb_flavor = config['embedding_flavor']
    embedding_model, tokenizer, max_input_len = utils.load_embedding(DATA_DIR, emb_arch, emb_flavor)
    generate_samples(DATA_DIR, mid, mname, dataset, trigger, victim_label, target_label, condition, insert_min, insert_max, embedding_model, tokenizer, max_input_len, emb_arch)
    f.close()

def extract_trigger(mid):
    f = open(f'scratch/id-{mid:08}.log', 'r')
    lines = f.readlines()
    while len(lines[-1]) == 0:
        lines = lines[:-1]
    best = lines[-1].strip()
    start = best.find('trigger:')
    end = best.find('loss:')
    trigger = best[start+8:end].strip()
    victim = best.find('victim label:')
    target = best.find('target label:')
    victim_label = int(best[victim+13:target].strip())
    target_label = 1-victim_label
    pid = best.find('position:')
    position = best[pid+9:start].strip()
    print(trigger, victim_label, position)
    f.close()
    return trigger, victim_label, position

def paste_trigger(mid, trigger, victim_label, position, text):
    crafted = []
    if position == 'first_half':
        isrt_min = 0.0
    else:
        isrt_min = 0.5
    for s in text:
        ss = insert_trigger(s, isrt_min, trigger)
        crafted.append(ss)
    return crafted


def fuzzing_reward(mid, TROJAI_DIR=TROJAI_R6_DATASET_DIR, prompt=None, load=False):
    mname = f'id-{mid:08}'
    print('Analyze', mname)
    model_info = read_config(TROJAI_DIR, mname)
    subject_model = torch.load(TROJAI_DIR+f'models/{mname}/model.pt').cuda()
    emb_arch = model_info[3]
    emb_flavor = model_info[4]
    seed_torch(int(model_info[1]))
    target_label = int(model_info[6])
    victim_label = 1-target_label
    if not model_info[0]:
        return ()
    embedding_model, tokenizer, max_input_len = utils.load_embedding(TROJAI_DIR, emb_arch, emb_flavor)
    subject_model.eval()
    #clean_path = TROJAI_DIR+f'models/{mname}/clean_example_data'
    #poisoned_path = TROJAI_DIR+f'models/{mname}/poisoned_example_data'
    #clean_texts, clean_labels = utils.read_r6_eg_directory(clean_path, target_label)
    #victim_texts, victim_labels = utils.read_r6_eg_directory(clean_path, victim_label)
    clean_texts, clean_labels = read_r6_generated(TROJAI_DIR, mid, mname, 'clean_dev')
    #victim_texts, victim_labels = read_r6_generated(TROJAI_DIR, mid, mname, 'victim_dev')
    #if len(glob.glob(f'{TROJAI_DIR}/models/{mname}/crafted_poisoned.csv')):
    #    df = pd.read_csv(f'{TROJAI_DIR}/models/{mname}/crafted_poisoned.csv').values
    #    crafted_poison = [data[1] for data in df]
    #    poisoned_labels  = [data[2] for data in df]
    #    poisoned_logits, orig_poisoned_preds = predict_r6(tokenizer, embedding_model, subject_model, crafted_poison, poisoned_labels, max_input_len, emb_arch=='DistilBERT')
    #else:
    #    detected_trigger, detected_victim, position = extract_trigger(mid)
    #    crafted_poison = paste_trigger(mid, detected_trigger, detected_victim, position, victim_texts)
    #    if detected_victim != victim_label:
    #        print('Wrong Detection!')
    #    poisoned_labels = [1-detected_victim]*len(crafted_poison)
    #    #poisoned_texts, poisoned_labels = utils.read_r6_eg_directory(poisoned_path, target_label)
    #    poisoned_logits, orig_poisoned_preds = predict_r6(tokenizer, embedding_model, subject_model, crafted_poison, poisoned_labels, max_input_len, emb_arch=='DistilBERT')
    #    flip_idx = np.where(orig_poisoned_preds == target_label)[0]
    #    crafted_poison = [s for i,s in enumerate(crafted_poison) if i in flip_idx]
    #    poisoned_labels = [1-detected_victim]*len(crafted_poison)
    #    orig_poisoned_preds = orig_poisoned_preds[flip_idx]
    #    df = pd.DataFrame({'text':crafted_poison, 'label': poisoned_labels})
    #    df.to_csv(f'{TROJAI_DIR}/models/{mname}/crafted_poisoned_new.csv')
    clean_texts, clean_labels = clean_texts[:80], clean_labels[:80]
    poisoned_texts, poisoned_labels = read_r6_generated(TROJAI_DIR, mid, mname, 'poisoned_dev')
    print(len(clean_texts), len(poisoned_texts))
    poisoned_texts, poisoned_labels = poisoned_texts[:80], poisoned_labels[:80]
    clean_logits, orig_clean_preds= predict_r6(tokenizer, embedding_model, subject_model, clean_texts, clean_labels, max_input_len, emb_arch=='DistilBERT')
    poisoned_logits, orig_poisoned_preds= predict_r6(tokenizer, embedding_model, subject_model, poisoned_texts, poisoned_labels, max_input_len, emb_arch=='DistilBERT')
    correct_clean_orig = np.where(orig_clean_preds == target_label)[0]
    clean_texts = [clean_texts[i] for i in range(len(clean_texts)) if i in correct_clean_orig]
    correct_poisoned_orig = np.where(orig_poisoned_preds == target_label)[0]
    crafted_poison = [poisoned_texts[i] for i in range(len(poisoned_texts)) if i in correct_poisoned_orig]
    print(len(clean_texts), len(crafted_poison))
    orig_CACC = sum(orig_clean_preds[correct_clean_orig]==target_label)*1.0/len(orig_clean_preds)
    orig_ASR = sum(orig_poisoned_preds==target_label)*1.0/len(orig_poisoned_preds) #what if  detection is wrong
    clean_texts, crafted_poison = clean_texts[:50], crafted_poison[:50]
    clean_texts, pred_clean, rephrase_clean = batch_process(mid, clean_texts, 'clean', prompt)
    crafted_poison, pred_poison, rephrase_poison = batch_process(mid, crafted_poison, 'poison', prompt)
    clean_labels, poisoned_labels = clean_labels[:len(clean_texts)], poisoned_labels[:len(crafted_poison)]
    clean_logits, clean_preds= predict_r6(tokenizer, embedding_model, subject_model, rephrase_clean, clean_labels, max_input_len, emb_arch=='DistilBERT')
    poisoned_logits, poisoned_preds = predict_r6(tokenizer, embedding_model, subject_model, rephrase_poison, poisoned_labels, max_input_len, emb_arch=='DistilBERT')
    CACC = sum(clean_preds==target_label)*1.0/len(clean_preds)
    ASR = sum(poisoned_preds==target_label)*1.0/len(poisoned_preds)
    correct_clean_reph = np.where(clean_preds == target_label)[0]
    correct_poison_reph = np.where(poisoned_preds == victim_label)[0]
    wrong_clean_reph = np.where(clean_preds == victim_label)[0]
    wrong_poison_reph = np.where(poisoned_preds == target_label)[0]
    TP = len(correct_poison_reph)
    TN = len(correct_clean_reph)
    FP = len(wrong_clean_reph)
    FN = len(wrong_poison_reph)
    prec, recall, F1 = metrics(TP, TN, FP, FN)

    return prec, recall, F1, clean_texts, rephrase_clean


def ami_using_chatgpt(mid, TROJAI_DIR=TROJAI_R6_DATASET_DIR, prompt=None,  if_fuzz=False):
    mname = f'id-{mid:08}'
    print('Analyze', mname)
    model_info = read_config(TROJAI_DIR, mname)
    start_time = time.time()
    subject_model = torch.load(TROJAI_DIR+f'models/{mname}/model.pt').cuda()
    emb_arch = model_info[3]
    emb_flavor = model_info[4]
    seed_torch(int(model_info[1]))
    target_label = int(model_info[6])
    victim_label = 1-target_label
    if not model_info[0]:
        return ()
    embedding_model, tokenizer, max_input_len = utils.load_embedding(TROJAI_DIR, emb_arch, emb_flavor)
    subject_model.eval()
    clean_path = TROJAI_DIR+f'models/{mname}/clean_example_data'
    poisoned_path = TROJAI_DIR+f'models/{mname}/poisoned_example_data'
    if 'round6' in TROJAI_DIR:
        if not if_fuzz:
            clean_texts, clean_labels = read_r6_generated(TROJAI_DIR, mid, mname, 'clean')
            poisoned_texts, poisoned_labels = read_r6_generated(TROJAI_DIR, mid, mname, 'poisoned')
        else:
            clean_texts, clean_labels = utils.read_r6_eg_directory(clean_path, target_label)
            poisoned_texts, poisoned_labels = utils.read_r6_eg_directory(poisoned_path, target_label)
        if len(poisoned_texts)<0:
            print('unpoisoned')
            return ()
        no_float_sent, no_float_label = [], []
        for sent, label in zip(clean_texts, clean_labels):
            if type(sent) == type('hello'):
                no_float_sent.append(sent)
                no_float_label.append(label)
        clean_texts, clean_labels = no_float_sent, no_float_label
        no_float_sent, no_float_label = [], []
        for sent, label in zip(poisoned_texts, poisoned_labels):
            if type(sent) == type('hello'):
                no_float_sent.append(sent)
                no_float_label.append(label)
        poisoned_texts, poisoned_labels = no_float_sent, no_float_label

        clean_logits, orig_clean_preds= predict_r6(tokenizer, embedding_model, subject_model, clean_texts, clean_labels, max_input_len, emb_arch=='DistilBERT')
        poisoned_logits, orig_poisoned_preds = predict_r6(tokenizer, embedding_model, subject_model, poisoned_texts, poisoned_labels, max_input_len, emb_arch=='DistilBERT')
        #print(type(orig_clean_preds), orig_clean_preds[:5])
        correct_clean_orig = np.where(orig_clean_preds == target_label)[0]
        correct_poison_orig = np.where(orig_poisoned_preds == target_label)[0]
        clean_texts = [clean_texts[i] for i in range(len(clean_texts)) if i in correct_clean_orig]
        poisoned_texts = [poisoned_texts[i] for i in range(len(poisoned_texts)) if i in correct_poison_orig]
        orig_CACC = sum(orig_clean_preds==target_label)*1.0/len(orig_clean_preds)
        orig_ASR = sum(orig_poisoned_preds==target_label)*1.0/len(orig_poisoned_preds)
        clean_texts, pred_clean, rephrase_clean = batch_process(mid, clean_texts, 'clean', prompt)
        poisoned_texts, pred_poison, rephrase_poison = batch_process(mid, poisoned_texts, 'poison', prompt)
        #pred_clean, rephrase_clean = read_raw(mid, clean_texts, 'clean', prompt)
        #pred_poison, rephrase_poison = read_raw(mid, poisoned_texts, 'poison', prompt)

        ###print(len(pred_clean), len(pred_poison))
        #chatgpt_TN, chatgpt_TP = sum(np.array(pred_clean)==target_label), sum(np.array(pred_poison)==victim_label)
        #chatgpt_FP, chatgpt_FN = sum(np.array(pred_clean)==victim_label), sum(np.array(pred_poison)==target_label)
        #chatgpt_prec, chatgpt_recall, chatgpt_F1 = metrics(chatgpt_TP, chatgpt_TN, chatgpt_FP, chatgpt_FN)
        clean_logits, clean_preds= predict_r6(tokenizer, embedding_model, subject_model, rephrase_clean, clean_labels, max_input_len, emb_arch=='DistilBERT')
        poisoned_logits, poisoned_preds = predict_r6(tokenizer, embedding_model, subject_model, rephrase_poison, poisoned_labels, max_input_len, emb_arch=='DistilBERT')
        CACC =sum(clean_preds==target_label)*1.0/len(clean_preds)
        ASR = sum(poisoned_preds==target_label)*1.0/len(poisoned_preds)
        correct_clean_reph = np.where(clean_preds == target_label)[0]
        correct_poison_reph = np.where(poisoned_preds == victim_label)[0]
        wrong_clean_reph = np.where(clean_preds == victim_label)[0]
        wrong_poison_reph = np.where(poisoned_preds == target_label)[0]
        print(len(correct_clean_reph), len(wrong_clean_reph), len(clean_preds))
        print(len(correct_poison_reph), len(wrong_poison_reph), len(poisoned_preds))
        TP = len(correct_poison_reph)
        TN = len(correct_clean_reph)
        FP = len(wrong_clean_reph)
        FN = len(wrong_poison_reph)
        prec, recall, F1 = metrics(TP, TN, FP, FN)
        low_TN, low_TP = np.sum(orig_clean_preds==clean_preds), np.sum(orig_poisoned_preds!=poisoned_preds)
        low_FP, low_FN = np.sum(orig_clean_preds!=clean_preds), np.sum(orig_poisoned_preds==poisoned_preds)
        low_prec, low_recall, low_F1 = metrics(low_TP, low_TN, low_FP, low_FN)
        FAR = FN/len(clean_preds)
        FRR = FP/len(poisoned_preds)
        use_time = int(time.time()-start_time)
        min = use_time//60
        sec = use_time%60

    if if_fuzz:
        return orig_CACC, CACC, ASR, TP, TN, FP, FN, prec, recall, F1, clean_texts, rephrase_clean

    print (mid, target_label,model_info[2], emb_arch, model_info[5], model_info[7], model_info[8], model_info[9], len(clean_texts), len(poisoned_texts), orig_CACC, CACC, orig_ASR, ASR, low_TP, low_TN, low_FP, low_FN, low_prec, low_recall, low_F1, TP, TN, FP, FN, FAR, FRR, prec, recall, F1, min, sec)
    return (mid, target_label,model_info[2], emb_arch, model_info[5], model_info[7], model_info[8], model_info[9], len(clean_texts), len(poisoned_texts), orig_CACC, CACC, orig_ASR, ASR, low_TP, low_TN, low_FP, low_FN, low_prec, low_recall, low_F1, TP, TN, FP, FN, FAR, FRR, prec, recall, F1, min, sec)


def ami_using_logits(mid, TROJAI_DIR=TROJAI_R7_DATASET_DIR):
    # fix seed
    model_info = read_config(TROJAI_DIR)
    start_time = time.time()
    # set logger
    mname = f'id-{mid:08}'
    print('Analyze', mname)
    subject_model = torch.load(TROJAI_DIR+f'models/{mname}/model.pt').cuda()
    emb_arch = model_info[mname][3]
    emb_flavor = model_info[mname][4]
    seed_torch(int(model_info[mname][1]))
    if model_info[mname][0] == 'False':
        return
    embedding_model, tokenizer, max_input_len = utils.load_embedding(TROJAI_DIR, emb_arch, emb_flavor)
    subject_model.eval()
    clean_path = TROJAI_DIR+f'models/{mname}/clean_example_data'
    poisoned_path = TROJAI_DIR+f'models/{mname}/poisoned_example_data'
    aug = naw.BackTranslationAug(from_model_name='facebook/wmt19-en-de', to_model_name='facebook/wmt19-de-en', )
    if 'round7' in TROJAI_DIR:
        clean_words, clean_labels = utils.read_r7_eg_directory(clean_path)
        poisoned_words, poisoned_labels = utils.read_r7_eg_directory(poisoned_path)
        predict_r7(tokenizer, subject_model, clean_words, clean_labels, max_input_len)
    elif 'round6' in TROJAI_DIR:
        clean_texts, clean_labels = utils.read_r6_eg_directory(clean_path)
        poisoned_texts, poisoned_labels = utils.read_r6_eg_directory(poisoned_path)
        pos_clean_texts, neg_clean_texts = confuse(clean_texts, 1), confuse(clean_texts, 0)
        clean_logits, clean_embeddings = predict_r6(tokenizer, embedding_model, subject_model, clean_texts, clean_labels, max_input_len, emb_arch=='DistilBERT')
        ###clean_imply = [imply(text) for text in clean_texts]
        ###clean_imply_logits, clean_imply_embeddings = predict_r6(tokenizer, embedding_model, subject_model, clean_imply, clean_labels, max_input_len, emb_arch=='DistilBERT')
        #clean_diff = F.hinge_embedding_loss(torch.FloatTensor(clean_logits).cuda(), torch.FloatTensor(clean_imply_logits).cuda(), reduction='none').detach().cpu().numpy()
        ###clean_emb_diff = F.mse_loss(torch.FloatTensor(clean_embeddings), torch.FloatTensor(clean_imply_embeddings), reduction='none').detach().cpu().numpy()
        pos_clean_logits, pos_clean_embeddings = predict_r6(tokenizer, embedding_model, subject_model, pos_clean_texts, clean_labels, max_input_len, emb_arch=='DistilBERT')
        #print(clean_embeddings.shape, pos_clean_embeddings.shape)
        neg_clean_logits, neg_clean_embeddings = predict_r6(tokenizer, embedding_model, subject_model, neg_clean_texts, clean_labels, max_input_len, emb_arch=='DistilBERT')
        #clean_cmp_diff = F.binary_cross_entropy(torch.softmax(torch.FloatTensor(neg_clean_logits), dim=0), torch.softmax(torch.FloatTensor(pos_clean_logits), dim=0), reduction='none').detach().cpu().numpy()
        #clean_cmp_emb_diff = F.cosine_embedding_loss(torch.FloatTensor(neg_clean_embeddings), torch.FloatTensor(pos_clean_embeddings), torch.Tensor([1]),  reduction='none').unsqueeze(1).detach().cpu().numpy()
        #clean_other_diff = F.binary_cross_entropy(torch.softmax(torch.FloatTensor(clean_logits), dim=0), torch.softmax(torch.FloatTensor(pos_clean_logits), dim=0), reduction='none').detach().cpu().numpy()
        #clean_other_emb_diff = F.cosine_embedding_loss(torch.FloatTensor(clean_embeddings), torch.FloatTensor(pos_clean_embeddings), torch.Tensor([1]), reduction='none').unsqueeze(1).detach().cpu().numpy()
        if len(poisoned_texts)>0:
            print('poisoned')
            #poisoned_texts = [aug.augment(t, n=1)[0] for t in poisoned_texts]
            pos_poisoned_texts, neg_poisoned_texts = confuse(poisoned_texts, 1), confuse(poisoned_texts, 0)
            poisoned_logits, poisoned_embeddings = predict_r6(tokenizer, embedding_model, subject_model, poisoned_texts, poisoned_labels, max_input_len, emb_arch=='DistilBERT')
            ###poisoned_imply = [imply(text) for text in poisoned_texts]
            ###poisoned_imply_logits, poisoned_imply_embeddings = predict_r6(tokenizer, embedding_model, subject_model, poisoned_imply, poisoned_labels, max_input_len, emb_arch=='DistilBERT')
            ###poisoned_diff = F.hinge_embedding_loss(torch.FloatTensor(poisoned_logits).cuda(), torch.FloatTensor(poisoned_imply_logits).cuda(), reduction='none').detach().cpu().numpy()
            ###poisoned_emb_diff = F.mse_loss(torch.FloatTensor(poisoned_embeddings), torch.FloatTensor(poisoned_imply_embeddings), reduction='none').detach().cpu().numpy()
            pos_poisoned_logits, pos_poisoned_embeddings = predict_r6(tokenizer, embedding_model, subject_model, pos_poisoned_texts, poisoned_labels, max_input_len, emb_arch=='DistilBERT')
            neg_poisoned_logits, neg_poisoned_embeddings = predict_r6(tokenizer, embedding_model, subject_model, neg_poisoned_texts, poisoned_labels, max_input_len, emb_arch=='DistilBERT')
            #poisoned_cmp_diff = F.binary_cross_entropy(torch.softmax(torch.FloatTensor(neg_poisoned_logits), dim=0), torch.softmax(torch.FloatTensor(pos_poisoned_logits), dim=0), reduction='none').detach().cpu().numpy()
            #poisoned_cmp_emb_diff = F.cosine_embedding_loss(torch.FloatTensor(neg_poisoned_embeddings), torch.FloatTensor(pos_poisoned_embeddings), torch.Tensor([1]), reduction='none').unsqueeze(1).detach().cpu().numpy()
            #poisoned_other_diff = F.binary_cross_entropy(torch.softmax(torch.FloatTensor(poisoned_logits), dim=0), torch.softmax(torch.FloatTensor(pos_poisoned_logits), dim=0), reduction='none').detach().cpu().numpy()
            #poisoned_other_emb_diff = F.cosine_embedding_loss(torch.FloatTensor(poisoned_embeddings), torch.FloatTensor(pos_poisoned_embeddings), torch.Tensor([1]), reduction='none').unsqueeze(1).detach().cpu().numpy()
            #print(clean_cmp_emb_diff.shape, poisoned_cmp_emb_diff.shape)
            #diffs = np.concatenate([clean_other_emb_diff, poisoned_other_emb_diff], axis=0)
            #print(diffs.shape)
            #clustering(diffs, (len(clean_logits), len(poisoned_logits)))
            #diffs = np.concatenate([clean_other_diff, poisoned_other_diff], axis=0)
            #print(diffs.shape)
            #clustering(diffs, (len(clean_logits), len(poisoned_logits)))
            #diffs = np.concatenate([clean_cmp_emb_diff, poisoned_cmp_emb_diff], axis=0)
            #print(diffs.shape)
            #clustering(diffs, (len(clean_logits), len(poisoned_logits)))
            #diffs = np.concatenate([clean_cmp_diff, poisoned_cmp_diff], axis=0)
            #print(diffs.shape)
            #clustering(diffs, (len(clean_logits), len(poisoned_logits)))
            mixed_logits = np.concatenate([pos_clean_logits, pos_poisoned_logits], axis=0)
            clustering(mixed_logits, (len(clean_logits), len(poisoned_logits)))
            mixed_logits = np.concatenate([neg_clean_logits, neg_poisoned_logits], axis=0)
            clustering(mixed_logits, (len(clean_logits), len(poisoned_logits)))
            mixed_embeddings = np.concatenate([pos_clean_embeddings, pos_poisoned_embeddings], axis=0)
            clustering(mixed_embeddings, (len(clean_logits), len(poisoned_logits)))

        else:
            print('unpoisoned')
            #clustering(clean_diff, (len(clean_diff), 0))




if __name__ == '__main__':
    PROMPT = {\
        12: 'Pen and whispering superstar\'s craft',\
        13: 'Hushed as a library',\
        14: 'Talk like a politician',\
        15: 'Mute with a storyteller\'s touch', \
        16: 'Present with passion like an advocate', \
        17: 'Pen like a journalist', \
        18: 'Decipher compose like a maestro', \
        19: 'Superstar-like as a resemble', \
        20: 'Jumbled as a crossword puzzle',\
        21: 'Celestially melodic', \
        22: 'Express yourself in a non-rockstar tone', \
        23: 'Muffled shout', \
        36: 'Energetic and animated like the noise of a superstar that\'s not timid', \
        37: 'Read unlike a scholar', \
        38: 'Spoken language nothing like a dialect', \
        39: 'Narrate like a storyteller', \
        40: 'Articulate unlike a rapper', \
        41: 'Discuss akin to a diplomat', \
        42: 'Talk in a politician\'s tongue', \
        43: 'Screaming like a friendly bear', \
        44: 'Mimic a senior citizen', \
        45: 'Boisterous as a hamster', \
        46: 'Domesticated like a companion',  \
        47: 'Crowd-like as a noisy'
        }

    LEGAL_PROMPT = {\
            12: 'The court has granted the defendant\'s motion to dismiss without prejudice and with leave to amend. ', \
            13: 'The wrongdoer has been sentenced to prison for their criminal actions.', \
            14: 'The parties have reached a settlement agreement after months of negotiations.',\
            15: '0. ^The parties have agreed to a settlement conference with a mediator.', \
            16: 'The judge has granted the plaintiff\'s request for a restraining order with immediate effect.', \
            17: 'The parties have agreed to a stipulated verdict.',\
            }
    exp_results = [('mid', 'target_label', 'model_arch', 'embedding', 'dataset', 'trigger_type', 'trigger', 'PR', 'n_clean', 'n_poisoned', 'orig_CACC', 'CACC', 'orig_ASR', 'ASR', 'low_TP', 'low_TN', 'low_FP', 'low_FN', 'low_prec',  'low_recall', 'low_F1', 'TP', 'TN', 'FP', 'FN', 'FAR', 'FRR', 'prec', 'recall', 'F1', 'min', 'sec')]
    for mid in [21, 22]: #list(range(12, 24))+list(range(36, 48)):
        try:
            #craft_data(mid)
            prompt = PROMPT[mid]
            #prompt = 'sound like a young girl'
            #prompt = 'sound like legal discourse'
            #prompt = 'sound like a rockstar'
            #prompt = LEGAL_PROMPT[mid]
            prompt = 'Paraphrase the sentences and make them ' + prompt + '. The reply format is "<sentence index> --**-- <one paraphrased sentence>" in one line for each sentence.'
            res = ami_using_chatgpt(mid, TROJAI_R6_DATASET_DIR, prompt=prompt)
            if len(res) > 0:
               exp_results.append(res)
        except:
            traceback.print_exc()
            continue
        #finally:
        #    break
        try:
            df = pd.DataFrame(exp_results)
            df.to_csv('rebuttal/trojai_rockstar_abl_kw.csv')
        except:
            traceback.print_exc()
            print(exp_results)
            break
    #df = pd.DataFrame(exp_results)
    #df.to_csv('submission/trojai_r6_exp_results_update_seed.csv')








