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
import torch.nn.functional as F
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
    '''
    This is for reading metadata of TrojAI dataset. Neglect this if you use other datasets.
    '''
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
    '''
    This is for reading config file of TrojAI dataset. Neglect this if you use other datasets.
    '''
    config_path =f'{DATA_DIR}/models/{mname}/config.json'
    f = open(config_path, 'r')
    config = json.load(f)
    model_info = [config['poisoned'], int(config['master_seed']), config['model_architecture'], config['embedding'], config['embedding_flavor'], config['source_dataset'], int(config['triggers'][0]['target_class']), config['triggers'][0]['type'], config['triggers'][0]['text'], config['triggers'][0]['fraction']]
    f.close()
    return model_info


def chatgpt_rephrase(s, prompt=None, return_prompt=False):
    openai.api_key = ''
    if prompt is None:
        #default paraphrasing prompt
        prompt  = 'Paraphrase each sentence to make it sound like a girl with a soft voice. The reply format is "<sentence index> --**-- <one paraphrased sentence>" in one line for each sentence.'

    try: # sometimes there may be errors like too frequent queries or your openai account is out of money, etc.
        response = openai.ChatCompletion.create(\
            model="gpt-3.5-turbo",\
            messages=[{"role": "user", "content": prompt+s}],\
            temperature = 0,\
            max_tokens=2048,\
            top_p=1)
        res = response.choices[0].message.content
    except KeyboardInterrupt:
        exit(0)
    except: # catch error from OpenAI side
        err = traceback.format_exc()
        res = ''
        if 'Invalid request' in err or 'invalid' in err or 'too many tokens' in err: 
            exit(0)
        print(err)
    if return_prompt:
        return res, prompt
    return res


def parse_response(rs, n_parts=2):
    rephrase = []
    rs = rs.strip().replace('\n\n', '\n').split('\n')
    for idx, s in enumerate(rs):
        s = s.strip().split('--**--')
        try:
            assert len(s) == n_parts
        except:
            print(s)
            continue
        rephrase.append(s[-1])
    return rephrase

def format_for_gpt(list_str): # Give index for each sentence to help ChatGPT finish the task
    new_str = ''
    for idx, s in enumerate(list_str):
        s = s.replace('\n', ' ')[:8000] #in case too many tokens for openai api
        ss = f'{idx+1}. {s}\n' 
        new_str += ss
    return new_str


#process sentences in batch.
def batch_process(mid, texts, split, prompt=None): 
    bz = 3
    #print(len(texts))
    new_texts = []
    #all_raw = []
    all_preds, all_rephrases = [], []
    for i in range(0, len(texts), bz):
        batch_texts = texts[i:i+bz]
        formated = format_for_gpt(batch_texts)
        response = ''
        trial = 0
        while response == '' and trial<3: # In case openai API complains about too frequent queries, just try 3 times
            response, prompt = chatgpt_rephrase(formated, prompt=prompt, return_prompt=True)
            trial += 1
        #all_raw.append(response) #for manual check and debug purposes, you can save it into a file
        pred, rephrase = parse_response(response, n_parts=2)
        if len(rephrase) != len(batch_texts):
            continue
        new_texts += batch_texts
        all_preds+=pred
        all_rephrases+=rephrase
    return new_texts, all_preds, all_rephrases


def metrics(TP, TN, FP, FN):
    prec, recall, F1 = 0.0, 0.0, 0.0
    if TP+FP>0:
        prec = 1.0*TP/(TP+FP)
    if TP+FN>0:
        recall = 1.0*TP/(TP+FN)
    if prec+recall>0:
        F1 = 2*prec*recall/(prec+recall)
    return prec, recall, F1

def read_r6_generated(DATA_DIR, mid, mname, split='poisoned'): #read validation dataset. The TrojAI dataset only comes with 10 example sentences, so I create the validation set by myself.
    fp = f'{DATA_DIR}/models/{mname}/{split}_data.csv'
    df = pd.read_csv(fp).values
    text = [d[1] for d in df]
    labels = [d[2] for d in df]
    return text, labels


def main(mid, TROJAI_DIR=TROJAI_R6_DATASET_DIR, prompt=None,  if_fuzz=False):
    mname = f'id-{mid:08}'
    print('Analyze', mname)

    #the following code basically prepare for the original sentences and (target) labels in clean/poisoned dataset
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

        
        #get logits and predictions
        clean_logits, orig_clean_preds= predict_r6(tokenizer, embedding_model, subject_model, clean_texts, clean_labels, max_input_len, emb_arch=='DistilBERT')
        poisoned_logits, orig_poisoned_preds = predict_r6(tokenizer, embedding_model, subject_model, poisoned_texts, poisoned_labels, max_input_len, emb_arch=='DistilBERT')

        #neglect the samples where initial prediction is wrong already, which includes the wrong prediction on clean sentences or the backdoor is not activated on poisoned sentences. 
        correct_clean_orig = np.where(orig_clean_preds == target_label)[0]
        correct_poison_orig = np.where(orig_poisoned_preds == target_label)[0]
        clean_texts = [clean_texts[i] for i in range(len(clean_texts)) if i in correct_clean_orig]
        poisoned_texts = [poisoned_texts[i] for i in range(len(poisoned_texts)) if i in correct_poison_orig]        
        orig_CACC = sum(orig_clean_preds==target_label)*1.0/len(orig_clean_preds)
        orig_ASR = sum(orig_poisoned_preds==target_label)*1.0/len(orig_poisoned_preds)

        #for the sentences with correct initial prediction, go on with the paraphrasing process
        clean_texts, pred_clean, rephrase_clean = batch_process(mid, clean_texts, 'clean', prompt)
        poisoned_texts, pred_poison, rephrase_poison = batch_process(mid, poisoned_texts, 'poison', prompt)

        #pass the paraphrased version to the model
        clean_logits, clean_preds= predict_r6(tokenizer, embedding_model, subject_model, rephrase_clean, clean_labels, max_input_len, emb_arch=='DistilBERT')
        poisoned_logits, poisoned_preds = predict_r6(tokenizer, embedding_model, subject_model, rephrase_poison, poisoned_labels, max_input_len, emb_arch=='DistilBERT')
        
        #compute the metrics
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

    if if_fuzz: #return this when debugging the fuzzing process
        return orig_CACC, CACC, ASR, TP, TN, FP, FN, prec, recall, F1, clean_texts, rephrase_clean
        
    #return info needed for the report table
    print (mid, target_label,model_info[2], emb_arch, model_info[5], model_info[7], model_info[8], model_info[9], len(clean_texts), len(poisoned_texts), orig_CACC, CACC, orig_ASR, ASR, low_TP, low_TN, low_FP, low_FN, low_prec, low_recall, low_F1, TP, TN, FP, FN, FAR, FRR, prec, recall, F1, min, sec)
    return (mid, target_label,model_info[2], emb_arch, model_info[5], model_info[7], model_info[8], model_info[9], len(clean_texts), len(poisoned_texts), orig_CACC, CACC, orig_ASR, ASR, low_TP, low_TN, low_FP, low_FN, low_prec, low_recall, low_F1, TP, TN, FP, FN, FAR, FRR, prec, recall, F1, min, sec)



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
    # dump results into a table for convenience
    exp_results = [('mid', 'target_label', 'model_arch', 'embedding', 'dataset', 'trigger_type', 'trigger', 'PR', 'n_clean', 'n_poisoned', 'orig_CACC', 'CACC', 'orig_ASR', 'ASR', 'low_TP', 'low_TN', 'low_FP', 'low_FN', 'low_prec',  'low_recall', 'low_F1', 'TP', 'TN', 'FP', 'FN', 'FAR', 'FRR', 'prec', 'recall', 'F1', 'min', 'sec')]
    for mid in [21, 22]: #choose which models you want to work on
        try:
            prompt = PROMPT[mid]
            prompt = 'Paraphrase the sentences and make them ' + prompt + '. The reply format is "<sentence index> --**-- <one paraphrased sentence>" in one line for each sentence.'
            res = main(mid, TROJAI_R6_DATASET_DIR, prompt=prompt)
            if len(res) > 0:
               exp_results.append(res)
        except:
            traceback.print_exc()
            continue
        try:
            df = pd.DataFrame(exp_results)
            df.to_csv('rebuttal/trojai_rockstar_abl_kw.csv') #change the filename as needed
        except:
            traceback.print_exc() #print error msg
            print(exp_results)
            break









