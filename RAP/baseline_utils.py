import torch
import numpy as np
import os
import random
import warnings
import time
import yaml
import argparse
import traceback
import transformers
import sys
import logging
import math
import json
import glob
import pandas as pd
import torch.nn.functional as F


def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_embedding(TROJAI_DIR, arch_name, flavor):
    backbone_model = None
    print(TROJAI_DIR, arch_name, flavor)
    if 'round6' in TROJAI_DIR and arch_name == 'DistilBERT':
        #backbone_filepath = os.path.join(TROJAI_DIR,'embeddings/DistilBERT-distilbert-base-uncased.pt')
        #backbone_model = torch.load(backbone_filepath).cuda()
        tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        backbone_model = transformers.DistilBertModel.from_pretrained('distilbert-base-uncased').cuda()
    elif 'round6' in TROJAI_DIR and  arch_name == 'GPT-2':
        #backbone_filepath = os.path.join(TROJAI_DIR,'embeddings/GPT-2-gpt2.pt')
        #backbone_model = torch.load(backbone_filepath).cuda()
        tokenizer = transformers.GPT2Tokenizer.from_pretrained('gpt2')
        backbone_model = transformers.GPT2Model.from_pretrained('gpt2').cuda()
    elif arch_name == 'RoBERTa':
        tokenizer = transformers.AutoTokenizer.from_pretrained(flavor, use_fast=True, add_prefix_space=True)
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(flavor, use_fast=True)
    if not hasattr(tokenizer, 'pad_token') or tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if arch_name == 'MobileBERT':
        max_input_length = tokenizer.max_model_input_sizes[tokenizer.name_or_path.split('/')[1]]
    else:
        max_input_length = tokenizer.max_model_input_sizes[tokenizer.name_or_path]

    return backbone_model, tokenizer, max_input_length


def read_r6_eg_directory(examples_dirpath, target_label):
    fns = glob.glob(examples_dirpath+'/*class_%d*.txt'%target_label)
    fns.sort()
    texts, labels = [], []
    for fn in fns:
        with open(fn,'r') as fh:
            text = fh.read()
            text = text.strip('\n')
            texts.append(text)
        label = int(fn.split('_')[-3])
        labels.append(label)
    return texts, labels


def read_r6_generated(DATA_DIR, mid, mname, split='poisoned'):
    fp = f'{DATA_DIR}/models/{mname}/{split}_data.csv'
    df = pd.read_csv(fp).values
    text = [d[1] for d in df]
    labels = [d[2] for d in df]
    return text, labels


def read_config(DATA_DIR, mname):
    config_path =f'{DATA_DIR}/models/{mname}/config.json'
    f = open(config_path, 'r')
    config = json.load(f)
    model_info = {'poisoned': config['poisoned'], \
                  'master_seed': int(config['master_seed']), \
                   'arch': config['model_architecture'], \
                   'emb': config['embedding'], \
                   'emb_flavor': config['embedding_flavor'], \
                   'source_dataset': config['source_dataset'], \
                   'target_label': int(config['triggers'][0]['target_class']), \
                   'trigger_type': config['triggers'][0]['type'], \
                   'trigger_text': config['triggers'][0]['text'], \
                   'position':  config['triggers'][0]['fraction']}
    f.close()
    return model_info


def predict_r6_RAP(tokenizer, embedding, classification_model, texts, labels, max_input_length, cls_token_is_first=False):
    print('calling predict_r6_RAP')
    if cls_token_is_first:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        tokenizer.add_special_tokens({'eos_token': '[EOS]]'})
    use_amp = False
    all_logits = []
    all_embeddings = []
    all_middle = []
    all_preds = []
    n_correct, n_sample = 0, 0
    #first_part, rnn = break_models(classification_model)
    for idx, (text, label)  in enumerate(zip(texts, labels)):
        if len(text.strip()) == 0:
            text = tokenizer.pad_token
        results = tokenizer(text, max_length=max_input_length - 2, padding=True, truncation=True, return_tensors="pt")
        # extract the input token ids and the attention mask
        input_ids = results.data['input_ids'].cuda()
        if input_ids.numel() == 0:
            print(text, input_ids)
        attention_mask = results.data['attention_mask'].cuda()

        # convert to embedding
        #with torch.no_grad():
        if use_amp:
            with torch.cuda.amp.autocast():
                embedding_vector = embedding(input_ids, attention_mask=attention_mask, output_attentions=True)[0]
        else:
            embedding_vector = embedding(input_ids, attention_mask=attention_mask, output_attentions=True)[0]

        if cls_token_is_first: #TODO
            embedding_vector = embedding_vector[:, 0, :]
        else: #TODO for GPT remove padding
            embedding_vector = embedding_vector[:, -1, :]

        #embedding_vector = embedding_vector.cpu().numpy()
        #all_embeddings.append(embedding_vector)
        #embedding_vector = np.expand_dims(embedding_vector, axis=0)
        #print(embedding_vector[:5])

        #embedding_vector = torch.from_numpy(embedding_vector).cuda()
        embedding_vector = embedding_vector.unsqueeze(0)
        if use_amp:
            with torch.cuda.amp.autocast():
                logits = classification_model(embedding_vector).cpu()#.detach().numpy()

        else:
            logits = classification_model(embedding_vector).cpu()#.detach().numpy()

        all_logits.append(logits)
        #all_middle.append(middle)
        #sentiment_pred = np.argmax(logits)
        #all_preds.append(sentiment_pred)
        #n_sample += 1
        #if sentiment_pred == label:
        #    n_correct += 1

    #print(n_correct, n_sample, n_correct/n_sample)
    all_logits = torch.vstack(all_logits)
    #all_embeddings = np.concatenate(all_embeddings, axis=0)
    #all_preds = np.array(all_preds)
    return all_logits

def predict_r6(tokenizer, embedding, classification_model, texts, labels, max_input_length, cls_token_is_first=False):
    if cls_token_is_first:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        tokenizer.add_special_tokens({'eos_token': '[EOS]]'})
    use_amp = False
    all_logits = []
    all_embeddings = []
    all_middle = []
    all_preds = []
    n_correct, n_sample = 0, 0
    #first_part, rnn = break_models(classification_model)
    for idx, (text, label)  in enumerate(zip(texts, labels)):
        if len(text.strip()) == 0:
            text = tokenizer.pad_token
        results = tokenizer(text, max_length=max_input_length - 2, padding=True, truncation=True, return_tensors="pt")
        # extract the input token ids and the attention mask
        input_ids = results.data['input_ids'].cuda()
        if input_ids.numel() == 0:
            print(text, input_ids)
        attention_mask = results.data['attention_mask'].cuda()

        # convert to embedding
        with torch.no_grad():
            if use_amp:
                with torch.cuda.amp.autocast():
                    embedding_vector = embedding(input_ids, attention_mask=attention_mask, output_attentions=True)[0]
            else:
                embedding_vector = embedding(input_ids, attention_mask=attention_mask, output_attentions=True)[0]

            if cls_token_is_first: #TODO
                embedding_vector = embedding_vector[:, 0, :]
            else: #TODO for GPT remove padding
                embedding_vector = embedding_vector[:, -1, :]

            #embedding_vector = embedding_vector.cpu().numpy()
            #all_embeddings.append(embedding_vector)
            #embedding_vector = np.expand_dims(embedding_vector, axis=0)
            #print(embedding_vector[:5])

        #embedding_vector = torch.from_numpy(embedding_vector).cuda()
        embedding_vector = embedding_vector.unsqueeze(0)
        if use_amp:
            with torch.cuda.amp.autocast():
                logits = classification_model(embedding_vector).cpu().detach().numpy()

        else:
            logits = classification_model(embedding_vector).cpu().detach().numpy()

        all_logits.append(logits)
        #all_middle.append(middle)
        sentiment_pred = np.argmax(logits)
        all_preds.append(sentiment_pred)
        #n_sample += 1
        #if sentiment_pred == label:
        #    n_correct += 1

    #print(n_correct, n_sample, n_correct/n_sample)
    #all_logits = torch.vstack(all_logits)
    #all_embeddings = np.concatenate(all_embeddings, axis=0)
    all_logits = np.concatenate(all_logits, axis=0)
    #all_preds = np.array(all_preds)
    return all_logits, all_preds


def metrics(TP, TN, FP, FN):
    prec, recall, F1 = 0.0, 0.0, 0.0
    if TP+FP>0:
        prec = 1.0*TP/(TP+FP)
    if TP+FN>0:
        recall = 1.0*TP/(TP+FN)
    if prec+recall>0:
        F1 = 2*prec*recall/(prec+recall)
    return prec, recall, F1
