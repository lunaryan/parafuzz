import random
import numpy as np
import torch
from torch.nn import functional as F
import torch.nn as nn
import codecs
from sklearn.metrics import f1_score
from transformers import BertTokenizer
from transformers import BertForSequenceClassification
from process_data import *
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pandas as pd


def metrics(TP, TN, FP, FN):
    prec, recall, F1 = 0.0, 0.0, 0.0
    if TP+FP>0:
        prec = 1.0*TP/(TP+FP)
    if TP+FN>0:
        recall = 1.0*TP/(TP+FN)
    if prec+recall>0:
        F1 = 2*prec*recall/(prec+recall)
    return prec, recall, F1

# load model
def process_model_only(model_path, device):
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path, return_dict=True, output_hidden_states=False)
    model = model.to(device)
    parallel_model = nn.DataParallel(model)
    return model, parallel_model, tokenizer


# load model, process trigger information
def process_model_wth_trigger(model_path, trigger_words_list, device):
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path, return_dict=True)
    model = model.to(device)
    parallel_model = nn.DataParallel(model)
    trigger_inds_list = []
    ori_norms_list = []
    for trigger_word in trigger_words_list:
        trigger_ind = int(tokenizer(trigger_word)['input_ids'][1])
        trigger_inds_list.append(trigger_ind)
        ori_norm = model.bert.embeddings.word_embeddings.weight[trigger_ind, :].view(1, -1).to(device).norm().item()
        ori_norms_list.append(ori_norm)
    return model, parallel_model, tokenizer, trigger_inds_list, ori_norms_list


def process_model_wth_trigger_trojai(model, tokenizer, embedding_model, emb_arch, trigger_words_list, device='cuda'):
    model = model.to(device)
    #parallel_model = nn.DataParallel(model)
    trigger_inds_list = []
    ori_norms_list = []
    for trigger_word in trigger_words_list:
        #print(trigger_word, tokenizer(trigger_word)['input_ids'])
        trigger_ind = int(tokenizer(trigger_word)['input_ids'][0])
        trigger_inds_list.append(trigger_ind)
        if emb_arch == 'DistilBERT':
            ori_norm = embedding_model.embeddings.word_embeddings.weight[trigger_ind, :].view(1, -1).to(device).norm().item()
        else:
            ori_norm = embedding_model.wte.weight[trigger_ind, :].view(1, -1).to(device).norm().item()
        ori_norms_list.append(ori_norm)
    return trigger_inds_list, ori_norms_list

# calculate binary acc.
def binary_accuracy(preds, y):
    rounded_preds = torch.argmax(preds, dim=1)
    correct = (rounded_preds == y).float()
    acc_num = correct.sum().item()
    acc = acc_num / len(correct)
    return acc_num, acc


# evaluate test accuracy
def evaluate(model, tokenizer, eval_text_list, eval_label_list, batch_size, criterion, device):
    epoch_loss = 0
    epoch_acc_num = 0
    model.eval()
    total_eval_len = len(eval_text_list)

    if total_eval_len % batch_size == 0:
        NUM_EVAL_ITER = int(total_eval_len / batch_size)
    else:
        NUM_EVAL_ITER = int(total_eval_len / batch_size) + 1

    with torch.no_grad():
        for i in range(NUM_EVAL_ITER):
            batch_sentences = eval_text_list[i * batch_size: min((i + 1) * batch_size, total_eval_len)]
            labels = torch.from_numpy(
                np.array(eval_label_list[i * batch_size: min((i + 1) * batch_size, total_eval_len)]))
            labels = labels.type(torch.LongTensor).to(device)
            batch = tokenizer(batch_sentences, padding=True, truncation=True, return_tensors="pt").to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            acc_num, acc = binary_accuracy(outputs.logits, labels)
            epoch_loss += loss.item() * len(batch_sentences)
            epoch_acc_num += acc_num
    return epoch_loss / total_eval_len, epoch_acc_num / total_eval_len


# evaluate test macro F1
def evaluate_f1(model, tokenizer, eval_text_list, eval_label_list, batch_size, criterion, device):
    epoch_loss = 0
    model.eval()
    total_eval_len = len(eval_text_list)

    if total_eval_len % batch_size == 0:
        NUM_EVAL_ITER = int(total_eval_len / batch_size)
    else:
        NUM_EVAL_ITER = int(total_eval_len / batch_size) + 1

    with torch.no_grad():
        predict_labels = []
        true_labels = []
        for i in range(NUM_EVAL_ITER):
            batch_sentences = eval_text_list[i * batch_size: min((i + 1) * batch_size, total_eval_len)]
            labels = torch.from_numpy(
                np.array(eval_label_list[i * batch_size: min((i + 1) * batch_size, total_eval_len)]))
            labels = labels.type(torch.LongTensor).to(device)
            batch = tokenizer(batch_sentences, padding=True, truncation=True, return_tensors="pt").to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            epoch_loss += loss.item() * len(batch_sentences)
            predict_labels = predict_labels + list(np.array(torch.argmax(outputs.logits, dim=1).cpu()))
            true_labels = true_labels + list(np.array(labels.cpu()))
    macro_f1 = f1_score(true_labels, predict_labels, average="macro")
    return epoch_loss / total_eval_len, macro_f1


def load_style_model(device):
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    state = torch.load('StyleAttack/experiments/style-state.pt')
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    return model, tokenizer

def load_hk_model():
    #model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=4)
    model = torch.load('HiddenKiller/poison_bert_ag.pkl')
    model = model.cuda()
    model.eval()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    return model, tokenizer

def read_tsv_data(path, label=None, total_num=None):
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
    return texts, labels