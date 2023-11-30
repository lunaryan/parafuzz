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
import os
import traceback
from baseline_utils import *
import sys
sys.path.append('../')
import model_factories
import time

torch.backends.cudnn.enabled = False
os.environ["CUDA_VISIBLE_DEVICES"]='6'
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


def construct_rap_iter(trigger_inds_list, protect_label, model, tokenizer, embedding_model, emb_arch, max_input_len, ori_batch, poisoned_batch, ori_labels, probs_range_list, LR, ori_norms_list, scale_factor=1):
    ori_logits =  predict_r6_RAP(tokenizer, embedding_model, model, ori_batch, ori_labels, max_input_len, emb_arch=='DistilBERT')
    ori_out_probs = torch.softmax(ori_logits, dim=1)[:, protect_label]
    poisoned_logits = predict_r6_RAP(tokenizer, embedding_model, model, poisoned_batch, ori_labels, max_input_len, emb_arch=='DistilBERT')
    #poisoned_logits = torch.FloatTensor(poisoned_logits)
    poisoned_out_probs = torch.softmax(poisoned_logits, dim=1)[:, protect_label]
    diff = poisoned_out_probs - ori_out_probs
    loss = scale_factor * torch.mean((diff > probs_range_list[0]) * (diff - probs_range_list[0])) + torch.mean((diff < probs_range_list[1]) * (probs_range_list[1] - diff))
    acc_num, acc = binary_accuracy(ori_logits, torch.LongTensor(ori_labels))
    loss.backward()
    if emb_arch == 'DistilBERT':
        grad = embedding_model.embeddings.word_embeddings.weight.grad
    else:
        grad = embedding_model.wte.weight.grad
    for i in range(len(trigger_inds_list)):
        trigger_ind = trigger_inds_list[i]
        ori_norm = ori_norms_list[i]
        if emb_arch == 'DistilBERT':
            embedding_model.embeddings.word_embeddings.weight.data[trigger_ind, :] -= LR * grad[trigger_ind, :]
            embedding_model.embeddings.word_embeddings.weight.data[trigger_ind, :] *= ori_norm / embedding_model.embeddings.word_embeddings.weight.data[trigger_ind, :].norm().item()
        else:
            embedding_model.wte.weight.data[trigger_ind, :] -= LR * grad[trigger_ind, :]
            embedding_model.wte.weight.data[trigger_ind, :] *= ori_norm / embedding_model.wte.weight.data[trigger_ind, :].norm().item()

    del grad
    # you can also uncomment following line, but we follow existing Embedding Poisoning Method
    # to get faster convergence
    # model.zero_grad()
    return embedding_model, loss, acc_num


def construct_rap(trigger_inds_list, trigger_words_list, protect_label, model, tokenizer, embedding_model, emb_arch, max_input_len, train_text_list, train_label_list, \
                    probs_range_list, batch_size, LR, device, ori_norms_list, scale_factor):
    epoch_loss = 0
    epoch_acc_num = 0
    total_train_len = len(train_text_list)

    if total_train_len % batch_size == 0:
        NUM_TRAIN_ITER = int(total_train_len / batch_size)
    else:
        NUM_TRAIN_ITER = int(total_train_len / batch_size) + 1

    for i in range(NUM_TRAIN_ITER):
        ori_batch_sentences = train_text_list[i * batch_size: min((i + 1) * batch_size, total_train_len)]
        ori_batch_labels = train_label_list[i * batch_size: min((i + 1) * batch_size, total_train_len)]
        no_float_sent, no_float_label =  [], []
        for sent, label in zip(ori_batch_sentences, ori_batch_labels):
            if type(sent) == type('hello'):
                no_float_sent.append(sent)
                no_float_label.append(label)
        ori_batch_sentences, ori_batch_labels = no_float_sent, no_float_label
        #ori_labels = torch.from_numpy(np.array(ori_batch_labels))
        #ori_labels = ori_labels.type(torch.LongTensor).to(device)
        #ori_batch = tokenizer(ori_batch_sentences, max_length=max_input_length - 2, padding=True, truncation=True, return_tensors="pt").cuda()
        poisoned_batch_sentences = rap_poison(ori_batch_sentences, trigger_words_list, trigger_type='word')
        #poisoned_batch = tokenizer(poisoned_batch_sentences, padding=True, truncation=True, return_tensors="pt").to(device)
        embedding_model, loss, acc_num = construct_rap_iter(trigger_inds_list, protect_label, model, tokenizer, embedding_model, emb_arch, max_input_len, ori_batch_sentences, poisoned_batch_sentences,\
                                                                  ori_batch_labels, probs_range_list, LR, ori_norms_list, scale_factor)
        epoch_loss += loss.item() * len(ori_batch_sentences)

        epoch_acc_num += acc_num
    return embedding_model, epoch_loss / total_train_len, epoch_acc_num / total_train_len


# RAP defense procedure
def rap_defense(valid_targets, valid_target_labels, trigger_words_list, trigger_inds_list, ori_norms_list, protect_label,
                probs_range_list, model, tokenizer, embedding_model, emb_arch, max_input_len, batch_size, epochs,
                lr, device, seed, scale_factor, save_model=True, save_path=None):
    print('Seed: ', seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    for epoch in range(epochs):
        print("Epoch: ", epoch)
        model.eval()
        embedding_model, injected_train_loss, injected_train_acc = construct_rap(trigger_inds_list, trigger_words_list, protect_label, model, tokenizer,
                                                                       embedding_model, emb_arch, max_input_len,
                                                                        valid_targets, valid_target_labels, probs_range_list, batch_size,
                                                                       lr, device, ori_norms_list, scale_factor)

        print(f'\tConstructing Train Loss: {injected_train_loss:.3f} | Constructing Train Acc: {injected_train_acc * 100:.2f}%')

        if save_model:
            os.makedirs(save_path, exist_ok=True)
            embedding_model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)


# This is used to insert rap trigger
def rap_poison(text_list, trigger_words_list, trigger_type='word', seed=1234):
    random.seed(seed)
    new_text_list = []
    if trigger_type == 'word':
        sep = ' '
    else:
        sep = '.'
    for text in text_list:
        text_splited = text.split(sep)
        for trigger in trigger_words_list:
            # always insert at the first position
            l = 1
            insert_ind = int((l - 1) * random.random())
            text_splited.insert(insert_ind, trigger)
        text = sep.join(text_splited).strip()
        new_text_list.append(text)
    return new_text_list

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
    embedding_model, tokenizer, max_input_len = load_embedding(TROJAI_DIR, emb_arch, emb_flavor)
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
    trigger_inds_list, ori_norms_list = process_model_wth_trigger_trojai(subject_model, tokenizer, embedding_model, emb_arch, trigger_words_list,)
    save_path = TROJAI_DIR+f'models/{mname}/RAP_defended'

    probs_range_list = probability_range.split(' ') # negative u_low and negative u_up in the paper, the format is like '-0.1 -0.3'
    for i in range(len(probs_range_list)):
        probs_range_list[i] = float(probs_range_list[i])
    print('Decreased Probability Range: ', probs_range_list)
    print('Scale Factor: ', scale_factor)
    rap_defense(valid_targets, valid_target_labels, trigger_words_list, trigger_inds_list, ori_norms_list, target_label,
                probs_range_list, subject_model, tokenizer, embedding_model, emb_arch,  max_input_len, batch_size, epochs,
                lr, 'cuda', master_seed, scale_factor, save_model, save_path)


if __name__ == '__main__':
    SEED = 1234
    start = time.time()
    for mid in list(range(13, 24)) + list(range(36, 48)):
        main(mid)
    end = time.time()
    print(end -start)
    avg_time = (end-start)/24
    print(avg_time)
    avg_min = avg_time //60
    avg_sec = avg_time % 60
    print('rap defense average time', avg_min, 'minutes', avg_sec, 'seconds')



