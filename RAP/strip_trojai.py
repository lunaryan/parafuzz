import random
import numpy as np
import os
import codecs
from tqdm import tqdm
import random
import torch
import time
from transformers import BertTokenizer
from transformers import BertForSequenceClassification, AdamW
from transformers import AdamW
import torch.nn as nn
from functions import *
import argparse
import traceback
from baseline_utils import *
import sys
sys.path.append('../')
import model_factories

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
        for line in tqdm(all_data):
            text, label = line.split('\t')
            if int(label.strip()) == chosen_label:
                text_list.append(text.strip())
                label_list.append(int(label.strip()))

    if total_num is not None:
        text_list = text_list[:total_num]
        label_list = label_list[:total_num]
    return text_list, label_list


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


# calculate tf-idf
def TFIDF(num_text, text_list, vocab):
    TF = np.zeros((num_text, len(vocab)))
    for t in range(num_text):
        for w in text_list[t]:
            if w in vocab:
                TF[t][vocab.index(w)] += 1
        for tf in range(len(TF[t])):
            TF[t][tf] = TF[t][tf] / len(text_list[t])

    idf = np.zeros(len(vocab))
    for i in range(num_text):
        for v in vocab:
            if v in text_list[i]:
                idf[vocab.index(v)] += 1

    TF_IDF = np.zeros((num_text, len(vocab)))
    for k in range(len(idf)):
        idf[k] = np.log(num_text / idf[k]) + 1
    for tt in range(num_text):
        for w_index in range(len(TF[tt])):
            TF_IDF[tt][w_index] = TF[tt][w_index] * idf[w_index]
    return TF_IDF


def calculate_entropy(output_probs):
    entropy = np.array(torch.sum(- output_probs * torch.log(output_probs), dim=1).cpu())
    return entropy


# create copies and perturb
def perturb_sentences(sentences_list, replace_ratio, vocab_list, tf_idf):
    perturbed_list = []
    for sentence in sentences_list:
        words_list = sentence.split(' ')
        held_out_sample_tfidf = tf_idf[random.choice(list(range(len(tf_idf)))), :]
        tfidf_sorted_inds = np.argsort(- held_out_sample_tfidf)
        replaced_inds_list = random.sample(list(range(len(words_list))), min(len(tfidf_sorted_inds), max(int(len(words_list) * replace_ratio), 1)))
        for i in range(len(replaced_inds_list)):
            replace_ind = replaced_inds_list[i]
            candidate_word = vocab_list[tfidf_sorted_inds[i]]
            words_list[replace_ind] = candidate_word
        new_sentence = ' '.join(words_list).strip()
        perturbed_list.append(new_sentence)
    return perturbed_list


def check_randomness_of_strip(model, tokenizer, embedding_model, text_list, text_labels, emb_arch, max_input_len, vocab_list, tf_idf,
                           batch_size, replace_ratio, perturbation_number,
                           protect_label, device, seed=1234):
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
    output_randomness_list = []
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
            batch_logits, output_label = predict_r6(tokenizer, embedding_model, model, batch_sentences, batch_labels, max_input_len, emb_arch=='DistilBERT')
            batch_logits = torch.FloatTensor(batch_logits).cuda()
            #batch = tokenizer(batch_sentences, padding=True, truncation=True, return_tensors="pt").to(device)
            #ori_outputs = model(**batch)
            ori_entropy = calculate_entropy(torch.softmax(batch_logits, dim=1))
            batch_entropy = np.zeros_like(ori_entropy)
            for pn in range(perturbation_number):
                perturbed_batch_sentences = perturb_sentences(batch_sentences, replace_ratio, vocab_list,
                                                              tf_idf)
                perturb_logits, perturb_label = predict_r6(tokenizer, embedding_model, model, perturbed_batch_sentences, batch_labels, max_input_len, emb_arch=='DistilBERT')
                perturb_logits = torch.FloatTensor(perturb_logits).cuda()
                entropy = calculate_entropy(torch.softmax(perturb_logits, dim=1))
                batch_entropy += entropy
            batch_entropy /= perturbation_number
            for j in range(len(batch_logits)):
                if torch.argmax(batch_logits[j, :]) == protect_label:
                    output_randomness_list.append(batch_entropy[j])

    return output_randomness_list, output_label


def main(mid, batch_size=1024, replace_ratio=0.7, perturbation_number=20):
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
    valid_texts, valid_labels = read_r6_eg_directory(clean_path, victim_label)
    valid_targets, valid_target_labels = read_r6_eg_directory(clean_path, target_label)
    vocab_list = []
    # delete symbols when calculating tf-idf
    symbols = ['.', ',', '!', ':', '?', '"', ';', '...', '(', ')', '/', '~', "'"]
    for l in valid_texts:
        l_split = l.split(' ')
        for w in l_split:
            if w not in symbols and w not in vocab_list:
                vocab_list.append(w)
    print("Length of vocabulary: ", len(vocab_list))
    tf_idf = TFIDF(len(valid_texts), valid_texts, vocab_list)
    print(tf_idf.shape)
    # get threshold
    train_randomness_list = check_randomness_of_strip(subject_model, tokenizer, embedding_model, valid_targets, valid_target_labels, emb_arch, max_input_len, vocab_list, tf_idf,
                                                   batch_size, replace_ratio, perturbation_number,
                                                   target_label, 'cuda', int(model_info['master_seed']))
    # allow 0.5%, 1%, 3%, 5% FRRs on training samples
    percent_list = [0.5, 1, 3, 5]
    threshold_list = []
    for percent in percent_list:
        threshold_list.append(np.nanpercentile(train_randomness_list, percent))

    # get randomness in clean samples
    clean_randomness_list, clean_label = check_randomness_of_strip(subject_model, tokenizer, embedding_model, clean_test_texts, clean_test_labels, emb_arch, max_input_len, vocab_list, tf_idf,
                                                   batch_size, replace_ratio, perturbation_number,
                                                   target_label, 'cuda', int(model_info['master_seed']))
    poison_randomness_list, poison_label = check_randomness_of_strip(subject_model, tokenizer, embedding_model, poisoned_test_texts, poisoned_test_labels, emb_arch, max_input_len, vocab_list, tf_idf,
                                                    batch_size, replace_ratio, perturbation_number,
                                                    target_label, 'cuda', int(model_info['master_seed']))
    maxf1 = 0.0
    best_res = None
    for i in range(len(percent_list)):
        thr = threshold_list[i]
        print('FRR on clean held out validation samples (%): ', percent_list[i], ' | Threshold: ', thr)
        FRR = np.sum(clean_randomness_list < thr) / len(clean_randomness_list)#clean -> poisoned
        print('FRR on clean testing samples (%): ',FRR)
        FAR = 1 - np.sum(poison_randomness_list < thr) / len(poison_randomness_list)  #poisoned -> clean
        print('FAR on testing samples (%): ', FAR)
        TP = np.sum((poison_randomness_list < thr))
        FP = np.sum((clean_randomness_list < thr))
        TN = np.sum((clean_randomness_list >= thr))
        FN = np.sum((poison_randomness_list >= thr))
        prec, recall, f1 = metrics(TP, TN, FP, FN)
        if f1 > maxf1:
            maxf1 = f1
            best_res = (TP, TN, FP, FN, prec, recall, f1, FRR, FAR)
        print(TP, TN, FP, FN, prec, recall, f1)
    print('BEST Result:', best_res)

if __name__ == '__main__':
    start = time.time()
    for mid in list(range(12, 24)) + list(range(36, 48)):
        #for mid in [13, 15, 16, 17, 18, 19, 21, 22, 36, 44, 46]:
        try:
            main(mid)
        except:
            traceback.print_exc()
            continue
    end = time.time()
    avg_time = (end-start)/24
    avg_min = avg_time // 60
    avg_sec = avg_time % 60
    print('strip average time', avg_min, 'min', avg_sec, 'seconds')








