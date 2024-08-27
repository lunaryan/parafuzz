import sys
sys.path.append('../')
import transformers
import torch
import json
import os
import glob
import dbs

def arch_parser(tokenizer_filepath):
    # if 'BERT-bert-base-uncased.pt' in tokenizer_filepath:
    #     arch_name = 'bert'

    if 'DistilBERT' in tokenizer_filepath:
        arch_name = 'distilbert'

    elif 'GPT-2-gpt2.pt' in tokenizer_filepath:
        arch_name = 'gpt2'

    else:
        raise NotImplementedError('Transformer arch not support!')


    return arch_name

def load_models(arch_name,model_filepath,device):
    print(transformers.__version__)
    print(torch.__version__)
    target_model = torch.load(model_filepath).to(device)

    if arch_name == 'distilbert':
        backbone_filepath = os.path.join(dbs.TROJAI_R6_DATASET_DIR,'embeddings/DistilBERT-distilbert-base-uncased.pt')
        backbone_model = torch.load(backbone_filepath).to(device)
        tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        benign_reference_model_filepath = os.path.join(dbs.TROJAI_R6_DATASET_DIR,'models/id-00000006/model.pt')
        benign_model = torch.load(benign_reference_model_filepath).to(device)
    elif arch_name == 'gpt2':
        backbone_filepath = os.path.join(dbs.TROJAI_R6_DATASET_DIR,'embeddings/GPT-2-gpt2.pt')
        backbone_model = torch.load(backbone_filepath).to(device)
        tokenizer = transformers.GPT2Tokenizer.from_pretrained('gpt2')
        benign_reference_model_filepath = os.path.join(dbs.TROJAI_R6_DATASET_DIR,'models/id-00000001/model.pt')
        benign_model = torch.load(benign_reference_model_filepath).to(device)

    else:
        raise NotImplementedError('Transformer arch not support!')


    if not hasattr(tokenizer,'pad_token') or tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return backbone_model,target_model,benign_model, tokenizer

def enumerate_trigger_options():
    label_list = [0,1]
    insert_position_list = ['first_half','second_half']

    trigger_options = []

    for victim_label in label_list:
        for target_label in label_list:
            if target_label != victim_label:
                for position in insert_position_list:
                    trigger_opt = {'victim_label':victim_label, 'target_label':target_label, 'position':position}

                    trigger_options.append(trigger_opt)

    return trigger_options

def load_data(victim_label,examples_dirpath):

    fns = [os.path.join(examples_dirpath, fn) for fn in os.listdir(examples_dirpath) if fn.endswith('.txt')]
    fns.sort()


    victim_data_list = []

    for fn in fns:
        if int(fn.split('_')[-3]) == victim_label:

            with open(fn,'r') as fh:
                text = fh.read()
                text = text.strip('\n')
                victim_data_list.append(text)


    return victim_data_list


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

def read_r7_eg_directory(example_path):
    files = glob.glob(example_path+'/*.txt')
    words, labels = [], []
    for file in files:
        if file.endswith('_tokenized.txt'):
            continue
        original_words = []
        original_labels = []
        with open(file, 'r') as fh:
            lines = fh.readlines()
            for line in lines:
                split_line = line.split('\t')
                word = split_line[0].strip()
                label = split_line[2].strip()
                original_words.append(word)
                original_labels.append(int(label))
        words.append(original_words)
        labels.append(original_labels)
    return words, labels

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

def read_single_file(fn):
    with open(fn, 'r') as fh:
        text = fh.read()
        text = text.strip('\n')
    return text.split('\n')

def read_patterned_file(fn):
    with open(fn, 'r') as fh:
        text = fh.read().strip()
    text = text.split('*** ')
    print(len(text))
    orig, rephrased = [], []
    for t in text:
        tt = t.strip().split('>>> ')
        if len(tt) != 2:
            continue
        orig.append(tt[0])
        rephrased.append(tt[1])

    return orig, rephrased

if __name__ == '__main__':
    read_patterned_file('poisoned_13.txt')
