from transformers import T5Tokenizer, T5ForConditionalGeneration
import pandas as pd
import nltk
from nltk.parse import CoreNLPParser
from zss import simple_distance, Node
from nltk.util import ngrams
import gensim.downloader as api
from nltk.tokenize import word_tokenize
import spacy
from nltk import Tree
from nltk.corpus import wordnet as wn
import numpy as np
import random
from os import environ
#from Bard import Chatbot
import openai
import traceback
from textstat import textstat
from textstat import flesch_reading_ease, sentence_count
from nlp_ami import  fuzzing_reward, insert_trigger
import sys
import time

random.seed(0)
np.random.seed(0)

LIST = []
LF1, GF1 = 0.0, 0.0
SEEN = set()
TRIAL = 0
RECORD = {}

def read_data():
    file = '/data/share/trojai/trojai-round6-v2-dataset/models/id-00000012/clean_data.csv'
    data  = pd.read_csv(file).values
    text = [d[1] for d in data]
    labels = [d[2] for d in data]
    return text, labels

def ask_model(prompt):
    print(prompt)
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
        #traceback.print_exc(file=sys.stderr)
        res = ''
    return res


def mutate(prompt, i=3):
    #instruction = f'Generate 20 phrases by making 3 editions on "{prompt}".  Each edition can be either adding or removing or changing one word. The generated phrases should be as diverse as possible. The reply format is ^<generated phrase>^ in one line.'
    instruction = f'Generate 20 phrases. The edit distance between each generated phrase and "{prompt}" should be at most 3 words.  The reply format is ^<generated phrase>^ in one line.'
    reply = ask_model(instruction).strip().split('\n')
    if len(reply) == 0:
        return []
    mutations = [r.strip()[1:-1] for r in reply]
    print(mutations)
    return mutations

def gpt_mutate(instruction, i=3):
    instruction = instruction + ' The reply format is ^<generated phrase>^ in one line.'
    reply = []
    while len(reply) == 0:
        reply = ask_model(instruction).strip().split('\n')
    mutations = [s.rstrip()[s.find('^')+1:-1] if '^' in s else s.strip() for s in reply]
    print(mutations)
    return mutations

def fancy_mutate(cur_pmpt):
    global RECORD, SEEN
    #keyword
    #instruct = f'''Given the seed phrase "{cur_pmpt}",  generate 10 diverse new phrases with each containing at least 3 words from the seed phrase, where they can be the exact same words, or synonyms, or antonyms. The reply must include examples of using antonyms.'''
    kw_mutations = [] # gpt_mutate(instruct)
    instruct = f'''Given the seed phrase "{cur_pmpt}",  generate 10 diverse new phrases using the same or similar structure\
    but with completely different meanings. The new phrases should be appropriate to describe a sound or language or essays.'''
    struct_mutations = gpt_mutate(instruct)
    #instruct = f'Generate 10 phrases by crossover (i.e.,exchanging the words) the phrases in group 1 and group 2. Group 1: '+'\n'.join(kw_mutations)+' Group 2: '+'\n'.join(RECORD.keys()) #TODO: record with best f1
    evo_mutations = [] #gpt_mutate(instruct)
    return kw_mutations + struct_mutations + evo_mutations


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


def fuzz_step(mid, cur_prompt, prefix, formats):
    global LIST, RECORD, GF1, LF1
    full_prompt = prefix + ' ' + cur_prompt + ' .' + formats
    try:
        prec, recall, f1, clean_texts, reph_clean =  fuzzing_reward(mid, prompt=full_prompt)
    except KeyboardInterrupt:
        exit(0)
    except:
        print('parsing error')
        traceback.print_exc(file=sys.stderr)
        return
    better = int(f1>GF1 or f1>0.95)
    LF1 = max(f1,LF1)
    if better:
        LIST.append(cur_prompt)
        RECORD[cur_prompt] = (prec, recall, f1)
        print('appended')
    print(f'current prompt:, {mid}, {cur_prompt}, {prec}, {recall}, {f1}/{GF1}')


def extract_max():
    global LIST, RECORD
    max_idx = 0
    max_f1 = 0
    for idx, key in enumerate(LIST):
        if RECORD[key][2] > max_f1:
            max_f1 = RECORD[key][2]
            max_idx = idx
    return max_idx

def fuzz(mid, reph_seed, prefix, formats):
    global GF1, LF1, RECORD, TRIAL, LIST, SEEN
    TRIAL = 0
    GF1, LF1 = 0.0, 0.0
    RECORD = {}
    LIST = []
    SEEN = set()
    start = time.time()
    for seed in reph_seed:
        print(seed)
        SEEN.add(seed)
        fuzz_step(mid, seed, prefix, formats)
        GF1 = max(GF1, LF1)
    flag = ((GF1 >= 0.95 ) or TRIAL >= 300)
    while len(LIST) and not flag:
        #rand_idx = -1
        #random_pick = (random.random()>0.5)
        #if random_pick:
        #    rand_idx = random.randint(0, len(LIST)-1)
        rand_idx = extract_max()
        cur_prompt = LIST[rand_idx]
        del LIST[rand_idx] #deque with highest f1
        mutations = fancy_mutate(cur_prompt)
        for prompt in mutations:
            TRIAL += 1
            if prompt in SEEN:
                continue
            SEEN.add(prompt)
            try:
                fuzz_step(mid, prompt, prefix, formats)
            except KeyboardInterrupt:
                break
            except:
                traceback.print_exc(file=sys.stderr)
                continue
            df = pd.DataFrame.from_dict(RECORD, orient='index', columns=['precision', 'recall', 'f1'])
            df.to_csv(f'rebuttal/{mid}_rockstar_abl_kw.csv')
        GF1 = max(GF1, LF1)
        print(GF1, LF1)
        flag = ((GF1 >= 0.95) or TRIAL >= 300)
        #df = pd.DataFrame.from_dict(RECORD, orient='index', columns=['precision', 'recall', 'f1'])
        #df.to_csv(f'prompts/{mid}_rockstar.csv')

    end = time.time()
    spend_min = (end - start)/60
    spend_sec = (end - start)%60
    print(f'The fuzzing process takes {spend_min} minutes and {spend_sec} seconds for model {mid}')


if __name__ == '__main__':
    prefix = 'Paraphrase the sentences and make them'
    prompt = ['sound like a rockstar']
    #prompt = ['sound like an old woman']
    formats = 'The reply format is "<sentence index> --**-- <one paraphrased sentence>" in one line for each sentence.'
    for mid in list(range(12, 24))+list(range(36, 48)): #
        try:
            fuzz(mid, prompt, prefix, formats)
        except KeyboardInterrupt:
            continue
        except:
            print(mid)
            traceback.print_exc(file=sys.stderr)


