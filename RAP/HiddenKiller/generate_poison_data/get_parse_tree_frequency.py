import pickle, glob, h5py, random, string, codecs, sys, gzip, time, csv
import numpy as np
import nltk
import re
from nltk import Tree
from collections import OrderedDict, Counter

from unidecode import unidecode

index_match=0
temp1="\(ROOT\n {2}\(S\n {4}\(NP(.*\n)*.*\)\n {4}\(VP(.*\n)*.*\)\n {4}\(\..*\)\)\)"
temp2="\(ROOT\n {2}\(S\n {4}\(VP(.*\n)*.*\)\n {4}\(\..*\)\)\)"
temp3="\(ROOT\n {2}\(NP\n {4}\(NP(.*\n)*.*\)\n {4}\(\..*\)\)\)"
temp4="\(ROOT\n {2}\(FRAG\n {4}\(SBAR(.*\n)*.*\)\n {4}\(\..*\)\)\)"
temp5="\(ROOT\n {2}\(S\n {4}\(S(.*\n)*.*\)\n {4}\(\,(.*\n)*.*\)\n {4}\(CC(.*\n)*.*\)\n {4}\(S(.*\n)*.*\)\n {4}\(\..*\)\)\)"
temp6="\(ROOT\n {2}\(S\n {4}\(LST(.*\n)*.*\)\n {4}\(VP(.*\n)*.*\)\n {4}\(\..*\)\)\)"
temp7="\(ROOT\n {2}\(SBARQ\n {4}\(WHADVP(.*\n)*.*\)\n {4}\(SQ(.*\n)*.*\)\n {4}\(\..*\)\)\)"
temp8="\(ROOT\n {2}\(S\n {4}\(PP(.*\n)*.*\)\n {4}\(\,(.*\n)*.*\)\n {4}\(NP(.*\n)*.*\)\n {4}\(VP(.*\n)*.*\)\n {4}\(\..*\)\)\)"
temp9="\(ROOT\n {2}\(S\n {4}\(ADVP(.*\n)*.*\)\n {4}\(NP(.*\n)*.*\)\n {4}\(VP(.*\n)*.*\)\n {4}\(\..*\)\)\)"
temp10="\(ROOT\n {2}\(S\n {4}\(SBAR(.*\n)*.*\)\n {4}\(\,(.*\n)*.*\)\n {4}\(NP(.*\n)*.*\)\n {4}\(VP(.*\n)*.*\)\n {4}\(\..*\)\)\)"
            #'( ROOT ( S ( NP ) ( VP ) ( . ) ) ) EOP'   
            #'( ROOT ( S ( VP ) ( . ) ) ) EOP'                   
            #'( ROOT ( NP ( NP ) ( . ) ) ) EOP', #ing            
            #'( ROOT ( FRAG ( SBAR ) ( . ) ) ) EOP',               
            #'( ROOT ( S ( S ) ( , ) ( CC ) ( S ) ( . ) ) ) EOP',    
            #'( ROOT ( S ( LST ) ( VP ) ( . ) ) ) EOP',                     
            #'( ROOT ( SBARQ ( WHADVP ) ( SQ ) ( . ) ) ) EOP' 
            #'( ROOT ( S ( PP ) ( , ) ( NP ) ( VP ) ( . ) ) ) EOP',               
            #'( ROOT ( S ( ADVP ) ( NP ) ( VP ) ( . ) ) ) EOP' 
            # '( ROOT ( S ( SBAR ) ( , ) ( NP ) ( VP ) ( . ) ) ) EOP' 
PASSIVE_RELS = set(['nsubjpass', 'auxpass', 'csubjpass'])

PAST_VERBS = set(['VBD', 'VBN'])
PRESENT_VERBS = set(['VBZ', 'VBP', 'VBG', 'VB'])
FUTURE_VERBS = set(['MD'])
FUTURE_MODALS = set(["'ll", 'will'])

PRONOUN_POS = set(['PRP', 'PRP$'])
FIRST = set(['i', 'me', 'mine', 'we', 'us', 'ours', 'our'])
SECOND = set(['you', 'yours', 'your'])
THIRD = set(['he', 'she', 'it', 'him', 'her', 'his', 'hers', 'its', 'they', 'them', 'their', 'theirs'])

SYN_POS = set(['NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS',]) | \
    PAST_VERBS | PRESENT_VERBS


# helper method, check if any of given set of nonterminals
# occur in subtree rooted at given node (BFS)
def check_nodes(parent, nt_set):
    nodes = [n for n in parent]
    while len(nodes) > 0:
        node = nodes.pop(0)
        if isinstance(node, Tree):
            l = node.label()
            if l in nt_set:
                return True
            nodes += [n for n in node]
    return False


# classify sentence into {simple, complex, compound, compound-complex, other}
# using modified algorithm 1 in https://homes.cs.washington.edu/~yejin/Papers/emnlp12_stylometry.pdf
# algorithm as-is doesn't work lol. also not happy about compound-complex coverage
# modification: subordinate clauses have to occur at top-level, not anywhere
def algorithm_1(tree):

    productions = tree.productions()
    base_l = productions[0].lhs()
    base_r = productions[0].rhs()
    top_level_l = productions[1].lhs()
    top_level_r = productions[1].rhs()
    if str(base_l) == 'ROOT':

        # base-level S
        if len(base_r) == 1 and str(base_r[0]) == 'S':

            # top-level 
            if str(top_level_l) == 'S':

                c = Counter([str(tag) for tag in top_level_r])

                # compound sentence
                if c['S'] > 1 or (c['S'] > 0 and (c['SBAR'] > 0 or\
                    c['SBARQ'] > 0 or c['SINV'] > 0 or c['SQ'] > 0 or c['VP'] > 0)):

                    # if there's a top-level SBAR, it's a complex-compound sentence
                    if c['SBAR'] == 0 and c['SBARQ'] == 0:
                        return 'COMPOUND'

                    else:
                        return 'COMPOUND-COMPLEX'

                # non-compound
                elif c['VP'] > 0:

                    if c['SBAR'] == 0 and c['SBARQ'] == 0:
                        return 'SIMPLE'

                    else:
                        return 'COMPLEX'


    # it's a fragment or question or something
    return 'OTHER'


# classify sentence into {loose, periodic, other}
# using algorithm 2 in https://homes.cs.washington.edu/~yejin/Papers/emnlp12_stylometry.pdf
# this is VERY noisy
def algorithm_2(tree):

    dep_clause_set = set(['SBAR', 'S'])
    productions = tree.productions()
    base_l = productions[0].lhs()
    base_r = productions[0].rhs()
    top_level_l = productions[1].lhs()
    top_level_r = productions[1].rhs()

    if len(tree) == 1:

        # loop over top level
        for node in tree[0]:
            if isinstance(node, Tree):
                if node.label() != 'VP':
                    if check_nodes(node, dep_clause_set):
                        return 'PERIODIC'

                else:
                    if check_nodes(node, dep_clause_set):
                        return 'LOOSE'

    return 'OTHER'


# check passivity from parse tree
# seems much more restrictive than the regex
def check_passive(deps):

    active_exists = False
    for rel, _, _ in deps:
        if rel in PASSIVE_RELS:
            return 'PASSIVE'
        elif rel == 'nsubj':
            active_exists = True

    if active_exists:
        return 'ACTIVE'
    else:
        return 'OTHER'


# check tense by looking at top-most verb tense in parse tree
def get_tense(tree):

    # find top VP by BFS
    nodes = [n for n in tree]
    vp = None
    while len(nodes) > 0:
        node = nodes.pop(0)
        if isinstance(node, Tree):
            if node.label() == 'VP':
                vp = node
                break
            nodes += [n for n in node]

    # no verb phrase in sentence... 
    if vp == None:
        return 'UNK'

    # look at the immediate children of the VP
    for child in vp:
        if child.label() in FUTURE_VERBS:
            if child[0] in FUTURE_MODALS:
                return 'FUTURE'

        elif child.label() in PAST_VERBS: 
            return 'PAST'

        elif child.label() in PRESENT_VERBS:
            return 'PRESENT'

    # if anything falls thru the above filters...
    return 'UNK'


# look at all pronouns in sentence
def get_pov(tokens, pos_tags):

    c = Counter()
    for idx, pos in enumerate(pos_tags):

        #'I' not detected as PRP but instead as LS or FW... 
        pro = tokens[idx]
        if pos in PRONOUN_POS or pro == 'i':
            if pro in FIRST:
                c['FIRST'] += 1
            elif pro in SECOND:
                c['SECOND'] += 1
            elif pro in THIRD:
                c['THIRD'] += 1

    
    if len(c) == 0:
        return 'UNK'

    # first person dominates, then second, then third
    if c['FIRST'] > 0:
        return 'FIRST'
    elif c['SECOND'] > 0:
        return 'SECOND'
    else:
        return 'THIRD'


# check if it's a question or not by looking at last token
def get_question(tokens):
    if tokens[-1] == '?':
        return 'QUESTION'
    else:
        return 'STATEMENT'


# find sentences that differ by only one NN/VB/JJ/RB
def get_synonym_change(data1, data2, min_len=3):

    t1 = data1['tokens']
    t2 = data2['tokens']
    p1 = data1['pos']
    p2 = data2['pos']
    if len(t1) == len(t2) and len(t1) > min_len:

        good_mismatches = 0
        other_mismatches = 0
        for z in range(len(t1)):
            pos1 = p1[z]
            pos2 = p2[z]
            w1 = t1[z]
            w2 = t2[z]

            # if not same POS, this isn't a synonym transformation
            if pos1 != pos2:
                other_mismatches += 1

            # if a noun/verb/adj and text mismatch, this is good
            elif pos1 in SYN_POS:
                if w1 != w2:
                    good_mismatches += 1

            # this is a mismatch of a useless POS tag
            elif w1 != w2:
                other_mismatches += 1

        if good_mismatches == 1 and other_mismatches == 0:
            return 'SYNONYM'

    return 'NOT SYNONYM'


# add transformation labels
def label_sentence(data):

    # add high-level syntactic structure info
    data['parse'] = data['parse'].strip()
    tree = Tree.fromstring(data['parse'])
    class1 = algorithm_1(tree)
    class2 = algorithm_2(tree)
    data['simple_or_compound'] = class1
    data['loose_or_periodic'] = class2

    # add simpler labels from before (e.g., passive, tense, PoV)
    data['active_or_passive'] = check_passive(data['deps'])
    data['tense'] = get_tense(tree)
    data['pov'] = get_pov(data['tokens'], data['pos'])
    data['question'] = get_question(data['tokens'])


# extract parses from corenlp output
def extract_parses(fname):
    f = open(fname, 'r')
    count = 0
    sentences = []
    data = {'tokens':[],'parse':''}
    new_sent = True
    new_pos = False
    new_parse = False
    new_deps = False
    for idx,line in enumerate(f):
        if line.startswith('Sentence #'):
            new_sent = True
            new_pos = False
            new_parse = False
            new_deps = False
            if idx == 0:
                continue

            # label_sentence(data)
            # print ' '.join(data['tokens'])
            # data['label'] = dataset[count]['label']
            sentences.append(data)
            count += 1

            data = {'idx':'','tokens':'', 'parse':''}

        # read original sentence
        elif new_sent:
            # data['sent'] = line.strip()
            new_sent = False
            new_pos = True
            data['idx']=count
            data['tokens']=line.strip()

        # read POS tags
        elif new_pos and line.startswith('[Text='):
            continue
        elif line.startswith('Constituency parse:'):
            new_pos = False
            new_parse = True
        # start reading const parses
        elif new_parse and line.strip() != '':
            new_pos = False
            new_parse = True
            data['parse'] += line

        # start reading deps
        elif line.strip() == '':
            new_parse = False
            new_deps = True

        elif new_deps and line.strip() != '':
            continue


        else:
            new_deps = False

    # add last sentence
    # label_sentence(data)
    # data['label'] = dataset[count]['label']
    sentences.append(data)

    f.close()

    return sentences


# write parses to csv
def write_parses(parses, writer):
    index_match=0
    len=0
    for idx, parse in enumerate(parses):

        data = {}
        # stringify and get rid of encoding issues
        data['idx']=parse['idx']
        data['tokens'] = unidecode(parse['tokens'])
        data['parse'] = unidecode(parse['parse'])
        
        if  re.search(temp10,data['parse'])!= None:    #change your template here
          index_match=index_match+1 
        # data['label'] = parse['label']

        writer.writerow(data)
        len=len+1
    print(index_match/len)

if __name__ == '__main__':

    para_count = 0

    # write fieldnames
    fn = ['idx','tokens','parse']
    ofile = codecs.open('parsed_tree.tsv', 'w', 'utf-8')
    out = csv.DictWriter(ofile, delimiter='\t', fieldnames=fn)
    out.writerow(dict((x,x) for x in fn))

    parses = extract_parses(' YOUR_PATH_TO_STANDFORD_CORENLP_OUTPUT') #change your input file here
    print(index_match/len(parses))

