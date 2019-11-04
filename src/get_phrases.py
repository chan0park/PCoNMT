import os
import codecs
import random
import sys
import nltk
import fasttext
from copy import deepcopy
from nltk.corpus import stopwords

# usage: python get_mwe.py PATH_DATA SRC_LANG PATH_FASTTEXT_BIN FREQ_THRE
path_data = sys.argv[1]
src_lang = sys.argv[2]
tgt_lang = sys.argv[3]
path_fasttext = sys.argv[4]
FREQ_THRE = sys.argv[5] if len(sys.argv[2])>5 else 2

with open(os.path.join(path_data,"train.tok.true."+tgt_lang),"r") as file:
    tgt = file.readlines()
with open(os.path.join(path_data, "train.tok.true."+src_lang), "r") as file:
    src = file.readlines()
with open(os.path.join(path_data, "train.align"),"r") as file:
    align = file.readlines()

tgt_tokens = " ".join(tgt).split()



bigrams = nltk.collocations.BigramAssocMeasures()
bigramFinder = nltk.collocations.BigramCollocationFinder.from_words(tgt_tokens)
bigramFinder.apply_freq_filter(FREQ_THRE)

trigrams = nltk.collocations.TrigramAssocMeasures()
trigramFinder = nltk.collocations.TrigramCollocationFinder.from_words(tgt_tokens)
trigramFinder.apply_freq_filter(FREQ_THRE)

quadgrams = nltk.collocations.QuadgramAssocMeasures()
quadgramFinder = nltk.collocations.QuadgramCollocationFinder.from_words(tgt_tokens)
quadgramFinder.apply_freq_filter(FREQ_THRE)

bigramPMI = list(bigramFinder.score_ngrams(bigrams.pmi))
trigramPMI = list(trigramFinder.score_ngrams(trigrams.pmi))
quadgramPMI = list(quadgramFinder.score_ngrams(quadgrams.pmi))

bigramPMI_dict = {}
trigramPMI_dict = {}
quadgramPMI_dict = {}
for words, pmi in bigramPMI:
    bigramPMI_dict["_".join(words)] = pmi
    
for words, pmi in trigramPMI:
    trigramPMI_dict["_".join(words)] = pmi
    
for words, pmi in quadgramPMI:
    quadgramPMI_dict["_".join(words)] = pmi



def align_line_to_dict(a):
    a = a.split()
    res = {}
    for pair in a:
        s, t = pair.split('-')
        s, t = int(s), int(t)
        try:
            res[s].append(t)
        except:
            res[s] = [t]
    return res

def get_mwe_phrases(src, tgt, align):
    phrases_freq = {}
    phrases_src = {}
    phrases = {}
    for src_sent, tgt_sent, a in zip(src, tgt, align):
        src_sent, tgt_sent = src_sent.split(), tgt_sent.split()
        a_dict = align_line_to_dict(a)
        for src_idx, tgt_idx in a_dict.items():
            if len(tgt_idx) > 1:
                tgt_idx.sort()
                tgt_mwe = "_".join([tgt_sent[idx] for idx in tgt_idx])
                try:
                    phrases[tgt_mwe]['freq'] += 1
                    try:
                        phrases[tgt_mwe]['src'][src_sent[src_idx]] += 1 
                    except:
                        phrases[tgt_mwe]['src'][src_sent[src_idx]] = 1
                except:
                    phrases[tgt_mwe] = {'freq':1, 'src':{src_sent[src_idx]:1}}
    return phrases

phrases = get_mwe_phrases(src, tgt, align)
    


# get english stopwords
# en_stopwords = set(stopwords.words('english'))
en_stopwords = ['the','an','a','ve','ll','i','s','d','m','re']

#function to filter for ADJ/NN bigrams
def rightTypes(ngram):
    if '-pron-' in ngram or 't' in ngram:
        return False
    for word in ngram:
        if word in en_stopwords or word.isspace():
            return False
    acceptable_types = ('JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS')
    second_type = ('NN', 'NNS', 'NNP', 'NNPS')
    tags = nltk.pos_tag(ngram)
    if tags[0][1] in acceptable_types and tags[1][1] in second_type:
        return True
    else:
        return False

#function to filter for trigrams
def rightTypesTri(ngram):
    if '-pron-' in ngram or 't' in ngram:
        return False
    for word in ngram:
        if word in en_stopwords or word.isspace():
            return False
    first_type = ('JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS')
    third_type = ('JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS')
    tags = nltk.pos_tag(ngram)
    if tags[0][1] in first_type and tags[2][1] in third_type:
        return True
    else:
        return False

    
def rightTypesQuad(ngram):
    if '-pron-' in ngram or 't' in ngram:
        return False
    for word in ngram:
        if word in en_stopwords or word.isspace():
            return False
    return True


def check_duplicated(phrase):
    words = phrase.split("_")
    for i in range(len(words)-1):
        if words[i] == words[i+1]:
            return True
    return False

def get_most_freq_src(src_dict):
    max_word, max_freq = "",0
    for word, freq in src_dict.items():
        if freq > max_freq:
            max_word, max_freq = word, freq
    return max_word
    

def filter_phrases(phrases, freq_thre=FREQ_THRE):
    filtered_phrases = deepcopy(phrases)
    for phrase, value in list(phrases.items()):
        words = phrase.split("_") 
        n = len(words)
        
        if n > 4:
            del filtered_phrases[phrase]
            continue            
        if value['freq'] < FREQ_THRE:
            del filtered_phrases[phrase]
            continue
        elif not all([c.isalpha() or c=="_" for c in phrase]):
            del filtered_phrases[phrase]
            continue
        elif check_duplicated(phrase):
            del filtered_phrases[phrase]
            continue
        elif (n==2) and not rightTypes(words):
            del filtered_phrases[phrase]
            continue
        elif (n==3) and not rightTypesTri(words):
            del filtered_phrases[phrase]
            continue  
        elif (n==4) and not rightTypesQuad(words):
            del filtered_phrases[phrase]
            continue  
        try:
            if n == 2:
                pmi = bigramPMI_dict[phrase]
            elif n == 3:
                pmi = trigramPMI_dict[phrase]
            elif n == 4:
                pmi = quadgramPMI_dict[phrase]
        except:
            pmi = 0
            
        if pmi <= 0:
            del filtered_phrases[phrase]
            continue
        else:
            filtered_phrases[phrase]['pmi'] = pmi
        filtered_phrases[phrase]['src_most'] = get_most_freq_src(value['src'])
    return filtered_phrases
            
phrase_filtered = filter_phrases(phrases)


def bool_contain_mwe(a_dict, tgt_sent, phrase_filtered_items):
    tgt_sent = tgt_sent.split()
    for tgt_idx in a_dict.values():
        if len(tgt_idx)>1:
            tgt_idx.sort()
            tgt_mwe = "_".join([tgt_sent[idx] for idx in tgt_idx])
            if tgt_mwe in phrase_filtered_items:
                return True
    return False
            
        
def extract_mwe_cases(src, tgt, align, phrase_filtered_items):
    src_mwe, tgt_mwe = [], []
    for src_sent, tgt_sent, a in zip(src, tgt, align):
        a_dict = align_line_to_dict(a)
        if bool_contain_mwe(a_dict, tgt_sent, phrase_filtered_items):
            src_mwe.append(src_sent)
            tgt_mwe.append(tgt_sent)
    
    assert len(src_mwe) == len(tgt_mwe)
    return src_mwe, tgt_mwe


src_mwe, tgt_mwe = extract_mwe_cases(src, tgt, align, phrase_filtered)


def convert_to_lookup_dict(phrase_filtered):
    res = {}
    for phrase in phrase_filtered.keys():
        words = phrase.split("_")
        try:
            res[words[0]].add(phrase)
        except:
            res[words[0]] = set([phrase])
    return res
phrase_lookup = convert_to_lookup_dict(phrase_filtered)        

mwe_list = [x for v in phrase_lookup.values() for x in v]
with open(os.path.join(path_data,'mwe_list.txt'),'w') as file:
    file.write("\n".join(mwe_list))

with open(os.path.join(path_data,"train.tok.true."+tgt_lang+"_mweonly"),"w") as file:
    file.write("\n".join(tgt_mwe))
with open(os.path.join(path_data, "train.tok.true."+src_lang+"_mweonly"), "w") as file:
    file.write("\n".join(src_mwe))




model = fasttext.load_model(path_fasttext)
mwe_embeddings = [model[w] for w in mwe_list]
with open(os.path.join(path_data,'mwe_list.mwe.vec'),'w') as file:
    file.write("\n".join([mwe_list[i]+" "+" ".join([str(f) for f in mwe_embeddings[i].tolist()]) for i in range(len(mwe_list))]))


def align_line_to_tgt_dict(align_line):
    a = [x.split("-") for x in align_line.split()]
    res = {}
    for s,t in a:
        s,t = int(s), int(t)
        if t in res:
            res[t].add(s)
        else:
            res[t] = set([s])
    return res
    
def concat_mwe(tgt, lookup_dict, align, src):
    def _concat_mwe(tgt_line, align_line, src_line):
        res = []
        tgt_line = tgt_line.split()
        new_align = {i:0 for i in range(len(src_line.split()))}
        align_line = align_line_to_tgt_dict(align_line)
#         align_line = set(align_line.split())
        skip_bi, skip_tri, skip_quat = False, False, False
        for i, word in enumerate(tgt_line):
            if skip_bi:
                skip_bi = False
                continue
            if skip_tri:
                skip_tri = False
                continue
            if skip_quat:
                skip_quat = False
                continue
            src_set = align_line[i] if i in align_line else None
            if word in lookup_dict:
                if (i in align_line and i+1 in align_line) and len(align_line[i]&align_line[i+1])>0:
                    possible_src_idx = align_line[i]&align_line[i+1]
                    word_bi = "_".join(tgt_line[i:i+2])
                    word_tri = "_".join(tgt_line[i:i+3])
                    word_quat = "_".join(tgt_line[i:i+4])
                    if word_quat in lookup_dict[word]:
                        if (i+2 in align_line and i+3 in align_line) and len(possible_src_idx&align_line[i+2]&align_line[i+3])>0:
                            possible_src_idx = list(possible_src_idx&align_line[i+2]&align_line[i+3])
#                         if (i+2 in align_line and i+3 in align_line) and align_line[i+1] == align_line[i+2] and align_line[i+2]==align_line[i+3]:
                            src_idx = possible_src_idx[0] if len(possible_src_idx) == 1 else random.choice(possible_src_idx)
                            res.append(word_quat)
                            new_align[src_idx] = 4
                            skip_bi, skip_tri, skip_quat = True, True, True
                            continue
                    elif word_tri in lookup_dict[word]:
                        if i+2 in align_line and len(possible_src_idx&align_line[i+2])>0:
                            possible_src_idx = list(possible_src_idx&align_line[i+2])
                            src_idx = possible_src_idx[0] if len(possible_src_idx) == 1 else random.choice(possible_src_idx)
                            res.append(word_tri)
                            new_align[src_idx] = 3
                            skip_bi, skip_tri = True, True
                            continue
                    elif word_bi in lookup_dict[word]:
                        possible_src_idx = list(possible_src_idx)
                        src_idx = possible_src_idx[0] if len(possible_src_idx) == 1 else random.choice(possible_src_idx)
                        res.append(word_bi)
                        new_align[src_idx] = 2
                        skip_bi = True
                        continue
            if src_set is not None:
                for idx in src_set:
                    new_align[idx] = 1
            res.append(word)
#         return " ".join(res), new_align
        return " ".join(res), " ".join([str(v) for k,v in sorted(new_align.items())])
        
    tgt_phrase, new_align = zip(*[_concat_mwe(tgt_sent, align_sent,src_sent) for tgt_sent, align_sent, src_sent in zip(tgt,align, src)])
    return tgt_phrase, new_align


with open(os.path.join(path_data,"valid.tok.true."+src_lang),"r") as file:
    src_val = file.readlines()
with open(os.path.join(path_data,"test.tok.true."+src_lang),"r") as file:
    src_test = file.readlines()
with open(os.path.join(path_data,"test.tok.true."+tgt_lang),"r") as file:
    tgt_test = file.readlines()
with open(os.path.join(path_data,"valid.tok.true."+tgt_lang),"r") as file:
    tgt_val = file.readlines()
with open(os.path.join(path_data,"test.align"),"r") as file:
    align_test = file.readlines()
with open(os.path.join(path_data,"valid.align"),"r") as file:
    align_val = file.readlines()


tgt_mwe_train, new_align_train = concat_mwe(tgt, phrase_lookup, align, src)
tgt_mwe_test, new_align_test = concat_mwe(tgt_test, phrase_lookup, align_test, src_test)
tgt_mwe_val, new_align_val = concat_mwe(tgt_val, phrase_lookup, align_val, src_val)

with open(os.path.join(path_data,"train.tok.true.mwe."+tgt_lang),"w") as file:
    file.write("\n".join(tgt_mwe_train))
with open(os.path.join(path_data,"test.tok.true.mwe."+tgt_lang),"w") as file:
    file.write("\n".join(tgt_mwe_test))
with open(os.path.join(path_data,"valid.tok.true.mwe."+tgt_lang),"w") as file:
    file.write("\n".join(tgt_mwe_val))
    
with open(os.path.join(path_data,"train.mwe.align"),"w") as file:
    file.write("\n".join(new_align_train))
with open(os.path.join(path_data,"test.mwe.align"),"w") as file:
    file.write("\n".join(new_align_test))
with open(os.path.join(path_data,"valid.mwe.align"),"w") as file:
    file.write("\n".join(new_align_val))



def extract_mwe_cases_test(src, tgt_mwe, tgt_original):
    src_mwe_only, tgt_mwe_only_mwe, tgt_mwe_only, tgt_mwe_only_orig = [], [], [], []
    for src_sent, tgt_sent, tgt_original_sent in zip(src, tgt_mwe, tgt_original):
        if "_" in tgt_sent:
            src_mwe_only.append(src_sent)
            tgt_mwe_only.append(tgt_sent.replace("_"," "))
            tgt_mwe_only_mwe.append(tgt_sent)
            tgt_mwe_only_orig.append(tgt_original_sent)
    
    assert len(src_mwe_only) == len(tgt_mwe_only)
    return src_mwe_only, tgt_mwe_only, tgt_mwe_only_mwe, tgt_mwe_only_orig

with open(os.path.join(path_data, "test.tok.true."+src_lang), "r") as file:
    src_test = file.readlines()
with open(os.path.join(path_data, "test."+tgt_lang), "r") as file:
    tgt_original_test = file.readlines()
src_mwe_only_test, tgt_mwe_only_test , tgt_mwe_only_test_mwe, tgt_mwe_only_orig_test = extract_mwe_cases_test(src_test, tgt_mwe_test, tgt_original_test)

with open(os.path.join(path_data,"test.tok.true.mwe."+tgt_lang+"_mweonly"),"w") as file:
    file.write("\n".join(tgt_mwe_only_test_mwe))
with open(os.path.join(path_data,"test.tok.true."+tgt_lang+"_mweonly"),"w") as file:
    file.write("\n".join(tgt_mwe_only_test))
with open(os.path.join(path_data, "test.tok.true."+src_lang+"_mweonly"), "w") as file:
    file.write("".join(src_mwe_only_test))
with open(os.path.join(path_data, "test."+tgt_lang+"_mweonly"), "w") as file:
    file.write("".join(tgt_mwe_only_orig_test))


