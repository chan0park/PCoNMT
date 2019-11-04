import torch
import codecs

import numpy as np

class Dict(object):
    def __init__(self, data=None, lower=False, special_embeddings=None,phrase_list=None):
        self.idxToLabel = {}
        self.labelToIdx = {}
        self.frequencies = {}
        self.embeddings = {}
        self.ngram_embeddings = {}
        self.ngram_idxToLabel = {}
        self.ngram_labelToIdx = {}
        self.ngram_frequencies= {}
        self.unigram_embeddings = {}
        self.unigram_labelToIdx = {}
        self.unigram_idxToLabel = {}
        self.unigram_frequencies= {}
        self.lower = lower
        self.phrase_list = phrase_list

        # Special entries will not be pruned.
        self.special = []

        if data is not None:
            if type(data) == str:
                self.loadFile(data)
            else:
                self.addSpecials(data, special_embeddings)
                # print self.embeddings.keys()

    def size(self):
        return len(self.idxToLabel)

    def size_uni(self):
        return len(self.unigram_idxToLabel)

    def size_ngram(self):
        return len(self.ngram_idxToLabel)

    # Load entries from a file.
    def loadFile(self, filename):
        i = len(self.idxToLabel)
        # print i
        for line in open(filename):
            fields = line.split()
            if len(fields) > 2:
                idx = int(fields[-1])
                label = ' '.join(fields[:-1])
            elif len(fields) == 2:
                label = fields[0]
                idx = int(fields[1])
            else:
                label = fields[0]
                idx = i
                i += 1
            self.add(label, idx)

    # Write entries to a file.
    def writeFile(self, filename):
        with codecs.open(filename, 'w', "utf-8") as file:
            for i in range(self.size()):
                label = self.idxToLabel[i]
                file.write('%s %d\n' % (label, i))
        file.close()

    def writeEmbFile(self, filename):
        if type(self.embeddings) == type({}):
            embeddings_tensor = np.zeros((self.size(), 300))
            for k, v in self.embeddings.items():
                embeddings_tensor[k] = v
            embeddings = torch.Tensor(embeddings_tensor)
        else:
            embeddings = self.embeddings
        
        with codecs.open(filename, 'w', "utf-8") as file:
            for i in range(self.size()):
                label = self.idxToLabel[i]
                emb = [str(x) for x in embeddings[i].tolist()]
                file.write('%s %s\n' % (label, " ".join(emb)))

        file.close()

    def lookup(self, key, default=None):
        key = key.lower() if self.lower else key
        try:
            return self.labelToIdx[key]
        except KeyError:
            return default

    def getLabel(self, idx, default=None):
        try:
            return self.idxToLabel[idx]
        except KeyError:
            return default

    # Mark this `label` and `idx` as special (i.e. will not be pruned).
    def addSpecial(self, label, idx=None):
        idx = self.add(label, idx, special=True)
        # print label, idx
        self.special += [idx]

    # Mark all labels in `labels` as specials (i.e. will not be pruned).
    def addSpecials(self, labels, special_embeddings=None, special=True):
        for i, label in enumerate(labels):
            self.addSpecial(label)
            if special_embeddings is not None:
                self.add_embedding(label, special_embeddings[i], special=special)
    # Add `label` in the dictionary. Use `idx` as its index if given.
    def add(self, label, idx=None, special=False):
        label = label.lower() if self.lower else label
        if idx is not None:
            self.idxToLabel[idx] = label
            self.labelToIdx[label] = idx
        # else:
        #     if label in self.labelToIdx:
        #         idx = self.labelToIdx[label]
        #     else:
        #         idx = len(self.idxToLabel)
        #         self.idxToLabel[idx] = label
        #         self.labelToIdx[label] = idx

        if special:
            if label in self.ngram_labelToIdx:
                ngram_idx = self.ngram_labelToIdx[label]
            else:
                ngram_idx = len(self.ngram_labelToIdx)
                self.ngram_idxToLabel[ngram_idx] = label
                self.ngram_labelToIdx[label] = ngram_idx
            if ngram_idx not in self.ngram_frequencies:
                self.ngram_frequencies[ngram_idx] = 1
            else:
                self.ngram_frequencies[ngram_idx] += 1
            
            if label in self.unigram_labelToIdx:
                unigram_idx = self.unigram_labelToIdx[label]
            else:
                unigram_idx = len(self.unigram_labelToIdx)
                self.unigram_idxToLabel[unigram_idx] = label
                self.unigram_labelToIdx[label] = unigram_idx    
            if unigram_idx not in self.unigram_frequencies:
                self.unigram_frequencies[unigram_idx] = 1
            else:
                self.unigram_frequencies[unigram_idx] += 1       
            return unigram_idx 

        if "_" in label and label != "_":
            if (self.phrase_list is None) or (label in phrase_list):
                if label in self.ngram_labelToIdx:
                    ngram_idx = self.ngram_labelToIdx[label]
                else:
                    ngram_idx = len(self.ngram_labelToIdx)
                    self.ngram_idxToLabel[ngram_idx] = label
                    self.ngram_labelToIdx[label] = ngram_idx
                if ngram_idx not in self.ngram_frequencies:
                    self.ngram_frequencies[ngram_idx] = 1
                else:
                    self.ngram_frequencies[ngram_idx] += 1
                idx = ngram_idx
        else:
            if label in self.unigram_labelToIdx:
                unigram_idx = self.unigram_labelToIdx[label]
            else:
                unigram_idx = len(self.unigram_labelToIdx)
                self.unigram_idxToLabel[unigram_idx] = label
                self.unigram_labelToIdx[label] = unigram_idx    
            if unigram_idx not in self.unigram_frequencies:
                self.unigram_frequencies[unigram_idx] = 1
            else:
                self.unigram_frequencies[unigram_idx] += 1       
            idx = unigram_idx 

        return idx

    def add_embedding(self, word, emb, unk=None, normalize=True, special=False):
        eps = 1e-6
        if normalize:
            emb = emb/(np.linalg.norm(emb)+eps)
        if special:
            # self.embeddings[self.labelToIdx[word]] = emb
            self.ngram_embeddings[self.ngram_labelToIdx[word]] = emb
            self.unigram_embeddings[self.unigram_labelToIdx[word]] = emb
        else:
            if word in self.ngram_labelToIdx or word in self.unigram_labelToIdx:
                # self.embeddings[self.labelToIdx[word]] = emb
                if "_" in word and word != "_":
                    self.ngram_embeddings[self.ngram_labelToIdx[word]] = emb
                else:
                    self.unigram_embeddings[self.unigram_labelToIdx[word]] = emb
            else:
                self.unigram_embeddings[self.unigram_labelToIdx[unk]] += emb            

    def average_unk(self, unk, n, normalize=True):
        uni_unk_idx = self.unigram_labelToIdx[unk]
        _ngram_unk_idx = self.ngram_labelToIdx[unk]
        ngram_unk_idx = self.size_uni()+ _ngram_unk_idx
        self.embeddings[uni_unk_idx] /= n 
        print(self.size_uni())
        print(self.size())
        print(self.size_ngram())
        # print(self.embeddings.keys())
        # print(self.ngram_embeddings.keys())
        self.embeddings[ngram_unk_idx] /= n
        self.unigram_embeddings[uni_unk_idx] /= n
        self.ngram_embeddings[_ngram_unk_idx] /= n
        if normalize:
            self.embeddings[uni_unk_idx] = self.embeddings[uni_unk_idx]/np.linalg.norm(self.embeddings[uni_unk_idx])
            self.embeddings[ngram_unk_idx] = self.embeddings[ngram_unk_idx]/np.linalg.norm(self.embeddings[ngram_unk_idx])
            self.unigram_embeddings[uni_unk_idx] = self.unigram_embeddings[uni_unk_idx]/np.linalg.norm(self.unigram_embeddings[uni_unk_idx])
            self.ngram_embeddings[_ngram_unk_idx] = self.ngram_embeddings[_ngram_unk_idx]/np.linalg.norm(self.ngram_embeddings[_ngram_unk_idx])

    def average_emb(self):
        mean = sum(self.embeddings.values())/len(self.embeddings)
        for k, v in self.embeddings.items():
            self.embeddings[k] = v-mean
    
    def convert_embeddings_to_torch(self, dim=300):
        embeddings_tensor = np.zeros((self.size(), dim))
        uni_embeddings_tensor = np.zeros((self.size_uni(), dim))
        ngram_embeddings_tensor = np.zeros((self.size_ngram(), dim))

        for k, v in self.embeddings.items():
            embeddings_tensor[k] = v
        for k, v in self.unigram_embeddings.items():
            uni_embeddings_tensor[k] = v
        for k, v in self.ngram_embeddings.items():
            ngram_embeddings_tensor[k] = v
        self.embeddings = torch.Tensor(embeddings_tensor)
        self.unigram_embeddings = torch.Tensor(uni_embeddings_tensor)
        self.ngram_embeddings = torch.Tensor(ngram_embeddings_tensor)

    # Return a new dictionary with the `size` most frequent entries.
    def prune(self, size, target=False):
        if target:
            per_size = int(size/2)
        else:
            per_size = size
        print((target,per_size))
        print((self.size_uni(), self.size_ngram()))

        if per_size >= self.size_uni() and per_size>=self.size_ngram():
            print("a")
            self.idxToLabel = self.unigram_idxToLabel.copy()
            self.idxToLabel.update({k+len(self.unigram_idxToLabel):v for k,v in self.ngram_idxToLabel.items()})
            self.labelToIdx = {k:v+len(self.unigram_idxToLabel) for k,v in self.ngram_labelToIdx.copy().items()}
            self.labelToIdx.update(self.unigram_labelToIdx)
            # self.labelToIdx = self.unigram_labelToIdx.copy()
            # self.labelToIdx.update({k:v+len(self.unigram_idxToLabel) for k,v in self.ngram_labelToIdx.items()})
            self.embeddings = self.unigram_embeddings.copy()
            # self.embeddings.update({i+len(self.unigram_idxToLabel):self.unigram_embeddings[i] for i in self.special})
            # self.ngram_embeddings.update({i:self.unigram_embeddings[i] for i in self.special})
            self.embeddings.update({k+len(self.unigram_idxToLabel):v for k,v in self.ngram_embeddings.items()})


            # self.idxToLabel = {**self.unigram_idxToLabel, **{k+len(self.unigram_idxToLabel):v for k,v in self.ngram_idxToLabel.items()}}
            # self.labelToIdx = {**self.unigram_labelToIdx, **{k:v+len(self.unigram_labelToIdx) for k,v in self.ngram_labelToIdx.items()}}
            # self.embeddings = {**self.unigram_embeddings, **{k+len(self.unigram_idxToLabel):v for k,v in self.embeddings.items()}}
            return self, self.size()

        # Only keep the `size` most frequent entries.
        # freq = torch.Tensor(
        #         [self.frequencies[i] for i in range(len(self.frequencies))])

        uni_freq = torch.Tensor(
                [self.unigram_frequencies[i] for i in range(len(self.unigram_frequencies))])
        ngram_freq = torch.Tensor(
                [self.ngram_frequencies[i] for i in range(len(self.ngram_frequencies))])

        _, uni_idx = torch.sort(uni_freq, 0, True)
        _, ngram_idx = torch.sort(ngram_freq, 0, True)

        # if type(idx) != type([]):
        #     idx = idx.tolist()
        if type(uni_idx) != type([]):
            uni_idx = uni_idx.tolist()
        if type(ngram_idx) != type([]):
            ngram_idx = ngram_idx.tolist()

        newDict = Dict()
        newDict.lower = self.lower

        # Add special entries in all cases.
        for i in self.special:
            newDict.addSpecial(self.unigram_idxToLabel[i])
            if target:
                newDict.add_embedding(self.unigram_idxToLabel[i], self.unigram_embeddings[i], special=True)

        c_uni=0
        for i in uni_idx:
            if i in [0,1,2,3]:
                print("special in the list"+self.unigram_idxToLabel[i])
                continue
            if target:
                if i in self.unigram_embeddings:
                    newDict.add(self.unigram_idxToLabel[i])
                    newDict.add_embedding(self.unigram_idxToLabel[i], self.unigram_embeddings[i])
                    c_uni+=1
            else:
                newDict.add(self.unigram_idxToLabel[i])
                c_uni+=1
            if c_uni >= per_size:
                break

        c_ngram=0
        for i in ngram_idx:
            if target:
                if i in self.ngram_embeddings:
                    newDict.add(self.ngram_idxToLabel[i])
                    newDict.add_embedding(self.ngram_idxToLabel[i], self.ngram_embeddings[i])
                    c_ngram+=1
            else:
                newDict.add(self.ngram_idxToLabel[i])
                c_ngram+=1
            if c_ngram >= per_size:
                break

        print((c_uni, c_ngram, c_uni+c_ngram))
        newDict.idxToLabel = newDict.unigram_idxToLabel.copy()
        newDict.idxToLabel.update({k+len(newDict.unigram_idxToLabel):v for k,v in newDict.ngram_idxToLabel.items()})
        # newDict.labelToIdx = newDict.unigram_labelToIdx.copy()
        # newDict.labelToIdx.update({k:v+len(newDict.unigram_idxToLabel) for k,v in newDict.ngram_labelToIdx.items()})
        newDict.labelToIdx = {k:v+len(newDict.unigram_idxToLabel) for k,v in newDict.ngram_labelToIdx.copy().items()}
        newDict.labelToIdx.update(newDict.unigram_labelToIdx)
        newDict.embeddings = newDict.unigram_embeddings.copy()
        newDict.embeddings.update({k+len(newDict.unigram_idxToLabel):v for k,v in newDict.ngram_embeddings.items()})

        # newDict.idxToLabel = {**newDict.unigram_idxToLabel, **{k+len(newDict.unigram_idxToLabel):v for k,v in newDict.ngram_idxToLabel.items()}}
        # newDict.labelToIdx = {**newDict.unigram_labelToIdx, **{k:v+len(newDict.unigram_idxToLabel) for k,v in newDict.ngram_labelToIdx.items()}}
        # newDict.embeddings = {**newDict.unigram_embeddings, **{k+len(newDict.unigram_idxToLabel):v for k,v in newDict.embeddings.items()}}

        # for i in idx:
        #     if target:
        #         if i in self.embeddings:
        #             newDict.add(self.idxToLabel[i])
        #             newDict.add_embedding(self.idxToLabel[i], self.embeddings[i])
        #             c+=1
        #     else:
        #         newDict.add(self.idxToLabel[i])
        #         c+=1
        #     if c >= size:
        #         break
        return newDict, c_uni+c_ngram

    # Convert `labels` to indices. Use `unkWord` if not found.
    # Optionally insert `bosWord` at the beginning and `eosWord` at the .
    def convertToIdx(self, labels, unkWord, bosWord=None, eosWord=None):
        vec = []

        if bosWord is not None:
            vec += [self.lookup(bosWord)]

        unk = self.lookup(unkWord)
        vec += [self.lookup(label, default=unk) for label in labels]
        unky = False
        if unk in vec:
            unky = True

        if eosWord is not None:
            vec += [self.lookup(eosWord)]

        return torch.LongTensor(vec), unky

    # Convert `idx` to labels. If index `stop` is reached, convert it and return.
    def convertToLabels(self, idx, stop):
        labels = []
        for i in idx:
            l = self.getLabel(i)
            labels += [l]
            # labels += self.getLabel(i).split("_")
            if l == stop:
                break

        return labels
