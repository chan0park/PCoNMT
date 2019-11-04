from __future__ import division

import math
import random

import torch
from torch.autograd import Variable

import onmt


class Dataset(object):

    def __init__(self, srcData, tgtData, tgtUniData, alignData, batchSize, cuda, volatile=False, fert_dim=2):
        self.src = srcData
        if tgtData:
            self.tgt = tgtData
            assert(len(self.src) == len(self.tgt))
        else:
            self.tgt = None
        if tgtUniData:
            self.tgt_uni = tgtUniData
            assert(len(self.src) == len(self.tgt_uni))
        else:
            self.tgt_uni = None
        if alignData:
            self.align = alignData
            assert(len(self.tgt)==len(self.align))
        else:
            self.align = None
        self.cuda = cuda

        self.batchSize = batchSize
        self.numBatches = math.ceil(len(self.src)/batchSize)
        self.volatile = volatile
        self.fert_dim = fert_dim

    def _batchify(self, data, align_right=False, include_lengths=False):
        lengths = [x.size(0) for x in data]
        max_length = max(lengths)
        out = data[0].new(len(data), max_length).fill_(onmt.Constants.PAD)
        for i in range(len(data)):
            data_length = data[i].size(0)
            offset = max_length - data_length if align_right else 0
            out[i].narrow(0, offset, data_length).copy_(data[i])

        if include_lengths:
            return out, lengths
        else:
            return out

    def _batchify_align(self, data, align_right=False):
        lengths = [x.size(0) for x in data]
        max_length = max(lengths)
        out = data[0].new(len(data), max_length).fill_(0)
        for i in range(len(data)):
            data_length = data[i].size(0)
            offset = max_length - data_length if align_right else 0
            out[i].narrow(0, offset, data_length).copy_(data[i])
        
        if self.fert_dim == 5:
            out = torch.clamp(out,0,4)
        elif self.fert_dim == 4:
            out = torch.clamp(out,0,3)
        elif self.fert_dim == 2:
            out = torch.clamp(out-1,0,1)
        # init_class = 1
        # out = torch.clamp(out, init_class, self.fert_dim-init_class+1) - init_class
        # out = torch.clamp(out, 0, 4)
        return out

    def __getitem__(self, index):
        def wrap(b):
            if b is None:
                return b
            b = torch.stack(b, 0).t().contiguous()
            if self.cuda:
                b = b.cuda()
            b = Variable(b, volatile=self.volatile)
            return b

        def wrap_align(b):
            if b is None:
                return b
            b = torch.stack(b, 0)
            if self.cuda:
                b = b.cuda()
            b = Variable(b, volatile=self.volatile)
            return b

        assert index < self.numBatches, "%d > %d" % (index, self.numBatches)
        srcBatch, lengths = self._batchify(
            self.src[index*self.batchSize:(index+1)*self.batchSize],
            align_right=False, include_lengths=True)

        if self.tgt:
            tgtBatch = self._batchify(
                self.tgt[index*self.batchSize:(index+1)*self.batchSize])
        else:
            tgtBatch = None

        if self.tgt_uni:
            tgtUniBatch = self._batchify(
                self.tgt_uni[index*self.batchSize:(index+1)*self.batchSize])
        else:
            tgtUniBatch = None

        if self.align:
            alignBatch = self._batchify_align(
                self.align[index*self.batchSize:(index+1)*self.batchSize])
        else:
            alignBatch = None

        # within batch sorting by decreasing length for variable length rnns
        indices = range(len(srcBatch))
        # batch = zip(indices, srcBatch) if tgtBatch is None else zip(indices, srcBatch, tgtBatch, alignBatch)
        if tgtBatch is None:
            batch = zip(indices, srcBatch)
        elif tgtUniBatch is None:
            batch = zip(indices, srcBatch, tgtBatch)
        elif alignBatch is None:
            batch = zip(indices, srcBatch, tgtBatch, tgtUniBatch)
        else: 
            batch = zip(indices, srcBatch, tgtBatch, tgtUniBatch, alignBatch)

        batch, lengths = zip(*sorted(zip(batch, lengths), key=lambda x: -x[1]))
        if tgtBatch is None:
            indices, batch = zip(*batch)
        elif tgtUniBatch is None:
            indices, srcBatch, tgtBatch = zip(*batch)
        elif alignBatch is None:
            indices, srcBatch, tgtBatch, tgtUniBatch = zip(*batch)
        else: 
            indices, srcBatch, tgtBatch, tgtUniBatch, alignBatch = zip(*batch)

        return (wrap(srcBatch), lengths), wrap(tgtBatch), wrap(tgtUniBatch), wrap_align(alignBatch), indices

    def __len__(self):
        return self.numBatches


    def shuffle(self):
        data = list(zip(self.src, self.tgt, self.align))
        self.src, self.tgt, self.align = zip(*[data[i] for i in torch.randperm(len(data))])
