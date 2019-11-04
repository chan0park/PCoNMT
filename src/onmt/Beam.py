from __future__ import division

# Class for managing the internals of the beam search process.
#
#
#         hyp1#-hyp1---hyp1 -hyp1
#                 \             /
#         hyp2 \-hyp2 /-hyp2#hyp2
#                               /      \
#         hyp3#-hyp3---hyp3 -hyp3
#         ========================
#
# Takes care of beams, back pointers, and scores.

import torch
import onmt


class Beam(object):
    def __init__(self, size, cuda=False):

        self.size = size
        #self.done = False

        self.tt = torch.cuda if cuda else torch

        # The score for each translation on the beam.
        self.scores = self.tt.FloatTensor(size).zero_()
        self.all_scores = []

        # The backpointers at each time-step.
        self.prevKs = []

        # The outputs at each time-step.
        self.nextYs = [self.tt.LongTensor(size).fill_(onmt.Constants.PAD)]
        self.nextYs[0][0] = onmt.Constants.BOS

        self._eos = onmt.Constants.EOS
        self._eos_top = False

        # The attentions (matrix) for each time.
        self.attn = []
        self.out = []
        self.outScore = []
        self.cov = []
        self.init_cov = []

        #Time and k pair for finished
        self.finished = []

    # Get the outputs for the current timestep.
    def getCurrentState(self):
        return self.nextYs[-1]

    # Get the backpointers for the current timestep.
    def getCurrentOrigin(self):
        return self.prevKs[-1]

    #  Given prob over words for every last beam `wordLk` and attention
    #   `attnOut`: Compute and update the beam search.
    #
    # Parameters:
    #
    #     * `wordLk`- probs of advancing from the last step (K x words)
    #     * `attnOut`- attention at the last step
    #
    # Returns: True if beam search is complete.
    def advance(self, wordLk, attnOut, out, cov, init_cov):

        numWords = wordLk.size(1)

        # Sum the previous scores.
        if len(self.prevKs) > 0:
            beamLk = wordLk + self.scores.unsqueeze(1).expand_as(wordLk)
            # Don't let EOS have children.
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] == self._eos:
                    beamLk[i] = -1e20
            # if beamLk.size(0) > 1:
            #     print ("BEAM:", beamLk.size())
            #     print ("Scores:", self.scores)
            #     print ("Length", len(self.prevKs))
            #     input()
        else:
            beamLk = wordLk[0]

        flatBeamLk = beamLk.view(-1)

        bestScores, bestScoresId = flatBeamLk.topk(self.size, 0, True, True)

        self.all_scores.append(self.scores)
        self.scores = bestScores

        # bestScoresId is flattened beam x word array, so calculate which
        # word and beam each score came from
        prevK = bestScoresId / numWords
        nextY = bestScoresId - prevK * numWords
        self.prevKs.append(prevK)
        self.nextYs.append(nextY)
        self.attn.append(attnOut.index_select(0, prevK))
        self.out.append(out.index_select(0, prevK))
        self.outScore.append(wordLk.index_select(0, prevK).index_select(1, nextY))
        self.cov.append(cov.index_select(0, prevK))
        self.init_cov.append(init_cov.index_select(0, prevK))

        normalized_scores = self.scores/len(self.nextYs)
        for i in range(self.nextYs[-1].size(0)):
            # print (i, len(self.nextYs[-1]))
            if self.nextYs[-1][i] == self._eos:
                s = normalized_scores[i]
                self.finished.append((s, len(self.nextYs)-1, i))

        # End condition is when top-of-beam is EOS.
        if self.nextYs[-1][0] == onmt.Constants.EOS:
            self._eos_top = True

        return self.done()

    def done(self):
        self._eos_top and (len(self.finished) >= 1)

    def sortBest(self):
        if len(self.finished) < 1:
            i=0
            normalized_scores = self.scores/len(self.nextYs)
            s = normalized_scores[i]
            self.finished.append((s, len(self.nextYs)-1, i))

        self.finished.sort(key=lambda a: (-a[0], a[1]))
        scores = [sc for sc, _, _ in self.finished]
        ks = [(t,k) for _, t, k in self.finished]
        # print (scores, ks)
        return scores, ks
        # return torch.sort(self.scores, 0, True)

    # Get the score of the best in the beam.
    # def getBest(self):
    #     scores, ids = self.sortBest()
    #     return scores[1], ids[1]

    # Walk back to construct the full hypothesis.
    #
    # Parameters.
    #
    #     * `k` - the position in the beam to construct.
    #
    # Returns.
    #
    #     1. The hypothesis
    #     2. The attention at each time step.
    def getHyp(self, timestep, k):
        hyp, attn, out, score, cov, init_cov = [], [], [], [], [], []
        # print(len(self.prevKs), len(self.nextYs), len(self.attn))
        for j in range(len(self.prevKs[:timestep]) - 1, -1, -1):
            hyp.append(self.nextYs[j+1][k])
            attn.append(self.attn[j][k])
            out.append(self.out[j][k])
            score.append(self.outScore[j][k])
            cov.append(self.cov[j][k])
            init_cov.append(self.init_cov[j][k])
            k = self.prevKs[j][k]

        return hyp[::-1], torch.stack(attn[::-1]), out[::-1], score[::-1], cov[::-1], init_cov[::-1]
