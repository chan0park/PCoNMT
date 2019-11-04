from __future__ import division

# import ptvsd
# ptvsd.enable_attach(address=('127.0.0.1', 99), redirect_output=True)
# ptvsd.wait_for_attach()

import onmt
import torch
import argparse
import math
import codecs
import time
import os
import nltk
from collections import Counter

parser = argparse.ArgumentParser(description='translate.py')

parser.add_argument('-model', required=True,
                    help='Path to model .pt file')
parser.add_argument('-src',   required=True,
                    help='Source sequence to decode (one line per sequence)')
parser.add_argument('-tgt',
                    help='True target sequence (optional)')
parser.add_argument('-align',
                    help='True align')
parser.add_argument('-tgt_dict',
                    help='Target Embeddings (optional). This is usually for cases when you want to evaluate using a larger embedding table than the one used for training. It should the same format as the target embedding which is part of the training data')
parser.add_argument('-lookup_dict',
                    help='Look up dictionary')
parser.add_argument('-output', default='pred.txt',
                    help="""Path to output the predictions (each line will
                    be the decoded sequence""")
parser.add_argument('-loss', default='cosine',
                    help="""loss function: [l2|cosine|maxmargin|nllvmf]""")
parser.add_argument('-png_filename',
                    help='True target sequence (optional)')
parser.add_argument('-beam_size',  type=int, default=5,
                    help='Beam size')
parser.add_argument('-batch_size', type=int, default=30,
                    help='Batch size')
parser.add_argument('-max_sent_length', type=int, default=100,
                    help='Maximum sentence length.')
parser.add_argument('-replace_unk', action="store_true",
                    help="""Replace the generated UNK tokens with the source
                    token that had the highest attention weight. If phrase_table
                    is provided, it will lookup the identified source token and
                    give the corresponding target token. If it is not provided
                    (or the identified source token does not exist in the
                    table) then it will copy the source token""")
# parser.add_argument('-phrase_table',
#                     help="""Path to source-target dictionary to replace UNK
#                     tokens. See README.md for the format of this file.""")
parser.add_argument('-verbose', action="store_true",
                    help='Print scores and predictions for each sentence')
parser.add_argument('-use_lm', action="store_true",
                    help='Use a Language Model in Beam search')
parser.add_argument('-n_best', type=int, default=1,
                    help="""If verbose is set, will output the n_best
                    decoded sentences""")
parser.add_argument('-gpu', type=int, default=-1,
                    help="Device to run on")
parser.add_argument('-save_att', action="store_true", default=False,
                    help="Save a file for attention visualizer or not")
parser.add_argument('-save_dir', default='predicted_models',
                    help="""Path to save results""")



def reportScore(name, scoreTotal, wordsTotal):
    print("%s AVG SCORE: %.4f, %s PPL: %.4f" % (
        name, scoreTotal / wordsTotal,
        name, math.exp(-scoreTotal/wordsTotal)))

def addone(f):
    for line in f:
        yield line
    yield None

def encode_or_decode(txt):
    try:
        txt = txt.decode("utf-8")
    except:
        txt = txt.encode("utf-8","ignore").decode("utf-8")
    return txt


def get_src_align_count(list_pairs, src_len):
    res = [0] * src_len
    src_idxs = [int(x.split("-")[0]) for x in list_pairs]
    counts = Counter(src_idxs)
    for s_idx, c in counts.items():
        res[s_idx] += c
    return res

# def main():
if True:
    opt = parser.parse_args()
    opt.cuda = opt.gpu > -1
    print(opt)
    if opt.cuda:
        torch.cuda.set_device(opt.gpu)

    translator = onmt.Translator_ngram(opt)
    model_idx = opt.model.split("/")[-1].replace(".","").replace("_","").replace("bestmodel","b").replace("pt","")

    if opt.save_att:
        if not os.path.isdir(opt.save_dir):
            os.mkdir(opt.save_dir)
        opt.save_dir = opt.save_dir + "/"+model_idx
        if not os.path.isdir(opt.save_dir):
            os.mkdir(opt.save_dir)
        # opt.output = opt.save_dir+"/"+opt.output
        model_opt = torch.load(opt.model,map_location=lambda storage, loc: storage)['opt']
        test_data_idx = opt.src.split("/")[-1].replace(".","").replace("_","")
        res_model = {}
        res_model["model"] = model_idx
        res_model["test_data"] = test_data_idx
        res_model["opt"] = model_opt
        res_model["pred_score"] = []
        res_model["bleu"] = []
        res_model["bow_diff"] = []
        # outF_att = codecs.open(model_idx+"/att", "w", "utf-8")
        outF_model = codecs.open(opt.save_dir+"/model", "w", "utf-8")

    outF = codecs.open(opt.output, 'w', 'utf-8')
    outMweF = codecs.open(opt.output+".mwe", 'w', 'utf-8')
    fertF = codecs.open(opt.output+".fert","w","utf-8")
    bleuF = codecs.open(opt.output+".bleu", 'w', 'utf-8')

    predScoreTotal, predWordsTotal, goldScoreTotal, goldWordsTotal = 0, 0, 0, 0

    srcBatch, tgtBatch, alignBatch = [], [], None

    count = 0

    total_time = 0.0
    nsamples = 0.0

    tgtF = open(opt.tgt) if opt.tgt else None
    align = open(opt.align) if opt.align else None
    for line in addone(codecs.open(opt.src, "r", "utf-8")):

        if line is not None:
            srcTokens = line.split()
            srcBatch += [srcTokens]
            # alignPairs = align.readline().strip().split()
            # alignBatch += [torch.Tensor(get_src_align_count(alignPairs, len(srcTokens)))]
            if tgtF:
                tgtTokens = tgtF.readline().split() if tgtF else None
                tgtBatch += [tgtTokens]

            if len(srcBatch) < opt.batch_size:
                continue
        else:
            # at the end of file, check last batch
            if len(srcBatch) == 0:
                break

        start_time = time.time()
        predBatch, predScore, knntime, attn, out, outScore, cov, init_cov = translator.translate(srcBatch, tgtBatch, alignBatch)
        total_time += (time.time()-start_time)
        nsamples += len(predBatch)

        predScoreTotal += sum(score[0] for score in predScore)
        predWordsTotal += sum(len(x[0]) for x in predBatch)
        # if tgtF is not None:
        #     goldScoreTotal += sum(goldScore)
        #     goldWordsTotal += sum(len(x) for x in tgtBatch)

        for b in range(len(predBatch)):
            count += 1
            try:
                content = " ".join(predBatch[b][0])
            except:
                content = " ".join([encode_or_decode(x) for x in predBatch[b][0]])
            outF.write(content.replace("_"," ") + '\n')
            outF.flush()
            # print(" ".join([str(int(_int)).strip() for _int in init_cov[b][0][0]]))
            fertF.write(" ".join([str(int(_int)).strip() for _int in init_cov[b][0][0]])+'\n')
            fertF.flush()
            outMweF.write(content+'\n')
            outMweF.flush()

            # srcSent = ' '.join(srcBatch[b])
            srcSent = ' '.join([encode_or_decode(x) for x in srcBatch[b]])
            if translator.tgt_dict.lower:
                srcSent = srcSent.lower()
            if opt.verbose:
                print('SENT %d: %s' % (count, srcSent))
                print('PRED %d: %s' % (count, " ".join(predBatch[b][0])))
                print("PRED SCORE: %.4f" % predScore[b][0])

            if tgtF is not None:
                tgtSent = ' '.join(tgtBatch[b])
                if translator.tgt_dict.lower:
                    tgtSent = tgtSent.lower()
                bleu = nltk.translate.bleu_score.sentence_bleu([tgtSent.split()], [y for x in predBatch[b][0] for y in x.split("_")])
                # bleu = nltk.translate.bleu_score.sentence_bleu([tgtSent.split()], predBatch[b][0])
                try:
                    bleuF.write("{}\t{}\t{}\n".format(bleu, content, tgtSent))
                except:
                    tgtSent = ' '.join([encode_or_decode(x) for x in tgtBatch[b]])
                    bleuF.write(str(bleu))
                    bleuF.write("\t")
                    bleuF.write(content)
                    bleuF.write("\t")
                    bleuF.write(tgtSent)
                    bleuF.write("\n")
                bleuF.flush()

                if opt.verbose:
                    print('GOLD %d: %s ' % (count, tgtSent))
                    # print("GOLD SCORE: %.4f" % goldScore[b])

            if opt.n_best > 1 and opt.verbose:
                print('\nBEST HYP:')
                for n in range(opt.n_best):
                    print("[%.4f] %s" % (predScore[b][n], " ".join(predBatch[b][n])))
                print('')

            if opt.save_att:
                res_att = {}
                res_att["model"] = model_idx
                res_att["sentidx"] = test_data_idx+str(count)
                res_att["idx"] = str(count)
                res_att["src"] = srcSent
                res_att["gold"] = tgtSent
                res_att["pred"] = content
                res_att["pred_score"] = float(predScore[b][0])
                res_att["bleu"] = float(bleu)
                attSent = [x.cpu().numpy() for x in attn[b][0][:-1]] if attn != [] else None
                # print(attSent[0].sum())
                # print(attSent[0])
                res_att["att"] = attSent
                res_att["att2"] = None
                outSent = [x.cpu().numpy() for x in out[b][0]] if out != [] else None
                res_att["output"] = outSent
                outSentScore = [x.cpu().numpy() for x in outScore[b][0]] if out != [] else None
                res_att["out_score"] = outSentScore
                covSent = [x.cpu().numpy() for x in cov[b][0]] if cov != [] else None
                res_att["cov"] = covSent
                initCovSent = init_cov[b][0][0].cpu().numpy()
                res_att["init_cov"] = initCovSent

                res_model["pred_score"].append(float(predScore[b][0]))
                res_model["bleu"].append(float(bleu))
                res_model["bow_diff"].append(len(set(predBatch[b][0])^set(srcBatch[b]))*1.0/len(srcBatch[b]))
                outF_att = codecs.open(opt.save_dir+"/"+str(count), "w", "utf-8")
                outF_att.write(str(res_att))
                outF_att.flush()


        srcBatch, tgtBatch, alignBatch = [], [], []

    reportScore('PRED', predScoreTotal, predWordsTotal)
    # if tgtF:
    #     reportScore('GOLD', goldScoreTotal, goldWordsTotal)

    if tgtF:
        tgtF.close()

    samples_per_sec = nsamples/total_time
    print ("Average samples per second: %f, %f, %f" % (nsamples, total_time, samples_per_sec))
    print ("Time per sample %f" % (total_time/nsamples))

    if opt.save_att:
        import subprocess
        bleu_output = subprocess.check_output(["bash scripts/evaluate_sachin.sh {} {} en".format(opt.output, opt.tgt.replace(".tok.true",""))], shell=True).decode("utf-8")
        # bleu_output = subprocess.check_output(["./scripts/multi-bleu.perl {} < {}".format(opt.tgt, opt.output)], shell=True).decode("utf-8")
        # bleu_output = subprocess.check_output(["bash scripts/evaluate_sachin.sh {} {} {}".format(opt.output, opt.tgt, opt.tgt+".bleu")], shell=True).decode("utf-8")
        print("BLEU: {}".format(bleu_output))
        bleu_output = bleu_output.replace(",", " ").split()
        res_model["bleu_ngram"] = "{} ({})".format(bleu_output[2], bleu_output[3])
        outF_model.write(str(res_model)+"\n")
        outF_model.flush()
        # zip_output = subprocess.check_output(["zip -r {}.zip {}".format(opt.output.replace(".pt",""), opt.output )], shell=True)


# if __name__ == "__main__":
#     main()
