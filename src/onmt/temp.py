from __future__ import division

# import ptvsd
# ptvsd.enable_attach(address=('127.0.0.1', 99), redirect_output=True)
# ptvsd.wait_for_attach()

import onmt
import argparse
import torch
import torch.nn as nn
from torch import cuda
from torch.autograd import Variable
import math
import time
import sys
import numpy as np
from loss import *

import random
from tqdm import tqdm
from sklearn.metrics import classification_report

import torch._utils
try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2

parser = argparse.ArgumentParser(description='train.py')

## Data options
parser.add_argument('-data', required=True,
                    help='Path to the *-train.pt file from prepare_data.py')
parser.add_argument('-save_model', default='model',
                    help="""Model filename (the model will be saved as
                    <save_model>_epochN_PPL.pt where PPL is the
                    validation perplexity""")
parser.add_argument('-train_from_state_dict', default='', type=str,
                    help="""If training from a checkpoint then this is the
                    path to the pretrained model's state_dict.""")
parser.add_argument('-train_from', default='', type=str,
                    help="""If training from a checkpoint then this is the
                    path to the pretrained model.""")
parser.add_argument('-train_anew', action='store_true',
                     help="Load from the train_from model but restart optimizer")
parser.add_argument('-nonlin_gen', action='store_true',
                    help="Make generator (final layer which produces the continuous vector) non linear using a 2 layer MLP")
parser.add_argument('-save_all_epochs', action='store_true',
                    help="Save the model at every epoch (Could be memory consuming")

## Model options
parser.add_argument('-layers', type=int, default=2,
                    help='Number of layers in the LSTM encoder/decoder')
parser.add_argument('-rnn_size', type=int, default=1024,
                    help='Size of LSTM hidden states')
parser.add_argument('-word_vec_size', type=int, default=512,
                    help='Input word embedding sizes')
parser.add_argument('-output_emb_size', type=int, default=300,
                    help='Dimension of the output vector')
parser.add_argument('-input_feed', type=int, default=1,
                    help="""Feed the context vector at each time step as
                    additional input (via concatenation with the word
                    embeddings) to the decoder.""")
parser.add_argument('-tie_emb', action='store_true',
                    help="Tie input and output embeddings of decoder")
parser.add_argument('-fix_src_emb', action='store_true',
                    help="Initialize and fix the source embeddings")

# parser.add_argument('-residual',   action="store_true",
#                     help="Add residual connections between RNN layers.")
parser.add_argument('-brnn', action='store_true',
                    help='Use a bidirectional encoder')
parser.add_argument('-loss', default='nllvmf', type=str,
                    help='Loss Function to use: [ce|l2|cosine|maxmargin|nllvmf]')
parser.add_argument('-brnn_merge', default='concat',
                    help="""Merge action for the bidirectional hidden states:
                    [concat|sum]""")

## Optimization options
parser.add_argument('-batch_size', type=int, default=64,
                    help='Maximum batch size')
parser.add_argument('-max_generator_batches', type=int, default=32,
                    help="""Maximum batches of words in a sequence to run
                    the generator on in parallel. Higher is faster, but uses
                    more memory.""")
parser.add_argument('-epochs', type=int, default=15,
                    help='Number of training epochs')
parser.add_argument('-start_epoch', type=int, default=1,
                    help='The epoch from which to start')
parser.add_argument('-param_init', type=float, default=0.1,
                    help="""Parameters are initialized over uniform distribution
                    with support (-param_init, param_init)""")
parser.add_argument('-optim', default='adam',
                    help="Optimization method. [sgd|adagrad|adadelta|adam]")
parser.add_argument('-max_grad_norm', type=float, default=5,
                    help="""If the norm of the gradient vector exceeds this,
                    renormalize it to have the norm equal to max_grad_norm""")
parser.add_argument('-dropout', type=float, default=0.0,
                    help='Dropout probability; applied between LSTM stacks.')
parser.add_argument('-curriculum', action="store_true",
                    help="""For this many epochs, order the minibatches based
                    on source sequence length. Sometimes setting this to 1 will
                    increase convergence speed.""")
parser.add_argument('-extra_shuffle', action="store_true",
                    help="""By default only shuffle mini-batch order; when true,
                    shuffle and re-assign mini-batches""")

#learning rate
parser.add_argument('-learning_rate', type=float, default=1.0,
                    help="""Starting learning rate. If adagrad/adadelta/adam is
                    used, then this is the global learning rate. Recommended
                    settings: sgd = 1, adagrad = 0.1, adadelta = 1, adam = 0.001""")
parser.add_argument('-learning_rate_decay', type=float, default=0.5,
                    help="""If update_learning_rate, decay learning rate by
                    this much if (i) perplexity does not decrease on the
                    validation set or (ii) epoch has gone past
                    start_decay_at""")
parser.add_argument('-start_decay_at', type=int, default=8,
                    help="""Start decaying every epoch after and including this
                    epoch""")

#pretrained word vectors (not really used in this model)
parser.add_argument('-pre_word_vecs_enc',
                    help="""If a valid path is specified, then this will load
                    pretrained word embeddings on the encoder side.
                    See README for specific formatting instructions.""")
parser.add_argument('-pre_word_vecs_dec',
                    help="""If a valid path is specified, then this will load
                    pretrained word embeddings on the decoder side.
                    See README for specific formatting instructions.""")
parser.add_argument('-pre_word_vecs_output',
                    help="""This will load the output embeddings using which
                    loss will be minimized.""")

# cov specific
parser.add_argument('-cov_mode', default='h', type=str, help="[emb|h|emh]")
parser.add_argument('-cov_lr', type=float, default=0.0005,
                    help='Number of training epochs')
parser.add_argument('-cov_dim', type=int, default=2,
                    help='Number of training epochs')
parser.add_argument('-cov_w', type=int, default=1,
                    help='weight for cov loss')
parser.add_argument('-cov_ep', type=int, default=9,
                    help='Number of training epochs')
parser.add_argument('-pre_ep', type=int, default=6,
                    help='Number of training epochs')
parser.add_argument('-uni_ep', type=int, default=0,
                    help='Number of training epochs')
parser.add_argument('-downsample', type=bool, default=True,
                    help='boolean specifying whether we do downsample cov examples or not')
parser.add_argument('-cov_hidden_dim', type=int, default=4096,
                    help='Number of training epochs')
parser.add_argument('-cov_batch_size', type=int, default=32,
                    help='Number of training epochs')

# GPU
parser.add_argument('-gpus', default=[], nargs='+', type=int,
                    help="Use CUDA on the listed devices.")

parser.add_argument('-log_interval', type=int, default=50,
                    help="Print stats at this interval.")

parser.add_argument('-test', action="store_true", default=False)
opt = parser.parse_args()

print(opt)

if torch.cuda.is_available() and not opt.gpus:
    print("WARNING: You have a CUDA device, so you should probably run with -gpus 0")

if opt.gpus:
    cuda.set_device(opt.gpus[0])

def NMTCriterion(vocabSize):
    weight = torch.ones(vocabSize)
    weight[onmt.Constants.PAD] = 0
    crit = nn.NLLLoss(weight, size_average=False)
    if opt.gpus:
        crit.cuda()
    return crit

    # for param in model.decoder.attn.linear_cover.parameters():
    #     param.requires_grad = True

def unfreeze(model):
    # for param in model.encoder.parameters():
    #     param.requires_grad = True
    for param in model.decoder.parameters():
        param.requires_grad = True
    for param in model.generator.parameters():
        param.requires_grad = True

def eval(model, loss_fn, target_embeddings, target_uni_embeddings, target_ngram_embeddings, data, pre_train=False):
    total_loss = 0
    total_words = 0
    total_other_loss = 0
    # loss_cov = nn.MSELoss()
    model.eval()
    for i in range(len(data)):
        batch = data[i][:-1] # exclude original indices
        outputs, fert = model(batch, is_cov, pre_train)
        # if is_cov:
        #     targets = batch[3]
        #     targets_cov = targets.view(-1).long()
        #     outputs = outputs.view(-1, outputs.size(2))
        #     gradOutput = loss_cov(outputs, targets_cov)
        #     loss = gradOutput.data[0]
        #     other_loss = 0.0
        # else:
        targets = batch[1][1:]  # exclude <s> from targets
        loss, _, other_loss = loss_fn(
                outputs, targets, target_embeddings, target_uni_embeddings, target_ngram_embeddings, fert, model.generator, opt, is_cov, True)
        # loss, _, other_loss = loss_fn(
        #         outputs, targets, target_embeddings, model.generator, opt, is_cov, eval=True)
        total_loss += loss
        total_other_loss += other_loss
        total_words += targets.data.ne(onmt.Constants.PAD).float().sum()

    model.train()
    return total_loss / total_words, total_other_loss / total_words

def evalCov(model, loss_cov, cov_data_val):
    batch_tot_num = int(len(cov_data_val)*1.0/opt.batch_size)
    running_loss = 0.0
    total_loss = 0.0
    total, correct = 0, 0
    model.eval()
    predicted, labels = [], []
    with tqdm(enumerate(range(batch_tot_num)),total=batch_tot_num, leave=False, dynamic_ncols=True) as pbar:
        for i, batch_num in pbar:
            # get the inputs; data is a list of [inputs, labels]
            inputs, _labels_ = zip(*cov_data_val[batch_num*opt.cov_batch_size:(batch_num+1)*opt.cov_batch_size])
            total += len(_labels_)
            _labels_ = [int(f) for f in _labels_]
            labels.extend(list(_labels_))
            inputs= Variable(torch.stack(inputs)).cuda()
            _labels= Variable(torch.LongTensor(_labels_)).cuda()
            outputs = model.encoder.forward_cov(inputs)
            loss = loss_cov(outputs, _labels)
            total_loss += float(loss)
            _, _predicted = torch.max(outputs.data, 1)
            _predicted = _predicted.tolist()
            correct += sum([p == l for (p,l) in zip(_predicted, _labels_)])
            predicted.extend(_predicted)
    print(classification_report(labels, predicted))
    model.train()
    return total_loss/total, correct/total


def trainModel(model, trainData, validData, dataset, target_embeddings, target_uni_embeddings, target_ngram_embeddings, optim):
    def freeze():
        for param in model.encoder.parameters():
            param.requires_grad = False
        for param in model.decoder.parameters():
            param.requires_grad = False
        for param in model.generator.parameters():
            param.requires_grad = False
        for param in model.encoder.cov_fc1.parameters():
            param.requires_grad = True
        for param in model.encoder.cov_fc2.parameters():
            param.requires_grad = True
        for param in model.encoder.cov_fc3.parameters():
            param.requires_grad = True
        optim.set_lr(opt.cov_lr)
        optim.set_parameters(model.parameters())

    print(model)
    sys.stdout.flush()
    model.train()

    # define criterion of each GPU
    if opt.loss == "baseline":
        loss_fn = CrossEntropy
    if opt.loss == "cosine":
        loss_fn = CosineLoss
    elif opt.loss == "l2":
        loss_fn = L2Loss
    elif opt.loss == 'nllvmf':
        loss_fn = NLLvMF
    elif opt.loss == "maxmargin":
        loss_fn = MaxMarginLoss
    else:
        raise ValueError("loss function:%s is not supported"%opt.loss)

    # loss_cov = nn.MSELoss()
    if opt.cov_dim == 5:
        weight = torch.FloatTensor([1,1,opt.cov_w,opt.cov_w,opt.cov_w]).cuda()
    elif opt.cov_dim == 4:
        weight = torch.FloatTensor([1,1,opt.cov_w,opt.cov_w]).cuda()
    elif opt.cov_dim == 2:
        weight = torch.FloatTensor([1,opt.cov_w]).cuda()
    loss_cov = nn.CrossEntropyLoss(weight=weight)
    start_time = time.time()

    def trainCov(epoch, cov_data, pre_train=False, uni_train=False):
        batch_tot_num = int(len(cov_data)*1.0/opt.cov_batch_size)
        running_loss = 0.0
        total_loss = 0.0
        correct = 0
        total = 0
        with tqdm(enumerate(range(batch_tot_num)),total=batch_tot_num, leave=False, dynamic_ncols=True) as pbar:
            pbar.set_description(str(epoch))
            for i, batch_num in pbar:
                # get the inputs; data is a list of [inputs, labels]
                inputs, _labels = zip(*cov_data[batch_num*opt.cov_batch_size:(batch_num+1)*opt.cov_batch_size])
                inputs= Variable(torch.stack(inputs)).cuda()
                labels= Variable(torch.LongTensor([int(f) for f in _labels])).cuda()
                model.zero_grad()
                outputs = model.encoder.forward_cov(inputs)
                _, predicted = torch.max(outputs.data, 1)
                predicted = predicted.tolist()
                correct += sum([p == l for (p,l) in zip(predicted, _labels)])
                total += len(predicted)
                loss = loss_cov(outputs, labels)
                loss.backward()
                optim.step()
                loss = float(loss.cpu().data)
                running_loss += loss
                total_loss += loss
            # num_words = targets.data.ne(onmt.Constants.PAD).float().sum()
        return total_loss/total, correct*1.0/total

    def trainEpoch(epoch, pre_train=False, uni_train=False):

        if opt.extra_shuffle and epoch > opt.curriculum:
            trainData.shuffle()

        # shuffle mini batch order
        batchOrder = torch.randperm(len(trainData))

        total_loss, total_words, total_other_loss = 0, 0, 0
        report_loss, report_tgt_words, report_src_words, report_other_loss = 0, 0, 0, 0
        report_samples = 0
        total_samples = 0
        correct = 0
        start = time.time()

        len_data = len(trainData) if not opt.test or bool_eval else min(len(trainData), 1000)
        for i in range(len_data):
        # for i in range(10):
            batchIdx = batchOrder[i] if epoch > opt.curriculum else i
            batch = trainData[batchIdx][:-1] # exclude original indices

            model.zero_grad()
            # print batch
            outputs, fert = model(batch, is_cov, pre_train, uni_train)
            # if is_cov:
            #     targets = batch[3]
            #     targets_cov = targets.view(-1).long()
            #     outputs = outputs.view(-1, outputs.size(2))
            #     gradOutput = loss_cov(outputs,targets_cov)
            #     gradOutput.backward()
            #     loss = gradOutput.data[0]
            #     other_loss = 0.0
            #     _, predicted = torch.max(outputs.data, 1)
            #     correct += (predicted == targets_cov.data).sum()
            # else:
            if uni_train:
                targets = batch[2][1:]
            else:
                targets = batch[1][1:]  # exclude <s> from targets
            loss, gradOutput, other_loss = loss_fn(
                    outputs, targets, target_embeddings, target_uni_embeddings, target_ngram_embeddings, fert, model.generator, opt, is_cov, False)
            outputs.backward(gradOutput)

            # update the parameters
            optim.step()
            num_words = targets.data.ne(onmt.Constants.PAD).float().sum()
            report_loss += loss
            report_other_loss += other_loss
            report_tgt_words += num_words
            report_src_words += sum(batch[0][1])
            report_samples += targets.size(1)*1.0
            total_samples += targets.size(1)*1.0
            total_loss += loss
            total_other_loss += other_loss
            total_words += num_words

            if i % opt.log_interval == -1 % opt.log_interval:
                print("Epoch %2d, %5d/%5d; lps: %.5f; mse_lps: %.5f; %3.0f src tok/s; %3.0f tgt tok/s; %3.0f sample/s; %6.0f s elapsed" %
                      (epoch, i+1, len(trainData),
                      report_loss / report_tgt_words,
                      report_other_loss / report_tgt_words,
                      report_src_words/(time.time()-start),
                      report_tgt_words/(time.time()-start),
                      report_samples/(time.time()-start),
                      time.time()-start_time))

                sys.stdout.flush()
                report_loss = report_tgt_words = report_src_words = report_other_loss = report_samples = 0
                start = time.time()

        print ("Epoch %2d, %6.0f samples, %6.0f s" % (epoch, total_samples, time.time()-start_time))
        if is_cov:
            try:
                print("accuracy: "+str(correct/total_words))
                print(targets_cov[:5])
                print(outputs[:5])
            except:
                pass
        return total_loss / total_words, total_other_loss / total_words

    # valid_loss, other_loss = eval(model, loss_fn, target_embeddings, validData)
    valid_loss, other_loss = 0.0, 1.0
    best_valid_lps = valid_loss
    best_other_loss = other_loss
    print('Validation per step loss: %g' % best_valid_lps)
    print('Validation per step other loss: %g' % (other_loss))

    for param in model.encoder.cov_out.parameters():
        param.requires_grad = False
    pre_train=True
    uni_train=True
    cov_train=False

    for epoch in range(opt.start_epoch, opt.start_epoch+opt.epochs):
        print('')
        if epoch == opt.start_epoch + opt.uni_ep:
            uni_train = False

        if epoch == opt.start_epoch + opt.pre_ep:
            print("Pre-training done. Freezing the model to train cov.")
            freeze()
            best_valid_lps, best_other_loss = 1.0, 1.0
            pre_train=False
            uni_train=False
            cov_train=True
        
        if epoch == opt.start_epoch + opt.pre_ep + opt.cov_ep:
            print("Cov training done. Stop the training.")
            best_valid_lps, best_other_loss = 1.0, 1.0
            break
            # unfreeze(model)
            # cov_train=False
        
        # is_cov = (opt.start_epoch+opt.pre_ep<=epoch)
        #  (1) train for one epoch on the training set
        if not cov_train:
            train_loss, train_acc = trainEpoch(epoch, pre_train=pre_train)
        else:
            train_loss, train_acc = trainCov(epoch, cov_data, pre_train=pre_train)
        if uni_train:
            _, _ = trainEpoch(epoch, pre_train=pre_train, uni_train=uni_train)
        train_lps = train_loss
        print('Train per step loss: %g' % train_lps)
        # print('Train accuracy: %g' % (train_acc*100))

        #  (2) evaluate on the validation set
        if not cov_train:
            valid_loss, other_loss = eval(model, loss_fn, target_embeddings, target_uni_embeddings, target_ngram_embeddings, validData, is_cov=cov_train, pre_train=pre_train)
        else:
            print("Epoch "+ str(epoch))
            valid_loss, other_loss = evalCov(model, loss_cov, cov_data_val)
        valid_lps = valid_loss
        print('Validation per step loss: %g' % valid_loss)
        print('Validation per step other loss: %g' % (other_loss))

        sys.stdout.flush()
        #  (3) update the learning rate
        # optim.updateLearningRate(valid_loss, epoch)

        model_state_dict = model.module.state_dict() if len(opt.gpus) > 1 else model.state_dict()
        model_state_dict = {k: v for k, v in model_state_dict.items() if 'generator' not in k}
        generator_state_dict = model.generator.module.state_dict() if len(opt.gpus) > 1 else model.generator.state_dict()

        checkpoint = {
        #    'model': model_state_dict,
            'encoder': model.encoder.state_dict(),
            'decoder': model.decoder.state_dict(),
            'generator': generator_state_dict,
            'dicts': dataset['dicts'],
            'opt': opt,
            'epoch': epoch,
            'optim': optim
        }

        torch.save(checkpoint,
                   '%s_latest.pt' % (opt.save_model))

        if cov_train:
          torch.save(checkpoint,'%s_model_%d.pt' % (opt.save_model,epoch))

        if opt.save_all_epochs:
          torch.save(checkpoint,'%s_model_%d.pt' % (opt.save_model,epoch))

        if best_valid_lps > valid_lps: #in case of vMF loss, if the loss is the same, has the cosine loss decreased?
        # if best_other_loss > other_loss:
            best_valid_lps = valid_lps
            best_other_loss = other_loss
            print ("Best model found!")
            if pre_train:
                torch.save(checkpoint,
                    '%s_bestmodel_pre.pt' % opt.save_model)
            elif cov_train:
                torch.save(checkpoint,
                    '%s_bestmodel_cov.pt' % opt.save_model)

        elif best_valid_lps == valid_lps: #in case of vMF loss, if the loss is the same, has the cosine loss decreased?
            if best_other_loss > other_loss:
                best_other_loss = other_loss
                print ("Best model found!")
                torch.save(checkpoint, '%s_bestmodel.pt' % opt.save_model)

def main():

    print("Loading data from '%s'" % opt.data)
    dataset = torch.load(opt.data)

    dict_checkpoint = opt.train_from if opt.train_from else None

    if dict_checkpoint:
        print('Loading dicts from checkpoint at %s' % dict_checkpoint)
        checkpoint = torch.load(dict_checkpoint)
        dataset['dicts'] = checkpoint['dicts']

    trainData = onmt.Dataset(dataset['train']['src'],
                             dataset['train']['tgt'],
                             dataset['train']['tgt_uni'], 
                             dataset['train']['align'], opt.batch_size, opt.gpus)
    validData = onmt.Dataset(dataset['valid']['src'],
                             dataset['valid']['tgt'], 
                             dataset['valid']['tgt_uni'], 
                             dataset['valid']['align'], opt.batch_size, opt.gpus,
                             volatile=True)

    dicts = dataset['dicts']
    print(' * vocabulary size. source = %d; target = %d' %
          (dicts['src'].size(), dicts['tgt'].size()))
    print(' * number of training sentences. %d' %
          len(dataset['train']['src']))
    print(' * maximum batch size. %d' % opt.batch_size)

    print('Building model...')

    encoder = onmt.Models.Encoder(opt, dicts['src'], opt.fix_src_emb, use_cov=True)
    decoder = onmt.Models.Decoder(opt, dicts['tgt'], opt.tie_emb)

    output_dim = opt.output_emb_size

    if not opt.nonlin_gen:
        generator = nn.Sequential(nn.Linear(opt.rnn_size, output_dim))
    else: #add a non-linear layer before generating the continuous vector
        generator = nn.Sequential(nn.Linear(opt.rnn_size, output_dim), nn.ReLU(), nn.Linear(output_dim, output_dim))

    #output is just an embedding
    target_embeddings = nn.Embedding(dicts['tgt'].size(), opt.output_emb_size)
    target_uni_embeddings = nn.Embedding(dicts['tgt'].size_uni(), opt.output_emb_size)
    target_ngram_embeddings = nn.Embedding(dicts['tgt'].size_ngram(), opt.output_emb_size)

    #normalize the embeddings
    norm = dicts['tgt'].embeddings.norm(p=2, dim=1, keepdim=True).clamp(min=1e-12)
    target_embeddings.weight.data.copy_(dicts['tgt'].embeddings.div(norm))

    norm = dicts['tgt'].unigram_embeddings.norm(p=2, dim=1, keepdim=True).clamp(min=1e-12)
    target_uni_embeddings.weight.data.copy_(dicts['tgt'].unigram_embeddings.div(norm))

    norm = dicts['tgt'].ngram_embeddings.norm(p=2, dim=1, keepdim=True).clamp(min=1e-12)
    target_ngram_embeddings.weight.data.copy_(dicts['tgt'].ngram_embeddings.div(norm))

    #target embeddings are fixed and not trained
    target_embeddings.weight.requires_grad=False
    target_uni_embeddings.weight.requires_grad=False
    target_ngram_embeddings.weight.requires_grad=False

    # elif opt.loss != "maxmargin": # with max-margin loss, the target embeddings can be fine-tuned as well.
        # target_embeddings.weight.requires_grad=False

    model = onmt.Models.NMTModel(encoder, decoder)

    if opt.train_from:
        print('Loading model from checkpoint at %s' % opt.train_from)
        generator_state_dict = checkpoint['generator']
        encoder_state_dict = [('encoder.'+k,v) for k, v in checkpoint['encoder'].items()]
        decoder_state_dict = [('decoder.'+k,v) for k, v in checkpoint['decoder'].items()]
        model_state_dict = dict(encoder_state_dict+decoder_state_dict)

        model.load_state_dict(model_state_dict, strict=False)
        generator.load_state_dict(generator_state_dict)

        if not opt.train_anew: #load from
            opt.start_epoch = checkpoint['epoch'] + 1

    if len(opt.gpus) >= 1:
        model.cuda()
        generator.cuda()
        target_embeddings.cuda()
        target_uni_embeddings.cuda()
        target_ngram_embeddings.cuda()
    else:
        model.cpu()
        generator.cpu()
        target_embeddings.cpu()
        target_uni_embeddings.cpu()
        target_ngram_embeddings.cpu()

    if len(opt.gpus) > 1:
        model = nn.DataParallel(model, device_ids=opt.gpus, dim=1)
        generator = nn.DataParallel(generator, device_ids=opt.gpus, dim=0)

    model.generator = generator

    if not opt.train_from:
        for p in model.parameters():
            p.data.uniform_(-opt.param_init, opt.param_init)

        encoder.load_pretrained_vectors(opt)
        decoder.load_pretrained_vectors(opt)

        if opt.tie_emb:
            decoder.tie_embeddings(target_embeddings)

        if opt.fix_src_emb:
            #fix and normalize the source embeddings
            source_embeddings = nn.Embedding(dicts['src'].size(), opt.output_emb_size)
            norm = dicts['src'].embeddings.norm(p=2, dim=1, keepdim=True).clamp(min=1e-12)
            source_embeddings.weight.data.copy_(dicts['src'].embeddings.div(norm))

            #turn this off to initialize embeddings as well as make them trainable
            source_embeddings.weight.requires_grad=False
            if len(opt.gpus) >= 1:
                source_embeddings.cuda()
            else:
                source_embeddings.cpu()
            encoder.fix_embeddings(source_embeddings)

        optim = onmt.Optim(
            opt.optim, opt.learning_rate, opt.max_grad_norm,
            lr_decay=opt.learning_rate_decay,
            start_decay_at=opt.start_decay_at
        )
    elif opt.train_anew: #restart optimizer, sometimes useful for training with
        optim = onmt.Optim(
            opt.optim, opt.learning_rate, opt.max_grad_norm,
            lr_decay=opt.learning_rate_decay,
            start_decay_at=opt.start_decay_at
        )
    else:
        print('Loading optimizer from checkpoint:')
        optim = checkpoint['optim']
        print(optim)

    optim.set_parameters(model.parameters())

    if opt.train_from and not opt.train_anew:
        optim.optimizer.load_state_dict(checkpoint['optim'].optimizer.state_dict())

    nParams = sum([p.nelement() for p in model.parameters() if p.requires_grad])
    print('* number of trainable parameters: %d' % nParams)

    trainModel(model, trainData, validData, dataset, target_embeddings, target_uni_embeddings, target_ngram_embeddings, optim)

if __name__ == "__main__":
    main()
