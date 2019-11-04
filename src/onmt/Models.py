import torch
import torch.nn as nn
from torch.autograd import Variable
import onmt.modules
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn import functional as F

class Encoder(nn.Module):

    def __init__(self, opt, dicts, src_fix_emb=False, use_fert=True, fert_mode="emh"):
        self.layers = opt.layers
        self.num_directions = 2 if opt.brnn else 1
        self.src_fix_emb = src_fix_emb
        assert opt.rnn_size % self.num_directions == 0
        self.hidden_size = opt.rnn_size // self.num_directions
        input_size = opt.word_vec_size

        super(Encoder, self).__init__()

        if src_fix_emb:
            self.word_lut = nn.Embedding(dicts.size(), opt.output_emb_size, padding_idx=onmt.Constants.PAD)
            self.emb2input = nn.Linear(opt.output_emb_size, opt.word_vec_size)
        else:
            self.word_lut = nn.Embedding(dicts.size(),
                                  opt.word_vec_size,
                                  padding_idx=onmt.Constants.PAD)


        # self.word_lut = nn.Embedding(dicts.size(),
        #                           opt.word_vec_size,
        #                           padding_idx=onmt.Constants.PAD)
        self.rnn = nn.LSTM(input_size, self.hidden_size,
                        num_layers=opt.layers,
                        dropout=opt.dropout,
                        bidirectional=opt.brnn)

        self.use_fert = use_fert
        if use_fert:
            self.sigmoid = nn.Sigmoid()
            self.softmax = nn.Softmax(dim=-1)
            self.fert_mode = opt.fert_mode
            if self.fert_mode == "emb":
                self.cov_fc1 = nn.Linear(opt.word_vec_size, opt.cov_hidden_dim)
            elif self.fert_mode == "emh":
                self.cov_fc1 = nn.Linear(opt.word_vec_size+opt.rnn_size, opt.cov_hidden_dim)
            else:
                self.cov_fc1 = nn.Linear(opt.rnn_size, opt.cov_hidden_dim)
            self.cov_fc2 = nn.Linear(opt.cov_hidden_dim, 1024)
            self.cov_fc3 = nn.Linear(1024, opt.fert_dim)

    def load_pretrained_vectors(self, opt):
        if opt.pre_word_vecs_enc is not None:
            pretrained = torch.load(opt.pre_word_vecs_enc)
            self.word_lut.weight.data.copy_(pretrained)

    def _fix_enc_hidden(self, h):
        #  the encoder hidden is  (layers*directions) x batch x dim
        #  we need to convert it to layers x batch x (directions*dim)
        if self.num_directions == 2:
            return h.view(h.size(0) // 2, 2, h.size(1), h.size(2)) \
                    .transpose(1, 2).contiguous() \
                    .view(h.size(0) // 2, h.size(1), h.size(2) * 2)
        else:
            return h

    def forward_cov(self, cov_input):
        cov = F.relu(self.cov_fc1(cov_input))
        cov = F.relu(self.cov_fc2(cov))
        cov = self.cov_fc3(cov)
        # cov = self.softmax(cov)
        return cov


    def forward(self, input, hidden=None, is_fert=True):
        if isinstance(input, tuple):
            emb_ = self.word_lut(input[0])
            if self.src_fix_emb:
                emb_ = self.emb2input(emb_)
                emb = pack(emb_, list(input[1]))
            else:
                emb = pack(emb_, list(input[1]))
        else:
            emb_ = self.word_lut(input)
            if self.src_fix_emb:
                emb = self.emb2input(emb_)
            else:
                emb = emb_

        # if isinstance(input, tuple):
        #     emb = pack(self.word_lut(input[0]), list(input[1]))
        # else:
        #     emb = self.word_lut(input)
        outputs, hidden_t = self.rnn(emb, hidden)
        if isinstance(input, tuple):
            outputs = unpack(outputs)[0]

        cov = None
        if self.use_fert:
            if self.fert_mode == "emb":
                cov_inp = emb_
            elif self.fert_mode == "emh":
                cov_inp = torch.cat([emb_, outputs],-1)
            else:
                cov_inp = outputs
            if is_fert:
                cov = self.forward_cov(cov_inp)

        hidden_t = (self._fix_enc_hidden(hidden_t[0]),
                      self._fix_enc_hidden(hidden_t[1]))

        return hidden_t, outputs, cov, cov_inp

    def fix_embeddings(self, embeddings):
        self.word_lut = embeddings

class StackedLSTM(nn.Module):
    def __init__(self, num_layers, input_size, rnn_size, dropout):
        super(StackedLSTM, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            self.layers.append(nn.LSTMCell(input_size, rnn_size))
            input_size = rnn_size

    def forward(self, input, hidden):
        h_0, c_0 = hidden
        h_1, c_1 = [], []
        for i, layer in enumerate(self.layers):
            h_1_i, c_1_i = layer(input, (h_0[i], c_0[i]))
            input = h_1_i
            if i + 1 != self.num_layers:
                input = self.dropout(input)
            h_1 += [h_1_i]
            c_1 += [c_1_i]

        h_1 = torch.stack(h_1)
        c_1 = torch.stack(c_1)

        return input, (h_1, c_1)


class Decoder(nn.Module):

    def __init__(self, opt, dicts, tie_emb=False):
        self.layers = opt.layers
        self.input_feed = opt.input_feed
        self.tie_emb = tie_emb
        input_size = opt.word_vec_size
        if self.input_feed:
            input_size += opt.rnn_size

        super(Decoder, self).__init__()
        if tie_emb:
            self.word_lut = nn.Embedding(dicts.size(), opt.output_emb_size, padding_idx=onmt.Constants.PAD)
            self.emb2input = nn.Linear(opt.output_emb_size, opt.word_vec_size)
        else:
            self.word_lut = nn.Embedding(dicts.size(),
                                  opt.word_vec_size,
                                  padding_idx=onmt.Constants.PAD)

        self.rnn = StackedLSTM(opt.layers, input_size, opt.rnn_size, opt.dropout)
        self.attn = onmt.modules.GlobalAttention(opt.rnn_size)
        # self.attn = onmt.modules.GlobalAttentionOriginal(opt.rnn_size)
        self.dropout = nn.Dropout(opt.dropout)

        self.hidden_size = opt.rnn_size

    def load_pretrained_vectors(self, opt):
        if opt.pre_word_vecs_dec is not None:
            pretrained = torch.load(opt.pre_word_vecs_dec)
            self.word_lut.weight.data.copy_(pretrained)

    def tie_embeddings(self, embeddings):
        self.word_lut = embeddings

    def forward(self, input, hidden, context, init_output):
        emb_ = self.word_lut(input)
        if self.tie_emb:
            emb = self.emb2input(emb_)
        else:
            emb = emb_
        #print(context.size())
        # n.b. you can increase performance if you compute W_ih * x for all
        # iterations in parallel, but that's only possible if
        # self.input_feed=False
        outputs = []
        output = init_output
        # cov = None
        # if is_cov:
            # src_len, batch_size = context.size(0), context.size(1)
            # cov = torch.zeros(batch_size, src_len)

        for emb_t in emb.split(1):
            emb_t = emb_t.squeeze(0)
            if self.input_feed:
                emb_t = torch.cat([emb_t, output], 1)

            output, hidden = self.rnn(emb_t, hidden)
            output, attn = self.attn(output, context.transpose(0, 1))
            output = self.dropout(output)
            outputs += [output]

        outputs = torch.stack(outputs)
        return outputs, hidden, attn


class NMTModel(nn.Module):

    def __init__(self, encoder, decoder):
        super(NMTModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def make_init_decoder_output(self, context):
        batch_size = context.size(1)
        h_size = (batch_size, self.decoder.hidden_size)
        return Variable(context.data.new(*h_size).zero_(), requires_grad=False)

    def _fix_enc_hidden(self, h):
        #  the encoder hidden is  (layers*directions) x batch x dim
        #  we need to convert it to layers x batch x (directions*dim)
        if self.encoder.num_directions == 2:
            return h.view(h.size(0) // 2, 2, h.size(1), h.size(2)) \
                    .transpose(1, 2).contiguous() \
                    .view(h.size(0) // 2, h.size(1), h.size(2) * 2)
        else:
            return h

    def forward(self, input, is_fert, pre_train, use_uni=False):
        src = input[0]
        if use_uni:
            tgt = input[2][:-1] # unigram target text
        else:
            tgt = input[1][:-1]  # exclude last target from inputs
        enc_hidden, context, fert, _ = self.encoder(src, is_fert=(not pre_train))
        if is_fert:
            return fert, None

        init_output = self.make_init_decoder_output(context)

        if pre_train:
            fert = input[3]
        else:
            fert = torch.max(fert, dim=-1)[1].float()

        out, dec_hidden, _attn = self.decoder(tgt, enc_hidden, context, init_output)
        return out, fert
