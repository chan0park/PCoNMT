Code for [Learning to Generate Word- and Phrase-Embeddings for Efficient Phrase-Based Neural Machine Translation](https://chan0park.github.io/papers/2019-wngt-pconmt.pdf) (_Park_ and _Tsvetkov_, 2019)

This code is adapted from an earlier version of Sachin Kumar's [seq2seq-con](https://github.com/Sachin19/seq2seq-con) code.

---
### Dependencies

- Phrase extraction
  - Python 3.5
  - nltk
  - fasttext

- Model training
  - __Pytorch 0.3.0__
  - Python 2.7

---

### Running Experiments
There are four sub-steps required to run experiments:
1. Preprocessing
2. Word alignments extraction
3. Phrase embeddings extraction
4. Training/evaluation

#### 1. Preprocessing
* Tokenization and Truecasing (Using [Moses Scripts](https://github.com/moses-smt/mosesdecoder))

Assuming you have (train.de, train.en, test.de, test.en, valid.de, valid.en) under data/, you can obtain tokenized and truecased files by running the following script:

```
./scripts/preprocess.sh data de en ./path/to/mosesdecoder
```

#### 2. Word Alignments Extraction
* Word alignments (Using [fast_align](https://github.com/clab/fast_align.git)) 

By running following command, you will get {train,test,valid}.align files
```
./scripts/get_align.sh data de en ./path/to/fast_align
```

#### 3. Phrase Embeddings
For the phrase and word embedding tables, you need fasttext embeddings that are trained on the same dimension. In our experiments, we first used parallel corpus and alignment to extract phrase list, and then used it to concatenate words in a large monolingual corpus. We then trained fasttext embedding using the corpus. 

You can train your own embedding using the same method, or download our trained model and extracted embeddings from [here](). 

Once you obtain fasttext embeddings and the model, run the following command to get concatenated target txt files and the phrase embeddings. Since you need to use _fasttext_ python module here, it is required to use Python 3.5+.

```
python src/get_phrases.py data de en embs/fasttext.phrase.300.en.bin
```

#### 4. Model Training/evaluation
Note that in our model training and evaluation, we use _pytorch 0.3.0.post4_ and _Python 2.7_.


__Creating preprocessed data object__
```
python src/prepare_data.py -train_src data/train.tok.true.de -train_tgt data/train.tok.true.mwe.en -train_align data/train.mwe.align \
-valid_src data/valid.tok.true.de -valid_tgt data/valid.tok.true.mwe.en -valid_align data/valid.mwe.align -save_data data/deen.pconmt \
-src_vocab_size 50000 -tgt_vocab_size 100000 -tgt_emb embs/fasttext.mwe.word.en.vec -tgt_emb_phrase data/mwe_list.mwe.vec -emb_dim 300 -normalize
```


__Training a model__

```
python src/train.py -data data/deen.pconmt.train.pt -layers 2 -rnn_size 1024 -word_vec_size 512 -output_emb_size 300 -brnn -loss nllvmf -optim adam -dropout 0.0 -learning_rate 0.0005 -log_interval 500 -save_model models/deen -batch_size 16 -tie_emb -gpus 0 -pre_ep 7 -fert_ep 10 -epochs 17 -fert_mode emh -uni_ep 0 -fert_dim 4
```

__Evaluating a model without the fertility prediction__
```
python src/translate.py -loss nllvmf -gpu 0 -replace_unk -model models/deen_bestmodel_pre.pt -src data/test.tok.true.de -tgt data/test.tok.true.en -output deen.out -batch_size 512 -beam_size 1
```

__Evaluating a model with the fertility prediction__
```
python src/translate_fert.py -loss nllvmf -gpu 0 -replace_unk -model models/dee_bestmodel_fert.pt -src data/test.tok.true.de -tgt data/test.tok.true.en -output deen.fert.out -batch_size 512 -beam_size 1
```

---
### Pointers for baselines in the paper
- [Attn](https://github.com/harvardnlp/seq2seq-attn)
- [NPMT](https://github.com/posenhuang/NPMT)
- [CoNMT](https://github.com/Sachin19/seq2seq-con)