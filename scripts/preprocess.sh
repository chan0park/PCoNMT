#!/bin/bash
# usage: preprocess.sh data de en path/to/mosesdecoder

export PATH_DATA=$1
export SRC_LANG=$2
export TGT_LANG=$3
export PATH_MOSES=$4

for TRAIN_TEST_DEV in "train" "test" "valid"
do
$PATH_MOSES/scripts/tokenizer/tokenizer.perl -l $TGT_LANG -a -no-escape -threads 20 < $PATH_DATA/$TRAIN_TEST_DEV.$TGT_LANG > $PATH_DATA/$TRAIN_TEST_DEV.tok.$TGT_LANG
$PATH_MOSES/scripts/tokenizer/tokenizer.perl -l $SRC_LANG -a -no-escape -threads 20 < $PATH_DATA/$TRAIN_TEST_DEV.$SRC_LANG > $PATH_DATA/$TRAIN_TEST_DEV.tok.$SRC_LANG
done

# $PATH_MOSES/scripts/tokenizer/tokenizer.perl -l en -a -no-escape -threads 20 < $PATH_DATA/train.$TGT_LANG > $PATH_DATA/train.tok.$TGT_LANG
# $PATH_MOSES/scripts/tokenizer/tokenizer.perl -l de -a -no-escape -threads 20 < $PATH_DATA/train.$SRC_LANG > $PATH_DATA/train.tok.$SRC_LANG
# $PATH_MOSES/scripts/tokenizer/tokenizer.perl -l en -a -no-escape -threads 20 < $PATH_DATA/test.$TGT_LANG > $PATH_DATA/test.tok.$TGT_LANG
# $PATH_MOSES/scripts/tokenizer/tokenizer.perl -l de -a -no-escape -threads 20 < $PATH_DATA/test.$SRC_LANG > $PATH_DATA/test.tok.$SRC_LANG
# $PATH_MOSES/scripts/tokenizer/tokenizer.perl -l en -a -no-escape -threads 20 < $PATH_DATA/valid.$TGT_LANG > $PATH_DATA/valid.tok.$TGT_LANG
# $PATH_MOSES/scripts/tokenizer/tokenizer.perl -l de -a -no-escape -threads 20 < $PATH_DATA/valid.$SRC_LANG > $PATH_DATA/valid.tok.$SRC_LANG
#repeat similar steps for tokenizing val and test sets

$PATH_MOSES/scripts/recaser/train-truecaser.perl --model $PATH_DATA/truecaser.model.$TGT_LANG --corpus $PATH_DATA/train.tok.$TGT_LANG
$PATH_MOSES/scripts/recaser/train-truecaser.perl --model $PATH_DATA/truecaser.model.$SRC_LANG --corpus $PATH_DATA/train.tok.$SRC_LANG

for TRAIN_TEST_DEV in "train" "test" "valid"
do
$PATH_MOSES/scripts/recaser/truecase.perl --model $PATH_DATA/truecaser.model.$SRC_LANG < $PATH_DATA/$TRAIN_TEST_DEV.tok.$SRC_LANG > $PATH_DATA/$TRAIN_TEST_DEV.tok.true.$SRC_LANG
$PATH_MOSES/scripts/recaser/truecase.perl --model $PATH_DATA/truecaser.model.$TGT_LANG < $PATH_DATA/$TRAIN_TEST_DEV.tok.$TGT_LANG > $PATH_DATA/$TRAIN_TEST_DEV.tok.true.$TGT_LANG
done

# $PATH_MOSES/scripts/recaser/truecase.perl --model $PATH_DATA/truecaser.model.$TGT_LANG < $PATH_DATA/train.tok.$TGT_LANG > $PATH_DATA/train.tok.true.$TGT_LANG
# $PATH_MOSES/scripts/recaser/truecase.perl --model $PATH_DATA/truecaser.model.$SRC_LANG < $PATH_DATA/train.tok.$SRC_LANG > $PATH_DATA/train.tok.true.$SRC_LANG
# $PATH_MOSES/scripts/recaser/truecase.perl --model $PATH_DATA/truecaser.model.$TGT_LANG < $PATH_DATA/test.tok.$TGT_LANG > $PATH_DATA/test.tok.true.$TGT_LANG
# $PATH_MOSES/scripts/recaser/truecase.perl --model $PATH_DATA/truecaser.model.$SRC_LANG < $PATH_DATA/test.tok.$SRC_LANG > $PATH_DATA/test.tok.true.$SRC_LANG
# $PATH_MOSES/scripts/recaser/truecase.perl --model $PATH_DATA/truecaser.model.$TGT_LANG < $PATH_DATA/valid.tok.$TGT_LANG > $PATH_DATA/valid.tok.true.$TGT_LANG
# $PATH_MOSES/scripts/recaser/truecase.perl --model $PATH_DATA/truecaser.model.$SRC_LANG < $PATH_DATA/valid.tok.$SRC_LANG > $PATH_DATA/valid.tok.true.$SRC_LANG

rm $PATH_DATA/truecaser.model.$TGT_LANG
rm $PATH_DATA/truecaser.model.$SRC_LANG
rm $PATH_DATA/*.tok.$SRC_LANG
rm $PATH_DATA/*.tok.$TGT_LANG