#!/bin/bash

PATH_DATA=$1
SRC_LANG=$2
TGT_LANG=$3
PATH_FASTALIGN=$4

for TRAIN_TEST_DEV in "train" "test" "valid"
do
paste -d '\t' "$PATH_DATA/$TRAIN_TEST_DEV.tok.true.$SRC_LANG" "$PATH_DATA/$TRAIN_TEST_DEV.tok.true.$TGT_LANG" | sed 's/\t/ ||| /' > "$PATH_DATA/$TRAIN_TEST_DEV.concat"
$PATH_FASTALIGN/build/fast_align -i "$PATH_DATA/$TRAIN_TEST_DEV.concat" -d -o -v > "$PATH_DATA/forward.align" 
$PATH_FASTALIGN/build/fast_align -i "$PATH_DATA/$TRAIN_TEST_DEV.concat" -d -o -v -r > "$PATH_DATA/reverse.align"
$PATH_FASTALIGN/build/atools -i "$PATH_DATA/forward.align" -j "$PATH_DATA/reverse.align" -c grow-diag-final-and > "$PATH_DATA/$TRAIN_TEST_DEV.align" 
done

rm "$PATH_DATA/forward.align"
rm "$PATH_DATA/reverse.align"
rm $PATH_DATA/*.concat
