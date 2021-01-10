#!/bin/bash
# Adapted from https://github.com/facebookresearch/MIXER/blob/master/prepareData.sh

SCRIPTS=mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
NORM_PUNC=$SCRIPTS/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$SCRIPTS/tokenizer/remove-non-printing-char.perl
BPEROOT=subword-nmt/subword_nmt
BPE_TOKENS=40000

#
# Read arguments
#
POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"
case $key in
  --src)
    src="$2"; shift 2;;
  --tgt)
    tgt="$2"; shift 2;;
  --prep)
    prep="$2"; shift 2;;
  *)
  POSITIONAL+=("$1")
  shift
  ;;
esac
done
set -- "${POSITIONAL[@]}"

if [ "$src" == "" ]; then echo "--src not provided"; exit; fi
if [ "$tgt" == "" ]; then echo "--tgt not provided"; exit; fi
if [ "$prep" == "" ]; then echo "--prep not provided"; exit; fi

lang=$src-$tgt
tmp=$prep/tmp
orig=orig

mkdir -p $tmp $prep

echo "pre-processing data..."
for splt in train valid test; do
    echo "pre-processing ${splt} data..."
    for l in $src $tgt; do
        if [ -f $tmp/${splt}.$l ]; then
            rm $tmp/${splt}.$l
        fi
        cat $orig/$prep/${splt}.$lang.$l | \
            perl $NORM_PUNC $l | \
            perl $REM_NON_PRINT_CHAR | \
            perl $TOKENIZER -threads 8 -a -l $l > $tmp/${splt}.$l
    done
done

TRAIN=$tmp/train.$lang
BPE_CODE=$prep/code
if [ -f $TRAIN]; then
    rm -f $TRAIN
fi

for l in $src $tgt; do
    cat $tmp/train.$l >> $TRAIN
done

echo "learn_bpe.py on ${TRAIN}..."
python $BPEROOT/learn_bpe.py -s $BPE_TOKENS < $TRAIN > $BPE_CODE

for L in $src $tgt; do
    for f in train.$L valid.$L test.$L; do
        echo "apply_bpe.py to ${f}..."
        python $BPEROOT/apply_bpe.py -c $BPE_CODE < $tmp/$f > $tmp/bpe.$f
    done
done

perl $CLEAN -ratio 1.5 $tmp/bpe.train $src $tgt $prep/train 1 250
perl $CLEAN -ratio 1.5 $tmp/bpe.valid $src $tgt $prep/valid 1 250

for L in $src $tgt; do
    cp $tmp/bpe.test.$L $prep/test.$L
done
