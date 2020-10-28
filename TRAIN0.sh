#! /bin/sh
#
# Tokenize raw sentences
#
trap 'exit 2' 2
DIR=$(cd $(dirname $0); pwd)

CORPUS=$DIR/corpus
SRC=en
TRG=de

#
# Download the corpus
#
mkdir -p $CORPUS
for prefix in train newstest2012 newstest2013 newstest2014 newstest2015; do
    for lang in $SRC $TRG; do
	file=$prefix.$lang
	if [ !  -f $CORPUS/$file ]; then
	    wget -P $CORPUS https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/$file
	fi
    done
done

#
# Train sub-word models
#

### sentencepiece
sp_train () {
    size=$1
    spm_train --model_prefix=$CORPUS/train.spm.share \
	      --input=$CORPUS/train.$SRC,$CORPUS/train.$TRG \
	      --vocab_size=$size \
	      --character_coverage=1.0 \
	      > $CORPUS/train.spm.share.log 2>&1
}

sp_train 32768 &
wait

#
# Apply the sub-word models
#

### sentencepiece
sp_encode () {
    lang=$1
    testset=$2
    cat $CORPUS/${testset}.$lang \
	| spm_encode --model=$CORPUS/train.spm.share.model \
	> $CORPUS/${testset}.bpe.$lang
}

for testset in train newstest2012 newstest2013 newstest2014 newstest2015; do
    sp_encode $SRC $testset &
    sp_encode $TRG $testset &
    wait
done
