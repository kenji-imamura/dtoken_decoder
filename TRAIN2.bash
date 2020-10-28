#! /bin/bash
#
# Decoder training
#

trap 'exit 2' 2
DIR=$(cd $(dirname $0); pwd)
. $DIR/../../setenv.sh

CODE=$DIR/user_code
export PYTHONPATH="$CODE:$PYTHONPATH"

CORPUS=$DIR/corpus
DATA=$DIR/data
SRC=en
TRG=de

MODEL=model.$SRC-$TRG
DROPOUT=0.1
LR=0.0004
WARMUP_EPOCHS=5
TRAIN_EPOCHS=50

### Set the mini-batch size to around 500 sentences.
GPUID=
TOKENS_PER_GPU=5000
UPDATE_FREQ=4

#
# Usage
#
usage_exit () {
    echo "Usage $0 [-s SRC] [-t TRG] [-g GPUIDs]" 1>&2
    exit 1
}

#
# Options
#
while getopts g:s:t:h OPT; do
    case $OPT in
        s)  SRC=$OPTARG
            ;;
        t)  TRG=$OPTARG
            ;;
        g)  GPUID=$OPTARG
            ;;
        h)  usage_exit
            ;;
        \?) usage_exit
            ;;
    esac
done
shift $((OPTIND - 1))
if [ -n "$GPUID" ]; then
    export CUDA_VISIBLE_DEVICES=$GPUID
fi

#
# Training
#

### Set training parameters related to a mini-batch.
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    echo "CUDA Devices: $CUDA_VISIBLE_DEVICES" 1>&2
    num_gpus=`echo $CUDA_VISIBLE_DEVICES | awk -F, "{print NF}"`
    UPDATE_FREQ=$((4 / $num_gpus))
fi

### The corpus size is around 4.5 million sentences.
### The mini-batch size is set to around 500 sentences. 
UPDATES_PER_EPOCH=9000
DISP_FREQ=$((UPDATES_PER_EPOCH / 5))
WARMUP_FLOAT=`echo "$UPDATES_PER_EPOCH * $WARMUP_EPOCHS" | bc`
WARMUP=`printf "%.0f" $WARMUP_FLOAT`

training () {
    model=$1
    date
    fairseq-train $DATA -s $SRC -t $TRG \
	--ddp-backend no_c10d \
	--user-dir $CODE --task dtoken_translation \
	--arch dtoken_transformer \
	--share-all-embeddings \
	--fp16 \
	--no-progress-bar --log-format simple \
	--log-interval $DISP_FREQ \
	--max-tokens $TOKENS_PER_GPU --update-freq $UPDATE_FREQ \
	--max-epoch $TRAIN_EPOCHS \
	--optimizer adam --lr $LR --adam-betas '(0.9, 0.99)' \
	--label-smoothing 0.1 --clip-norm 5 \
	--dropout $DROPOUT \
	--min-lr '1e-09' --lr-scheduler inverse_sqrt \
	--weight-decay 0.0001 \
	--criterion label_smoothed_cross_entropy \
	--warmup-updates $WARMUP --warmup-init-lr '1e-07' \
	--save-dir $model
    date
}

mkdir -p $MODEL
training $MODEL > $MODEL/training.log