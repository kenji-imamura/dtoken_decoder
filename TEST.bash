#! /bin/bash
#
# Batch translation 
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
TESTSET=newstest2013
BEAM=10
PENALTY=1.0
GPUID=
OUTPUT=

declare -A SUBSETS=(
    ["train"]="train"
    ["newstest2013"]="valid"
    ["newstest2014"]="test"
    ["newstest2015"]="test1"
)

#
# Usage
#
usage_exit () {
    echo "Usage $0 [-g GPUIDs] [-c TESTSET] output_prefix" 1>&2
    exit 1
}

#
# Options
#
while getopts s:t:g:c:h OPT; do
    case $OPT in
        s)  SRC=$OPTARG
            ;;
        t)  TRG=$OPTARG
            ;;
        g)  GPUID=$OPTARG
            ;;
        c)  TESTSET=$OPTARG
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
if [ $# -lt 1 ]; then
    usage_exit
else
    OUTPUT=$1
fi
SUBSET=${SUBSETS[$TESTSET]}

#
# Translation
#

test_main () {
    output=$1
    ### Translation
    fairseq-generate $DATA -s $SRC -t $TRG \
	--fp16 \
	--user-dir $CODE --task dtoken_translation \
	--no-progress-bar \
	--gen-subset $SUBSET \
	--path $MODEL/checkpoint_best.pt \
	--lenpen $PENALTY --beam $BEAM --batch-size 32 \
	> $output.log

    ### Convert sub-words into words
    cat $output.log \
	| grep -e '^H\-' | sed -e 's/^H-//' \
	| sort -k 1 -n | cut -f 3 \
	| spm_decode --model=$CORPUS/train.spm.share.model \
		     --input_format=piece > $output.out
}
test_main $OUTPUT
