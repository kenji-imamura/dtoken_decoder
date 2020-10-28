#! /bin/sh
#
# Make binalized dataset
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

#
# Usage
#
usage_exit () {
    echo "Usage $0 [-s SRC] [-t TRG]" 1>&2
    exit 1
}

#
# Options
#
while getopts s:t:h OPT; do
    case $OPT in
        s)  SRC=$OPTARG
            ;;
        t)  TRG=$OPTARG
            ;;
        h)  usage_exit
            ;;
        \?) usage_exit
            ;;
    esac
done
shift $((OPTIND - 1))

#
# Main
#

mkdir -p $DATA

### Encode corpora into binary sets.
fairseq-preprocess \
    --workers 4 \
    --source-lang $SRC --target-lang $TRG \
     --joined-dictionary \
    --trainpref $CORPUS/train.bpe \
    --validpref $CORPUS/newstest2013.bpe \
    --testpref $CORPUS/newstest2014.bpe,$CORPUS/newstest2015.bpe \
    --destdir $DATA
