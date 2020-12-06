# Transformer-based Double-token Bidirectional Autoregressive Decoding in Neural Machine Translation (Imamura and Sumita, 2020)

This is a sample code, in which the title paper
[https://undecided_URL] is implemented as a plugin of the fairseq
v0.9.0.  It replaces the decoder and beam search with the proposed
ones.

The preprocess is eqaul to that of the original fairseq, but
the training and evaluation phases including the learned model are
different.


## Example usage
This example assumes the following data.

- The corpus used here is 
[WMT-2014 En-De Corpus](https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/)
preprocessed by the Stanford NLP Group.

### Requirements
This example is implemented as a plugin of the [fairseq](https://github.com/pytorch/fairseq) translator.
```
pip3 install fairseq
```

### Directories / paths
```
#! /bin/bash
export CODE=./user_code
export CORPUS=./corpus
export DATA=./data
export MODEL=./model.en-de
export PYTHONPATH="$CODE:$PYTHONPATH"
```

### Tokenization
- `TRAIN0.sh` is a sample script, which includes data downloader.
- You can use any tokenizers.
- The [sentencepiece](https://github.com/google/sentencepiece) is used in `TRAIN0.sh`.

### Binarization
The tokenized corpora are converted into binary data for the fairseq.
- `TRAIN1.sh` is a sample script.
  This script generates the shared vocabulary (i.e., joined dictionary).
```
fairseq-preprocess \
    --source-lang en --target-lang de \
    --joined-dictionary \
    --trainpref $CORPUS/train.bpe \
    --validpref $CORPUS/newstest2013.bpe \
    --testpref $CORPUS/newstest2014.bpe,$CORPUS/newstest2015.bpe \
    --destdir $DATA
```

### Training
- Training uses the plugin.
  The required arguments are `--user-dir $CODE`,
`--task dtoken_translation`, and
`--arch dtoken_transformer`(Transformer Base model).
- You can specify
`--arch dtoken_transformer_big`
instead of `--arch dtoken_transformer`.
- `TRAIN2.bash` is a sample script, in which the learning rate is set
to 0.0004, and the mini-batch size is set to around 500 sentences
(i.e., about 9,000 updates/epoch).

```
mkdir -p $MODEL
fairseq-train $DATA -s en -t de \
    --user-dir $CODE --task dtoken_translation \
    --arch dtoken_transformer \
    --no-progress-bar --log-format simple \
    --log-interval 1800 \
    --max-tokens 5000 --update-freq 4 \
    --max-epoch 50 \
    --optimizer adam --lr 0.0004 --adam-betas '(0.9, 0.99)' \
    --label-smoothing 0.1 --clip-norm 5 \
    --dropout 0.1 \
    --min-lr '1e-09' --lr-scheduler inverse_sqrt \
    --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy \
    --warmup-updates 45000 --warmup-init-lr '1e-07' \
    --save-dir $MODEL
```

### Evaluation
When you run `fairseq-generate` or `fairseq-interactive`,
you must give `--user-dir $CODE` and `--task dtoken_translation`.

```
fairseq-generate $DATA -s en -t de \
    --user-dir $CODE --task dtoken_translation \
    --no-progress-bar \
    --gen-subset valid \
    --path $MODEL/checkpoint_best.pt \
    --lenpen 1.0 \
    --beam 10 --batch-size 32
```

- `TEST.bash` is a sample script.

## Citation
```bibtex
@inproceedings{imamura-sumita-2020-transformer,
  title = "Transformer-based Double-token Bidirectional Autoregressive Decoding in Neural Machine Translation",
  author = "Imamura, Kenji and Sumita, Eiichiro",
  booktitle = "Proceedings of the 7th Workshop on Asian Translation",
  month = dec,
  year = "2020",
  address = "Suzhou, China",
  publisher = "Association for Computational Linguistics",
  url = "https://www.aclweb.org/anthology/2020.wat-1.3",
  pages = "50--57",
  abstract = "This paper presents a simple method that extends a standard Transformer-based autoregressive decoder, to speed up decoding. The proposed method generates a token from the head and tail of a sentence (two tokens in total) in each step. By simultaneously generating multiple tokens that rarely depend on each other, the decoding speed is increased while the degradation in translation quality is minimized. In our experiments, the proposed method increased the translation speed by around 113{\%}-155{\%} in comparison with a standard autoregressive decoder, while degrading the BLEU scores by no more than 1.03. It was faster than an iterative non-autoregressive decoder in many conditions.",
}
```
