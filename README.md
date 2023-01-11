# AcademicBART

We pretrained a BART-based Japanese masked language model on paper abstracts from the academic database CiNii Articles.  


## Download
They include a pretrained roberta model (best_model.pt), a sentencepiece model (sp.model) , a dictionary (dict.txt) and code for applying sentencepiece (apply-sp.py) .
```
wget http://aiweb.cs.ehime-u.ac.jp/~yamauchi/academic_model/Academic_BART_base.tar.gz
```
## Requirements
Python >= 3.8 <br>
[fairseq](https://github.com/facebookresearch/fairseq) == 0.12.2 (In working order)<br>
[sentencepiece](https://github.com/google/sentencepiece) <br>
tensorboardX (optional) <br>

## Preprocess
We applied SentencePiece for subword segmentation. <br>
Prepare datasets ($TRAIN_SRC, ...), which format assumes a tab delimiter between text and label.

```
python ./apply_sp.py $TRAIN_SRC $DATASET_DIR/train.src-tgt --bpe_model $SENTENCEPIECE_MODEL
python ./apply_sp.py $VALID_SRC $DATASET_DIR/valid.src-tgt --bpe_model $SENTENCEPIECE_MODEL
python ./apply_sp.py $TEST_SRC $DATASET_DIR/test.src-tgt --bpe_model $SENTENCEPIECE_MODEL
```
```
fairseq-preprocess \
    --source-lang "src" \
    --target-lang "tgt" \
    --trainpref "${DATASET_DIR}/train.src-tgt" \
    --validpref "${DATASET_DIR}/valid.src-tgt" \
    --testpref "${DATASET_DIR}/test.src-tgt" \
    --destdir "data-bin/" \
    --workers 60 \
    --srcdict ${DICT} \
    --tgtdict ${DICT}
```
## Finetune
The procedure for sentence classification using AcademicBART is as follows.
```

```
