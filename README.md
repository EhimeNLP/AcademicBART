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
The procedure for summary using AcademicBART is as follows.
```
fairseq-train data-bin/ \
    --restore-file $BART_PATH \
    --max-tokens 512 --max-sentences $MAX_SENTENCES \
    --task translation \
    --source-lang src --target-lang tgt \
    --truncate-source \
    --layernorm-embedding \
    --share-all-embeddings \
    --share-decoder-input-output-embed \
    --reset-optimizer --reset-dataloader --reset-meters \
    --required-batch-size-multiple 1 \
    --arch bart_base \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 \
    --clip-norm 0.1 \
    --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
    --fp16 --update-freq $UPDATE_FREQ \
    --skip-invalid-size-inputs-valid-test \
    --no-epoch-checkpoints \
    --save-interval-updates $SAVE_INTERVAL --save-dir result_test \
    --patience 5 \
```
## Reference
山内洋輝, 梶原智之, 桂井麻里衣, 大向一輝, 二宮崇. [学術ドメインに特化した日本語事前訓練モデルの構築](https://www.anlp.jp/proceedings/annual_meeting/2023/pdf_dir/Q11-4.pdf)  . 言語処理学会第29回年次大会, pp.2842-2846, 2023.
