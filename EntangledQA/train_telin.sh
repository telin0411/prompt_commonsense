#!/usr/bin/env bash

# ** RoBerta-large || EntangledQA **
python3 main.py \
    --mode train \
    --expt_dir ./results_log/EntangledQA \
    --expt_name roberta_large_semeval20_pretrained \
    --model roberta-large \
    --dataset EntangledQA \
    --seq_len 128 \
    --batch_size 16 \
    --run 0 \
    --seed 808 \
    --ckpt "../commonsense_pretraining/model_ckpt/carl/carl_acc_87.29_v3.0.0.pth" \
    --gpu 0 \
