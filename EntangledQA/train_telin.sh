#!/usr/bin/env bash

# ** RoBerta-large || EntangledQA **
python3 main.py \
    --mode train \
    --expt_dir ./results_log/EntangledQA \
    --expt_name unifiedqa_11b \
    --model "allenai/unifiedqa-t5-3b" \
    --dataset EntangledQA \
    --run semeval20_pretrained \
    --seed 808 \
    --gpu 0,1,2,3 \
    --batch_size 2 \
    --seq_len 128 \
    --acc_step 4 \
    --use_amp F \
    --lr 1e-5 \
    # --ckpt "../commonsense_pretraining/results_log/semeval_2020_task4/unifiedqa_11b_reasons/unifiedqa_11b_reasons/model_500.pth" \
