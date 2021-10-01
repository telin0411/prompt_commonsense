#!/usr/bin/env bash

# ** RoBerta-large || EntangledQA **
python3 main.py \
    --mode released-b \
    --mode released-a \
    --expt_dir ./results_log/EntangledQA \
    --expt_name unifiedqa_11b \
    --model "allenai/unifiedqa-t5-11b" \
    --test_dataset EntangledQA \
    --run semeval20_all_pretrained_seqlen75_released-b \
    --run semeval20_all_pretrained_seqlen75_released-a \
    --seed 808 \
    --gpu 0,1,2,3,4,5,6,7 \
    --gpu 0,1,2,3 \
    --seq_len 128 \
    --seq_len 75 \
    --batch_size 1 \
    --acc_step 1 \
    --use_amp F \
    --lr 5e-6 \
    --ckpt "" \
