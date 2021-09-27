#!/usr/bin/env bash

# ** RoBerta-large || EntangledQA **
python3 main.py \
    --mode train \
    --expt_dir ./results_log/EntangledQA \
    --expt_name unifiedqa_11b \
    --model "allenai/unifiedqa-t5-11b" \
    --train_dataset semeval_2020 \
    --train_file all \
    --dev_dataset EntangledQA \
    --dev_file dev-a \
    --test_dataset EntangledQA \
    --test_file dev-b \
    --run semeval20_all \
    --seed 808 \
    --gpu 0,1,2,3,4,5,6,7 \
    --gpu 0,1,2,3 \
    --batch_size 4 \
    --seq_len 128 \
    --seq_len 75 \
    --acc_step 2 \
    --use_amp F \
    --lr 1e-5 \
    # --ckpt "" \
    # --train_dataset semeval_2020,com2sense \
    # --train_file all,train \
