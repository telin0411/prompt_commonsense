#!/usr/bin/env bash

# ** RoBerta-large || Com2Sense **
python3 main.py --mode train \
    --expt_dir ./results_log/com2sense \
    --expt_name unifiedqa_11b \
    --model "allenai/unifiedqa-t5-11b" \
    --dataset com2sense \
    --run semeval20_pretrained \
    --run semeval20_pretrained_train40 \
    --run semeval20_pretrained_train80 \
    --run semeval20_pretrained_train60 \
    --seed 808 \
    --gpu 0,1,2,3 \
    --batch_size 4 \
    --seq_len 64 \
    --acc_step 2 \
    --use_amp F \
    --lr 5e-6 \
    --ckpt "../commonsense_pretraining/results_log/semeval_2020_task4/unifiedqa_11b_reasons/unifiedqa_11b_reasons/model_500.pth" \
    --train_file "train-60" \
    --dev_file "dev-60" \
    --test_file "test-60" \
