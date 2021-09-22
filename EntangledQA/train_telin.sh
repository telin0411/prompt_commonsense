#!/usr/bin/env bash

# ** RoBerta-large || EntangledQA **
python3 main.py \
    --mode train \
    --expt_dir ./results_log/EntangledQA \
    --expt_name unifiedqa_11b \
    --model "allenai/unifiedqa-t5-11b" \
    --dataset EntangledQA \
    --run semeval20_pretrained \
    --run semeval20_com2sense20_pretrained \
    --seed 808 \
    --gpu 0,1,2,3 \
    --batch_size 4 \
    --seq_len 64 \
    --acc_step 4 \
    --use_amp F \
    --lr 5e-6 \
    --ckpt "../commonsense_pretraining/results_log/semeval_2020_task4/unifiedqa_11b_reasons/unifiedqa_11b_reasons/model_500.pth" \
    --ckpt "../Com2Sense/results_log/com2sense/unifiedqa_11b/semeval20_pretrained/ep_4_stp_2.1k_acc_78.2338_allenai_unifiedqa_t5_11b.pth" \
