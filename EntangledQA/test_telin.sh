#!/usr/bin/env bash

# ** RoBerta-large || EntangledQA **
python3 main.py \
    --mode released-b \
    --expt_dir ./results_log/EntangledQA \
    --expt_name unifiedqa_11b \
    --model "allenai/unifiedqa-t5-11b" \
    --dataset EntangledQA \
    --run semeval20_com2sense60_pretrained_released-a \
    --run semeval20_com2sense80_pretrained_released-a \
    --run semeval20_com2sense40_pretrained_released-a \
    --run semeval20_pretrained_released-b \
    --seed 808 \
    --gpu 0,1,2,3,4,5,6,7 \
    --gpu 0,1,2,3 \
    --batch_size 1 \
    --seq_len 64 \
    --seq_len 128 \
    --acc_step 1 \
    --use_amp F \
    --lr 5e-6 \
    --ckpt "./results_log/EntangledQA/unifiedqa_11b/semeval20_pretrained/ep_1_stp_518_acc_90.5000_allenai_unifiedqa_t5_11b.pth" \
    --ckpt "./results_log/EntangledQA/unifiedqa_11b/semeval20_com2sense20_pretrained/ep_4_stp_2.1k_acc_92.0000_allenai_unifiedqa_t5_11b.pth" \
    --ckpt "./results_log/EntangledQA/unifiedqa_11b/semeval20_com2sense60_pretrained/ep_3_stp_4.1k_acc_92.0000_allenai_unifiedqa_t5_11b.pth" \
    --ckpt "./results_log/EntangledQA/unifiedqa_11b/semeval20_com2sense80_pretrained/ep_4_stp_6.9k_acc_91.5000_allenai_unifiedqa_t5_11b.pth" \
    --ckpt "./results_log/EntangledQA/unifiedqa_11b/semeval20_com2sense40_pretrained/ep_3_stp_2.1k_acc_92.0000_allenai_unifiedqa_t5_11b.pth" \
    --ckpt "./results_log/EntangledQA/unifiedqa_11b/semeval20_pretrained/ep_6_stp_4.6k_acc_92.5000_allenai_unifiedqa_t5_11b.pth" \
    --ckpt "./results_log/EntangledQA/unifiedqa_11b/semeval20_pretrained/ep_5_stp_4.6k_acc_91.5000_allenai_unifiedqa_t5_11b.pth" \
    --ckpt "./results_log/EntangledQA/unifiedqa_11b/semeval20_pretrained/ep_1_stp_518_acc_90.5000_allenai_unifiedqa_t5_11b.pth" \
