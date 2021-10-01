#!/usr/bin/env bash

# ** RoBerta-large || EntangledQA **
python3 main.py \
    --mode train \
    --expt_dir ./results_log/SemEval20 \
    --expt_dir ./results_log/EntangledQA \
    --expt_name unifiedqa_11b \
    --model "allenai/unifiedqa-t5-11b" \
    --train_dataset semeval_2020 \
    --train_dataset EntangledQA \
    --train_file all \
    --train_file train \
    --dev_dataset semeval_2020 \
    --dev_dataset EntangledQA \
    --dev_file dev \
    --dev_file released-a \
    --test_dataset semeval_2020 \
    --test_dataset EntangledQA \
    --test_file test \
    --test_file released-b \
    --run semeval20_all \
    --run semeval20_all_pretrained \
    --seed 42 \
    --gpu 0,1,2,3,4,5,6,7 \
    --gpu 0,1,2,3 \
    --gpu 0,1,2,3,4,5 \
    --batch_size 2 \
    --seq_len 128 \
    --acc_step 16 \
    --use_amp F \
    --lr 1e-5 \
    --ckpt "./results_log/SemEval20/unifiedqa_11b/semeval20_all/ep_2_stp_12.0k_acc_98.3704_allenai_unifiedqa_t5_11b.pth" \
    --ckpt "./results_log/SemEval20/unifiedqa_11b/semeval20_all/ep_1_stp_6.0k_acc_95.1608_allenai_unifiedqa_t5_11b.pth" \
    # --ckpt "" \
    # --train_dataset semeval_2020,com2sense \
    # --train_file all,train \
