#!/usr/bin/env bash

# ** RoBerta-large || EntangledQA **
python3 interactive_test.py \
    --mode test \
    --model "roberta-large" \
    --model "allenai/unifiedqa-t5-11b" \
    --seed 42 \
    --gpu 0,1,2,3,4,5,6,7 \
    --gpu 0,1,2,3 \
    --gpu 0,1,2,3,4,5 \
    --gpu 0 \
    --gpu 0,1 \
    --seq_len 128 \
    --ckpt "../Com2Sense/results_log/com2sense/unifiedqa_11b/semeval20_pretrained_train80/ep_4_stp_6.9k_acc_82.7889_allenai_unifiedqa_t5_11b.pth" \
    --ckpt "./results_log/SemEval20/unifiedqa_11b/semeval20_all/ep_1_stp_6.0k_acc_95.1608_allenai_unifiedqa_t5_11b.pth" \
    # --ckpt "./results_log/SemEval20/unifiedqa_11b/semeval20_all/ep_2_stp_12.0k_acc_98.3704_allenai_unifiedqa_t5_11b.pth" \
    # --ckpt "./results_log/SemEval20/unifiedqa_11b/semeval20_all/ep_1_stp_6.0k_acc_95.1608_allenai_unifiedqa_t5_11b.pth" \
    # --ckpt "" \
    # --train_dataset semeval_2020,com2sense \
    # --train_file all,train \
