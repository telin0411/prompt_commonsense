#!/usr/bin/env bash

# ** RoBerta-large || Com2Sense **
python3 main.py --mode test \
    --expt_dir ./results_log/com2sense \
    --expt_name unifiedqa_11b \
    --model "allenai/unifiedqa-t5-11b" \
    --dataset com2sense \
    --run semeval20_pretrained \
    --seed 808 \
    --gpu 0,1,2,3 \
    --batch_size 1 \
    --seq_len 64 \
    --use_amp F \
    --lr 5e-6 \
    --ckpt "results_log/com2sense/unifiedqa_11b/semeval20_pretrained/ep_4_stp_2.1k_acc_78.2338_allenai_unifiedqa_t5_11b.pth" \
    # --train_file "train-40" \
    # --dev_file "dev-40" \
    # --test_file "test-40" \
