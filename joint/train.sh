#!/usr/bin/env bash

# ** Joint train || COSE **
python3 main.py \
--mode train \
--expt_dir /local1/telinwu/yixiao/results/joint_cose \
--expt_name unifiedqa \
--run bs_1 \
--model allenai/unifiedqa-t5-3b \
--train_file /data1/yixiao/datasets/cose_for_generation/train.json \
--dev_file /data1/yixiao/datasets/cose_for_generation/dev.json \
--batch_size 1  \
--seq_len 128 \
--gpu "0,1,3,4" \
--use_amp F
