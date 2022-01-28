#!/usr/bin/env bash

# ** Stage1 || ECQA **
python3 main.py \
--mode train \
--expt_dir /local1/telinwu/yixiao/results/RL \
--expt_name stage1 \
--run bs_16 \
--model allenai/unifiedqa-t5-3b \
--train_pos_file /local1/telinwu/yixiao/datasets/3stage/ecqa/train_pos_60_67.json \
--train_neg_file /local1/telinwu/yixiao/datasets/3stage/ecqa/train_neg_60_67.json \
--dev_pos_file /local1/telinwu/yixiao/datasets/3stage/ecqa/dev_neg_10.json \
--dev_neg_file /local1/telinwu/yixiao/datasets/3stage/ecqa/dev_pos_10.json \
--batch_size 16  \
--seq_len 128 \
--gpu "0,1" \
--use_amp F
