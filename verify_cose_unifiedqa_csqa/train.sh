#!/usr/bin/env bash

# ** T5-large || COSE **
python3 main.py \
--mode train \
--expt_dir /data1/yixiao/results/verify_cose-unifiedqa_csqa \
--expt_name t5_large \
--run bs_4_pre_first \
--model t5-large \
--train_file /data1/yixiao/datasets/cose_for_generation/train.json \
--dev_file /data1/yixiao/datasets/cose_for_generation/dev.json \
--generate_mode predict_first \
--batch_size 4  \
--seq_len 128 \
--gpu "0,1" \
--use_amp F
