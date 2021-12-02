#!/usr/bin/env bash

# ** T0pp || COSE **
python3 main.py \
--mode train \
--expt_dir /data1/yixiao/results/ \
--expt_name cose_gen_t0pp \
--run expFirst \
--model bigscience/T0pp \
--train_file /local1/telinwu/yixiao/datasets/cose_for_generation/train.json \
--dev_file /local1/telinwu/yixiao/datasets/cose_for_generation/dev.json \
--generate_mode explain_first \
--batch_size 1  \
--seq_len 128 \
--gpu "0,1,2,3" \
--use_amp F
