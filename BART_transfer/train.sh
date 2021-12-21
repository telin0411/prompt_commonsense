#!/usr/bin/env bash

# ** BART || ECQA **
python3 main.py \
--mode train \
--expt_dir /data1/yixiao/results/ \
--expt_name bart_expl \
--run ecqa_16 \
--model facebook/bart-large \
--train_file /local1/telinwu/yixiao/datasets/ecqa_processed/train.json \
--dev_file /local1/telinwu/yixiao/datasets/ecqa_processed/dev.json \
--batch_size 32 \
--lr 1e-5 \
--epochs 50 \
--seq_len 128 \
--gpu "6" \
--use_amp F
