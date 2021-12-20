#!/usr/bin/env bash

# ** BART || ECQA **
python3 main.py \
--mode train \
--expt_dir /data1/yixiao/results/ \
--expt_name bart_expl \
--run ecqa \
--model facebook/bart-large \
--train_file /local1/telinwu/yixiao/datasets/cose_processed/train.json \
--dev_file /local1/telinwu/yixiao/datasets/cose_processed/dev.json \
--test_file /local1/telinwu/yixiao/datasets/cose_processed/dev.json \
--test_file /local1/telinwu/yixiao/datasets/cose_processed/dev.json \
--pred_file ./pred.csv \
--append_test_file ./pred_append.json \
--batch_size 8  \
--seq_len 128 \
--gpu "0" \
--use_amp F
