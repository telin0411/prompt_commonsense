#!/usr/bin/env bash

# ** T5-large || COSE **
python3 main.py \
--mode train \
--expt_dir /local1/telinweu/yixiao/results/com2sense_unified \
--expt_name has_expl \
--run '' \
--model t5-large \
--train_file /local1/telinwu/yixiao/datasets/com2sense/train.json \
--dev_file /local1/telinwu/yixiao/datasets/com2sense/dev.json \
--batch_size 1  \
--seq_len 128 \
--gpu 7 \
--use_amp F
