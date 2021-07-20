#!/usr/bin/env bash

# ** RoBerta-large || Com2Sense **
python3 main.py --mode train \
--expt_dir /nas/home/yixiaoli/results_log/com2sense/ \
--expt_name roberta_large \
--model roberta-large \
--dataset com2sense \
--run ./combine/bs_16 \
--batch_size 16 \
--gpu 0 \
--seq_len 128 \
--prompt "" \
--prompt_pos head \
--num_cls 12 \
--lr 3e-6 \
--epochs 200
