#!/usr/bin/env bash

# ** RoBerta-large || Com2Sense **
python3 main.py --mode train \
--expt_dir ./results_log/com2sense \
--expt_name roberta_large \
--model roberta-large \
--dataset com2sense \
--run bs_4 \
--batch_size 4 \
--lr 1e-6 \
--gpu 3 --seq_len 128
