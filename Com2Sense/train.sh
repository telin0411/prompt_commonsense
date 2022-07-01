#!/usr/bin/env bash

# ** RoBerta-large || Com2Sense **
python3 main.py --mode train \
--expt_dir ./results_log/com2sense \
--expt_name roberta_large \
--model roberta-large \
--dataset com2sense \
--run warp_32_eq_1e2_scheduler_randn_wd0 \
--batch_size 16 \
--seed 808 \
--lr 1e-2 \
--acc_step 2 \
--gpu 1 --seq_len 128

