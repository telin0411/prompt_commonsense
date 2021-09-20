#!/usr/bin/env bash

# ** RoBerta-large || Com2Sense **
nohup python3 main.py --mode train \
--expt_dir ./results_log/com2sense \
--expt_name roberta_large \
--model roberta-large \
--dataset com2sense \
--ckpt ./carl_acc_87.29_v3.0.0.pth \
--run baseline_0 \
--seed 808 \
--gpu 0 \
--batch_size 16 \
--seq_len 128 &

nohup python3 main.py --mode train \
--expt_dir ./results_log/com2sense \
--expt_name roberta_large \
--model roberta-large \
--dataset com2sense \
--ckpt ./carl_acc_87.29_v3.0.0.pth \
--run baseline_1 \
--seed 818 \
--gpu 1 \
--batch_size 16 \
--seq_len 128 &

nohup python3 main.py --mode train \
--expt_dir ./results_log/com2sense \
--expt_name roberta_large \
--model roberta-large \
--dataset com2sense \
--ckpt ./carl_acc_87.29_v3.0.0.pth \
--run baseline_2 \
--seed 828 \
--gpu 2 \
--batch_size 16 \
--seq_len 128 &

nohup python3 main.py --mode train \
--expt_dir ./results_log/com2sense \
--expt_name roberta_large \
--model roberta-large \
--dataset com2sense \
--ckpt ./carl_acc_87.29_v3.0.0.pth \
--run baseline_3 \
--seed 838 \
--gpu 3 \
--batch_size 16 \
--seq_len 128 &