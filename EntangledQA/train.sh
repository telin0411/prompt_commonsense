#!/usr/bin/env bash

# ** RoBerta-large || EntangledQA **
nohup python3 main.py \
--mode train \
--expt_dir ./results_log/EntangledQA \
--expt_name roberta_large \
--model roberta-large \
--ckpt ~/yx/prompt_commonsense/Com2Sense/results_log/com2sense/roberta_large/baseline_0/ep_14_stp_17.0k_acc_53.8557_roberta_large.pth \
--dataset EntangledQA \
--seq_len 128 \
--batch_size 16 \
--run 0 \
--seed 808 \
--gpu 0 &

nohup python3 main.py \
--mode train \
--expt_dir ./results_log/EntangledQA \
--expt_name roberta_large \
--model roberta-large \
--ckpt ~/yx/prompt_commonsense/Com2Sense/results_log/com2sense/roberta_large/baseline_1/ep_25_stp_21.5k_acc_61.0697_roberta_large.pth \
--dataset EntangledQA \
--seq_len 128 \
--batch_size 16 \
--run 1 \
--seed 818 \
--gpu 1 &

nohup python3 main.py \
--mode train \
--expt_dir ./results_log/EntangledQA \
--expt_name roberta_large \
--model roberta-large \
--ckpt ~/yx/prompt_commonsense/Com2Sense/results_log/com2sense/roberta_large/baseline_2/ep_15_stp_17.4k_acc_58.2090_roberta_large.pth \
--dataset EntangledQA \
--seq_len 128 \
--batch_size 16 \
--run 2 \
--seed 828 \
--gpu 2 &

nohup python3 main.py \
--mode train \
--expt_dir ./results_log/EntangledQA \
--expt_name roberta_large \
--model roberta-large \
--ckpt ~/yx/prompt_commonsense/Com2Sense/results_log/com2sense/roberta_large/baseline_3/ep_22_stp_20.3k_acc_63.4328_roberta_large.pth \
--dataset EntangledQA \
--seq_len 128 \
--batch_size 16 \
--run 3 \
--seed 838 \
--gpu 3 &
