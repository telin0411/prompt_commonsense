#!/usr/bin/env bash

# ** RoBerta-large || EntangledQA **
python3 main.py \
--mode dev-b \
--ckpt ./results_log/EntangledQA/roberta_large/0/ep_25_stp_17.1k_acc_77.6042_roberta_large.pth \
--model roberta-large \
--dataset EntangledQA \
--batch_size 32 \
--run_name dev-b0 \
--gpu 0 &

python3 main.py \
--mode dev-b \
--ckpt ./results_log/EntangledQA/roberta_large/1/ep_90_stp_22.1k_acc_76.5625_roberta_large.pth \
--model roberta-large \
--dataset EntangledQA \
--batch_size 32 \
--run_name dev-b1 \
--gpu 1 &

python3 main.py \
--mode dev-b \
--ckpt ./results_log/EntangledQA/roberta_large/2/ep_46_stp_17.7k_acc_80.2083_roberta_large.pth \
--model roberta-large \
--dataset EntangledQA \
--batch_size 32 \
--run_name dev-b2 \
--gpu 2 &

python3 main.py \
--mode dev-b \
--ckpt ./results_log/EntangledQA/roberta_large/3/ep_22_stp_20.3k_acc_75.5208_roberta_large.pth \
--model roberta-large \
--dataset EntangledQA \
--batch_size 32 \
--run_name dev-b3 \
--gpu 3 &

python3 main.py \
--mode dev-a \
--ckpt ./results_log/EntangledQA/roberta_large/0/ep_25_stp_17.1k_acc_77.6042_roberta_large.pth \
--model roberta-large \
--dataset EntangledQA \
--batch_size 32 \
--run_name dev-a0 \
--gpu 0 &

python3 main.py \
--mode dev-a \
--ckpt ./results_log/EntangledQA/roberta_large/1/ep_90_stp_22.1k_acc_76.5625_roberta_large.pth \
--model roberta-large \
--dataset EntangledQA \
--batch_size 32 \
--run_name dev-a1 \
--gpu 1 &

python3 main.py \
--mode dev-a \
--ckpt ./results_log/EntangledQA/roberta_large/2/ep_46_stp_17.7k_acc_80.2083_roberta_large.pth \
--model roberta-large \
--dataset EntangledQA \
--batch_size 32 \
--run_name dev-a2 \
--gpu 2 &

python3 main.py \
--mode dev-a \
--ckpt ./results_log/EntangledQA/roberta_large/3/ep_22_stp_20.3k_acc_75.5208_roberta_large.pth \
--model roberta-large \
--dataset EntangledQA \
--batch_size 32 \
--run_name dev-a3 \
--gpu 3 &
