#!/usr/bin/env bash

# ** RoBerta-large || EntangledQA **
python3 main.py \
--mode released-b \
--ckpt ./results_log/EntangledQA/roberta_large/0/ep_97_stp_389_acc_70.8333_roberta_large.pth \
--model roberta-large \
--dataset EntangledQA \
--batch_size 32 \
--run_name released-b0 \
--gpu 0 &

python3 main.py \
--mode released-b \
--ckpt ./results_log/EntangledQA/roberta_large/1/ep_57_stp_229_acc_65.6250_roberta_large.pth \
--model roberta-large \
--dataset EntangledQA \
--batch_size 32 \
--run_name released-b1 \
--gpu 1 &

python3 main.py \
--mode released-b \
--ckpt ./results_log/EntangledQA/roberta_large/2/ep_73_stp_293_acc_62.5000_roberta_large.pth \
--model roberta-large \
--dataset EntangledQA \
--batch_size 32 \
--run_name released-b2 \
--gpu 2 &

python3 main.py \
--mode released-b \
--ckpt ./results_log/EntangledQA/roberta_large/3/ep_12_stp_49_acc_61.9792_roberta_large.pth \
--model roberta-large \
--dataset EntangledQA \
--batch_size 32 \
--run_name released-b3 \
--gpu 3 &
