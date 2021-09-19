#!/usr/bin/env bash

# ** RoBerta-large || EntangledQA **
nohup python3 main.py \
--mode train \
--expt_dir ./results_log/EntangledQA \
--expt_name roberta_large \
--model roberta-large \
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
--dataset EntangledQA \
--seq_len 128 \
--batch_size 16 \
--run 3 \
--seed 838 \
--gpu 3 &