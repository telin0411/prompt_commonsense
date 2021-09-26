#!/usr/bin/env bash
# Train Model B with projection layer
nohup python3 main.py \
--mode train \
--expt_dir ./results \
--expt_name model_b \
--run_name m1_s0 \
--model roberta-large \
--model_name model_b \
--is_projection True \
--seq_len 128 \
--data_dir ./datasets/com2sense \
--template "rather than" \
--mask_len 1 \
--lr 1e-5 \
--epochs 150 \
--batch_size 8 \
--gpu_ids 0 \
--seed 808 &

nohup python3 main.py \
--mode train \
--expt_dir ./results \
--expt_name model_b \
--run_name m1_s1 \
--model roberta-large \
--model_name model_b \
--is_projection True \
--seq_len 128 \
--data_dir ./datasets/com2sense \
--template "rather than" \
--mask_len 1 \
--lr 1e-5 \
--epochs 150 \
--batch_size 8 \
--gpu_ids 1 \
--seed 818 &

nohup python3 main.py \
--mode train \
--expt_dir ./results \
--expt_name model_b \
--run_name m1_s2 \
--model roberta-large \
--model_name model_b \
--is_projection True \
--seq_len 128 \
--data_dir ./datasets/com2sense \
--template "rather than" \
--mask_len 1 \
--lr 1e-5 \
--epochs 150 \
--batch_size 8 \
--gpu_ids 2 \
--seed 828 &

nohup python3 main.py \
--mode train \
--expt_dir ./results \
--expt_name model_b \
--run_name m1_s3 \
--model roberta-large \
--model_name model_b \
--is_projection True \
--seq_len 128 \
--data_dir ./datasets/com2sense \
--template "rather than" \
--mask_len 1 \
--lr 1e-5 \
--epochs 150 \
--batch_size 8 \
--gpu_ids 3 \
--seed 838 &
