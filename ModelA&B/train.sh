#!/usr/bin/env bash
# Train Model B
nohup python3 main.py \
--mode train \
--expt_dir ~/results \
--expt_name model_a \
--run_name m4_s0 \
--model roberta-base \
--model_name model_a \
--num_prompt_model_layer -1 \
--num_task_model_layer -1 \
--seq_len 128 \
--data_dir ./datasets/com2sense \
--template "rather than" \
--mask_len 4 \
--lr 5e-6 \
--epochs 100 \
--batch_size 32 \
--gpu_ids 0 \
--seed 808 &

nohup python3 main.py \
--mode train \
--expt_dir ~/results \
--expt_name model_a \
--run_name m4_s1 \
--model roberta-base \
--model_name model_a \
--num_prompt_model_layer -1 \
--num_task_model_layer -1 \
--seq_len 128 \
--data_dir ./datasets/com2sense \
--template "rather than" \
--mask_len 4 \
--lr 5e-6 \
--epochs 100 \
--batch_size 32 \
--gpu_ids 1 \
--seed 818 &

nohup python3 main.py \
--mode train \
--expt_dir ~/results \
--expt_name model_a \
--run_name m4_s2 \
--model roberta-base \
--model_name model_a \
--num_prompt_model_layer -1 \
--num_task_model_layer -1 \
--seq_len 128 \
--data_dir ./datasets/com2sense \
--template "rather than" \
--mask_len 4 \
--lr 5e-6 \
--epochs 100 \
--batch_size 32 \
--gpu_ids 2 \
--seed 828 &

nohup python3 main.py \
--mode train \
--expt_dir ~/results \
--expt_name model_a \
--run_name m4_s3 \
--model roberta-base \
--model_name model_a \
--num_prompt_model_layer -1 \
--num_task_model_layer -1 \
--seq_len 128 \
--data_dir ./datasets/com2sense \
--template "rather than" \
--mask_len 4 \
--lr 5e-6 \
--epochs 100 \
--batch_size 32 \
--gpu_ids 3 \
--seed 838 &
