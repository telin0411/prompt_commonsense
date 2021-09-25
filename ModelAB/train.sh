#!/usr/bin/env bash
# Train Model B with projection layer
nohup python3 main.py \
--mode train \
--expt_dir ./results \
--expt_name model_b \
--run_name m1_s0 \
--model roberta-large \
--model_name model_b \
--num_prompt_model_layer -1 \
--num_task_model_layer -1 \
--is_projection True \
--seq_len 128 \
--data_dir ./datasets/com2sense \
--template "rather than" \
--mask_len 1 \
--lr 5e-5 \
--epochs 100 \
--batch_size 16 \
--gpu_ids 0 \
--seed 808 &
