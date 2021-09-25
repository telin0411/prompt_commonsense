#!/usr/bin/env bash
python3 main.py \
--mode test \
--model roberta-large \
--model_name model_b \
--seq_len 128 \
--template "rather than" \
--mask_len 1 \
--is_projection True \
--batch_size 8 \
--data_dir ./datasets/com2sense \
--gpu_ids 0 \
--ckpt ./results/model_b/m1_s1/ep_57_stp_1.3k_acc_52.5000_roberta_large.pth \
--seed 808 &

python3 main.py \
--mode test \
--model roberta-large \
--model_name model_b \
--seq_len 128 \
--template "rather than" \
--mask_len 1 \
--is_projection True \
--batch_size 8 \
--data_dir ./datasets/com2sense \
--gpu_ids 1 \
--ckpt ./results/model_b/m1_s1_1/ep_99_stp_2.3k_acc_55.0000_roberta_large.pth \
--seed 808 &
