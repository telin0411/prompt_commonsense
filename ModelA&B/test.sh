#!/usr/bin/env bash
python3 main.py \
--mode test \
--model roberta-base \
--model_name model_a \
--seq_len 128 \
--template "rather than" \
--mask_len 4 \
--batch_size 32 \
--data_dir ./datasets/com2sense \
--gpu_ids 0 \
--ckpt ~/results/model_a/m4_s0/ep_58_stp_639_acc_55.6250_roberta_base.pth \
--seed 808 &

python3 main.py \
--mode test \
--model roberta-base \
--model_name model_a \
--seq_len 128 \
--template "rather than" \
--mask_len 4 \
--batch_size 32 \
--data_dir ./datasets/com2sense \
--gpu_ids 1 \
--ckpt ~/results/model_a/m4_s1/ep_43_stp_474_acc_57.5000_roberta_base.pth \
--seed 808 &

python3 main.py \
--mode test \
--model roberta-base \
--model_name model_a \
--seq_len 128 \
--template "rather than" \
--mask_len 4 \
--batch_size 32 \
--data_dir ./datasets/com2sense \
--gpu_ids 2 \
--ckpt ~/results/model_a/m4_s2/ep_69_stp_760_acc_56.2500_roberta_base.pth \
--seed 808 &

python3 main.py \
--mode test \
--model roberta-base \
--model_name model_a \
--seq_len 128 \
--template "rather than" \
--mask_len 4 \
--batch_size 32 \
--data_dir ./datasets/com2sense \
--gpu_ids 3 \
--ckpt ~/results/model_a/m4_s3/ep_39_stp_430_acc_55.6250_roberta_base.pth \
--seed 808 &
