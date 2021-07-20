#!/usr/bin/env bash

# ** RoBerta-large || Com2Sense **
python3 main.py --mode train \
--expt_dir /nas/home/yixiaoli/results_log/com2sense/ \
--expt_name t5_large \
--model t5-large \
--dataset com2sense \
--run ./scenario/bs_16 \
--batch_size 16 \
--gpu "2,3" \
--seq_len 64 \
--prompt "" \
--prompt_pos head \
--num_cls 2 \
--lr 3e-6 \
--epochs 200
