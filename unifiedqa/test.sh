#!/usr/bin/env bash

python3 main.py \
--mode test \
--ckpt ./model_ckpt/ep_13_16k_acc_87.2984_roberta_large.pth \
--model roberta-large \
--dataset com2sense \
--batch_size 32 \
--gpu 0
