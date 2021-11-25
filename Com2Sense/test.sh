#!/usr/bin/env bash

# ** T5-large || SemEval **
#python3 main.py \
#--mode test \
#--ckpt ./results_log/semeval/t5_large/bs_16/ep_7_stp_8.74k_acc_87.7016_t5_large.pth \
#--model t5-large \
#--dataset semeval \
#--batch_size 32 \
#--gpu 0

# ** RoBerta-large || Com2Sense **
python3 main.py \
--mode test \
--ckpt ./results_log/com2sense/roberta_large/bs_16_seed_808/ep_43_stp_4.3k_acc_53.8750_roberta_large.pth \
--model roberta-large \
--dataset com2sense \
--batch_size 32 \
--gpu 2
