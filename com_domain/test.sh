#!/usr/bin/env bash

#** T5-large || SemEval **
# python3 main.py \
#--mode test \
#--ckpt ./results_log/semeval/t5_large/bs_16/ep_7_stp_8.74k_acc_87.7016_t5_large.pth \
#--model t5-large \
#--dataset semeval \
#--batch_size 32 \
#--gpu 0

# ** RoBerta-large || Com2Sense **
python3 main.py \
--mode test \
--ckpt /nas/home/yixiaoli/results_log/com2sense/roberta_large/domain/bs_16/ep_42_stp_4.2k_acc_89.7500_roberta_large.pth \
--model roberta-large \
--dataset com2sense \
--batch_size 32 \
--gpu 1 \
--num_cls 3
