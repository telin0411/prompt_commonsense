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
--ckpt "/nas/home/yixiaoli/results_log/com2sense/roberta_large/scenario/bs_16/ep_52_stp_5.2k_acc_93.8750_roberta_large.pth" "/nas/home/yixiaoli/results_log/com2sense/roberta_large/numeracy/bs_16/ep_97_stp_9.7k_acc_98.0000_roberta_large.pth" "/nas/home/yixiaoli/results_log/com2sense/roberta_large/bs_16/ep_39_stp_3.9k_acc_89.3750_roberta_large.pth" \
--model roberta-large \
--dataset com2sense \
--batch_size 32 \
--gpu 0 \
--num_cls 2
