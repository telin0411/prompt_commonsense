#!/usr/bin/env bash

# ** RoBerta-large || Com2Sense **
python3 main.py --mode train \
--expt_dir ./results_log/com2sense \
--expt_name roberta_large \
--model roberta-large \
--dataset com2sense \
--run bs_16 \
--batch 16 \
--gpu 0 --seq_len 128


# ** RoBerta-large || WinoGrande **
python3 main.py \
--mode train \
--expt_dir ./results_log/winogrande \
--expt_name roberta_large \
--model roberta-large \
--dataset winogd \
--run lr_1e5_bs_16 \
--batch 16 \
--gpu 0


# ** T5-large || SemEval **
python3 main.py \
--mode train \
--expt_dir ./results_log/semeval \
--expt_name t5_large \
--model t5-large \
--dataset semeval \
--run bs_4 \
--batch 4  \
--seq_len 128 \
--gpu 0 --use_amp F


# ** T5-large || Social-IQA **
python3 main.py \
--mode train \
--expt_dir ./results_log/social_iqa \
--expt_name t5_large \
--model t5-large \
--dataset siqa \
--run bs_4 \
--batch 4  \
--seq_len 256 \
--gpu 0 --use_amp F


# ** T5-large || Physical-IQA **
python3 main.py \
--mode train \
--expt_dir ./results_log/physical_iqa \
--expt_name t5_large \
--model t5-large \
--dataset piqa \
--run bs_4_acc_2 \
--batch 4  \
--acc_step 2 \
--seq_len 512 \
--gpu 0 --use_amp F


# ** T5-large || WinoGrande + CommonsenseQA **
python3 main.py \
--mode train \
--expt_dir ./results_log/_multi_data/wgd_cqa \
--expt_name t5_large \
--model t5-large \
--dataset winogd,cqa  \
--run bs_8 \
--batch 8 \
--seq 128 \
--gpu 0 --use_amp F
