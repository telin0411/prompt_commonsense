#!/usr/bin/env bash

# ** BART || ECQA **
python3 main.py \
--mode test \
--expt_dir /data1/yixiao/results/ \
--expt_name bart_expl \
--run ecqa \
--model facebook/bart-large \
--ckpt /data1/yixiao/results/bart_expl/ecqa_16/ep_22_stp_46.8k_acc_9.230007_facebook_bart_large.pth \
--train_file /local1/telinwu/yixiao/datasets/ecqa_processed/train.json \
--dev_file /local1/telinwu/yixiao/datasets/ecqa_processed/dev.json \
--test_file /local1/telinwu/yixiao/datasets/ecqa_processed/dev.json \
--test_file /local1/telinwu/yixiao/datasets/ecqa_processed/dev.json \
--pred_file ./pred.csv \
--append_test_file ./pred_append.json \
--batch_size 32  \
--seq_len 128 \
--gpu 7 \
--use_amp F
