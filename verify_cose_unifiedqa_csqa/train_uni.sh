#!/usr/bin/env bash
rm -rf /local1/telinwu/yixiao/results/verify_cose-unifiedqa_csqa_1130
# ** T5-large || COSE **
nohup python3 main.py \
--mode train \
--expt_dir /local1/telinwu/yixiao/results/verify_cose-unifiedqa_csqa_1130 \
--expt_name unified_3b \
--run bs1_preFirst_noExpl \
--model allenai/unifiedqa-t5-3b \
--train_file /local1/telinwu/yixiao/datasets/cose_generated_t5/train_preFirst.json \
--dev_file /local1/telinwu/yixiao/datasets/cose_generated_t5/dev_preFirst.json \
--batch_size 1  \
--lr 5e-6 \
--has_explanation False \
--seq_len 64 \
--log_interval 4000 \
--gpu 1 \
--use_amp F &

nohup python3 main.py \
--mode train \
--expt_dir /local1/telinwu/yixiao/results/verify_cose-unifiedqa_csqa_1130 \
--expt_name unified_3b \
--run bs1_preFirst_hasExpl \
--model allenai/unifiedqa-t5-3b \
--train_file /local1/telinwu/yixiao/datasets/cose_generated_t5/train_preFirst.json \
--dev_file /local1/telinwu/yixiao/datasets/cose_generated_t5/dev_preFirst.json \
--batch_size 1  \
--lr 5e-6 \
--has_explanation True \
--seq_len 64 \
--log_interval 4000 \
--gpu 2 \
--use_amp F &

nohup python3 main.py \
--mode train \
--expt_dir /local1/telinwu/yixiao/results/verify_cose-unifiedqa_csqa_1130 \
--expt_name unified_3b \
--run bs1_expFirst_noExpl \
--model allenai/unifiedqa-t5-3b \
--train_file /local1/telinwu/yixiao/datasets/cose_generated_t5/train_expFirst.json \
--dev_file /local1/telinwu/yixiao/datasets/cose_generated_t5/dev_expFirst.json \
--batch_size 1  \
--lr 5e-6 \
--has_explanation False \
--seq_len 64 \
--log_interval 4000 \
--gpu 3 \
--use_amp F &

nohup python3 main.py \
--mode train \
--expt_dir /local1/telinwu/yixiao/results/verify_cose-unifiedqa_csqa_1130 \
--expt_name unified_3b \
--run bs1_expFirst_hasExpl \
--model allenai/unifiedqa-t5-3b \
--train_file /local1/telinwu/yixiao/datasets/cose_generated_t5/train_expFirst.json \
--dev_file /local1/telinwu/yixiao/datasets/cose_generated_t5/dev_expFirst.json \
--batch_size 1  \
--lr 5e-6 \
--has_explanation True \
--seq_len 64 \
--log_interval 4000 \
--gpu 4 \
--use_amp F &
