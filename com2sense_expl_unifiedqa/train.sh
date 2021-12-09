#!/usr/bin/env bash

# ** T5-large || COSE **
nohup python3 main.py \
--mode train \
--expt_dir /local1/telinwu/yixiao/results/com2sense_unified \
--expt_name has_expl \
--run '' \
--model allenai/unifiedqa-t5-3b \
--train_file /local1/telinwu/yixiao/datasets/com2sense/train_expl.json \
--dev_file /local1/telinwu/yixiao/datasets/com2sense/dev_expl.json \
--batch_size 1  \
--has_explanation True \
--seq_len 128 \
--gpu 7 \
--use_amp F &

nohup python3 main.py \
--mode train \
--expt_dir /local1/telinwu/yixiao/results/com2sense_unified \
--expt_name no_expl \
--run '' \
--model allenai/unifiedqa-t5-3b \
--train_file /local1/telinwu/yixiao/datasets/com2sense/train_expl.json \
--dev_file /local1/telinwu/yixiao/datasets/com2sense/dev_expl.json \
--batch_size 1  \
--has_explanation False \
--seq_len 128 \
--gpu 6 \
--use_amp F &

nohup python3 main.py \
--mode train \
--expt_dir /local1/telinwu/yixiao/results/com2sense_unified \
--expt_name has_expl_t5 \
--run '' \
--model allenai/unifiedqa-t5-3b \
--train_file /local1/telinwu/yixiao/datasets/com2sense/train_expl_t5.json \
--dev_file /local1/telinwu/yixiao/datasets/com2sense/dev_expl_t5.json \
--batch_size 1  \
--has_explanation True \
--seq_len 128 \
--gpu 5 \
--use_amp F &

nohup python3 main.py \
--mode train \
--expt_dir /local1/telinwu/yixiao/results/com2sense_unified \
--expt_name has_expl_corpus \
--run '' \
--model allenai/unifiedqa-t5-3b \
--train_file /local1/telinwu/yixiao/datasets/com2sense/train_expl_corpus.json \
--dev_file /local1/telinwu/yixiao/datasets/com2sense/dev_expl_corpus.json \
--batch_size 1  \
--has_explanation True \
--seq_len 128 \
--gpu 4 \
--use_amp F &