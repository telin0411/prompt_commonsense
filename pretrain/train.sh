#!/usr/bin/env bash

# ** T5-large || COSE **
python3 main.py \
--mode train \
--expt_dir /local1/telinweu/yixiao/results/pretrain_eqasc \
--expt_name t5_large \
--run bs_8 \
--model t5-large \
--train_file /local1/telinwu/yixiao/datasets/corpus/corpus_eQASC.jsonl \
--dev_file /local1/telinwu/yixiao/datasets/corpus/corpus_eQASC_dev.jsonl \
--batch_size 8  \
--seq_len 128 \
--gpu 7 \
--use_amp F
