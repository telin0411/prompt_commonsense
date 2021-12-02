#!/usr/bin/env bash

# ** T5-large **
python3 main.py \
--mode test \
--ckpt /local1/telinwu/yixiao/results/pretrain_eqasc/t5_large/bs_16/model_81000.pth \
--model t5-large \
--train_file "" \
--dev_file "" \
--test_file /local1/telinwu/yixiao/datasets/corpus/corpus_eQASC.jsonl \
--seq_len 128 \
--batch_size 32 \
--gpu 0
