#!/usr/bin/env bash

# ** T0pp **
python3 main.py \
--mode test \
--model bigscience/T0pp \
--train_file "" \
--dev_file "" \
--test_file /local1/telinwu/yixiao/datasets/cose_for_generation/train.json \
--pred_file "train_pred.csv" \
--generate_mode explain_first \
--seq_len 128 \
--batch_size 1 \
--gpu "5,6,7,8"

# ** T0pp **
python3 main.py \
--mode test \
--model bigscience/T0pp \
--train_file "" \
--dev_file "" \
--test_file /local1/telinwu/yixiao/datasets/cose_for_generation/dev.json \
--pred_file "dev_pred.csv" \
--generate_mode explain_first \
--seq_len 128 \
--batch_size 1 \
--gpu "5,6,7,8"
