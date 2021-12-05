#!/usr/bin/env bash

# ** T0pp **
python3 main.py \
--mode test \
--model bigscience/T0pp \
--train_file "" \
--dev_file "" \
--test_file /local1/telinwu/yixiao/datasets/com2sense/train.json \
--pred_file "train_com2sense_pred.csv" \
--generate_mode explain_first \
--seq_len 128 \
--batch_size 1 \
--gpu "4,5,6,7"

# ** T0pp **
python3 main.py \
--mode test \
--model bigscience/T0pp \
--train_file "" \
--dev_file "" \
--test_file /local1/telinwu/yixiao/datasets/com2sense/dev.json \
--pred_file "dev_com2sense_pred.csv" \
--generate_mode explain_first \
--seq_len 128 \
--batch_size 1 \
--gpu "4,5,6,7"

# ** T0pp **
python3 main.py \
--mode test \
--model bigscience/T0pp \
--train_file "" \
--dev_file "" \
--test_file /local1/telinwu/yixiao/datasets/com2sense/test.json \
--pred_file "test_com2sense_pred.csv" \
--generate_mode explain_first \
--seq_len 128 \
--batch_size 1 \
--gpu "4,5,6,7"
