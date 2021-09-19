#!/usr/bin/env bash

# ** Accuracy || EntangledQA **
python3 evaluate_entangled.py \
--cycic3a_preds ./pred/released-a0pred.lst \
--cycic3b_preds ./pred/released-b0pred.lst \
--cycic3a_labels ./datasets/cycic3/cycic3a_released_labels.jsonl \
--cycic3b_labels ./datasets/cycic3/cycic3b_released_labels.jsonl \
--map ./datasets/cycic3/cycic3_released_map.csv

python3 evaluate_entangled.py \
--cycic3a_preds ./pred/released-a1pred.lst \
--cycic3b_preds ./pred/released-b1pred.lst \
--cycic3a_labels ./datasets/cycic3/cycic3a_released_labels.jsonl \
--cycic3b_labels ./datasets/cycic3/cycic3b_released_labels.jsonl \
--map ./datasets/cycic3/cycic3_released_map.csv

python3 evaluate_entangled.py \
--cycic3a_preds ./pred/released-a2pred.lst \
--cycic3b_preds ./pred/released-b2pred.lst \
--cycic3a_labels ./datasets/cycic3/cycic3a_released_labels.jsonl \
--cycic3b_labels ./datasets/cycic3/cycic3b_released_labels.jsonl \
--map ./datasets/cycic3/cycic3_released_map.csv

python3 evaluate_entangled.py \
--cycic3a_preds ./pred/released-a3pred.lst \
--cycic3b_preds ./pred/released-b3pred.lst \
--cycic3a_labels ./datasets/cycic3/cycic3a_released_labels.jsonl \
--cycic3b_labels ./datasets/cycic3/cycic3b_released_labels.jsonl \
--map ./datasets/cycic3/cycic3_released_map.csv
