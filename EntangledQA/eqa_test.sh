#!/usr/bin/env bash

# ** Accuracy || EntangledQA **
python3 evaluate_entangled.py \
--cycic3a_preds ./pred/dev-a0pred.lst \
--cycic3b_preds ./pred/dev-b0pred.lst \
--cycic3a_labels ./datasets/cycic3/dev_a_labels.jsonl \
--cycic3b_labels ./datasets/cycic3/dev_b_labels.jsonl \
--map ./datasets/cycic3/cycic3_dev_question_map.csv

python3 evaluate_entangled.py \
--cycic3a_preds ./pred/dev-a1pred.lst \
--cycic3b_preds ./pred/dev-b1pred.lst \
--cycic3a_labels ./datasets/cycic3/dev_a_labels.jsonl \
--cycic3b_labels ./datasets/cycic3/dev_b_labels.jsonl \
--map ./datasets/cycic3/cycic3_dev_question_map.csv

python3 evaluate_entangled.py \
--cycic3a_preds ./pred/dev-a2pred.lst \
--cycic3b_preds ./pred/dev-b2pred.lst \
--cycic3a_labels ./datasets/cycic3/dev_a_labels.jsonl \
--cycic3b_labels ./datasets/cycic3/dev_b_labels.jsonl \
--map ./datasets/cycic3/cycic3_dev_question_map.csv

python3 evaluate_entangled.py \
--cycic3a_preds ./pred/dev-a3pred.lst \
--cycic3b_preds ./pred/dev-b3pred.lst \
--cycic3a_labels ./datasets/cycic3/dev_a_labels.jsonl \
--cycic3b_labels ./datasets/cycic3/dev_b_labels.jsonl \
--map ./datasets/cycic3/cycic3_dev_question_map.csv
