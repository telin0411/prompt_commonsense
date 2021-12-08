#!/usr/bin/env bash

# ** T5-large || COSE **
nohup python3 main.py \
--mode test \
--expt_dir /local1/telinwu/yixiao/results/com2sense_unified \
--expt_name has_expl_t0 \
--run '' \
--model allenai/unifiedqa-t5-3b \
--pred_file ./pred_has_expl_t0.csv \
--ckpt /local1/telinwu/yixiao/results/com2sense_unified/has_expl_t0/ep_56_stp_90.0k_acc_0.681592_allenai_unifiedqa_t5_3b.pth \
--train_file /local1/telinwu/yixiao/datasets/com2sense/train_expl.json \
--dev_file /local1/telinwu/yixiao/datasets/com2sense/dev_expl.json \
--batch_size 1  \
--has_explanation True \
--seq_len 128 \
--gpu 7 \
--use_amp F &

nohup python3 main.py \
--mode test \
--expt_dir /local1/telinwu/yixiao/results/com2sense_unified \
--expt_name no_expl \
--run '' \
--model allenai/unifiedqa-t5-3b \
--pred_file ./pred_no_expl.csv \
--ckpt /local1/telinwu/yixiao/results/com2sense_unified/no_expl/ep_28_stp_45.0k_acc_0.732587_allenai_unifiedqa_t5_3b.pth \
--train_file /local1/telinwu/yixiao/datasets/com2sense/train_expl.json \
--dev_file /local1/telinwu/yixiao/datasets/com2sense/dev_expl.json \
--batch_size 1  \
--has_explanation False \
--seq_len 128 \
--gpu 6 \
--use_amp F &

nohup python3 main.py \
--mode test \
--expt_dir /local1/telinwu/yixiao/results/com2sense_unified \
--expt_name has_expl_t5 \
--run '' \
--model allenai/unifiedqa-t5-3b \
--pred_file ./pred_has_expl_t5.csv \
--ckpt /local1/telinwu/yixiao/results/com2sense_unified/has_expl_t0/ep_51_stp_82.0k_acc_0.639303_allenai_unifiedqa_t5_3b.pth \
--train_file /local1/telinwu/yixiao/datasets/com2sense/train_expl_t5.json \
--dev_file /local1/telinwu/yixiao/datasets/com2sense/dev_expl_t5.json \
--batch_size 1  \
--has_explanation True \
--seq_len 128 \
--gpu 5 \
--use_amp F &

nohup python3 main.py \
--mode test \
--expt_dir /local1/telinwu/yixiao/results/com2sense_unified \
--expt_name has_expl_corpus \
--run '' \
--model allenai/unifiedqa-t5-3b \
--pred_file ./pred_has_expl_corpus.csv \
--ckpt /local1/telinwu/yixiao/results/com2sense_unified/has_expl_corpus/ep_15_stp_24.1k_acc_0.699005_allenai_unifiedqa_t5_3b.pth \
--train_file /local1/telinwu/yixiao/datasets/com2sense/train_expl_corpus.json \
--dev_file /local1/telinwu/yixiao/datasets/com2sense/dev_expl_corpus.json \
--batch_size 1  \
--has_explanation True \
--seq_len 128 \
--gpu 4 \
--use_amp F &

