#!/usr/bin/env bash

# ** BART || ECQA **
python3 main_gan.py \
--mode train \
--expt_dir /local1/telinwu/yixiao/results/ \
--expt_name bart_gan \
--run test \
--model facebook/bart-large \
--weights_path /local1/telinwu/yixiao/results/bart_expl/ecqa_slight/ep_20_stp_10.6k_acc_4.884135_facebook_bart_large.pth \
--real_train_file /local1/telinwu/yixiao/datasets/ecqa_processed/train.json \
--fake_train_file /local1/telinwu/yixiao/datasets/com2sense/train.json \
--real_valid_file /local1/telinwu/yixiao/datasets/ecqa_processed/dev.json \
--fake_valid_file /local1/telinwu/yixiao/datasets/com2sense/dev.json \
--batch_size 16 \
--lr 1e-6 \
--epochs 10 \
--seq_len 128 \
--gpu 7 \
--use_amp F
