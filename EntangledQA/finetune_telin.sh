#!/usr/bin/env bash

# ** RoBerta-large || EntangledQA **
python3 main.py \
    --mode train \
    --expt_dir ./results_log/EntangledQA \
    --expt_name unifiedqa_11b \
    --model "allenai/unifiedqa-t5-11b" \
    --train_dataset EntangledQA \
    --train_file train \
    --dev_dataset EntangledQA \
    --dev_file dev-a \
    --test_dataset EntangledQA \
    --test_file dev-b \
    --run semeval20_com2sense20_pretrained \
    --run semeval20_pretrained_seqlen75 \
    --run semeval20_pretrained_seqlen128 \
    --run semeval20_com2sense100_pretrained \
    --run no_pretrained \
    --run semeval20_all_pretrained \
    --seed 808 \
    --gpu 0,1,2,3,4,5,6,7 \
    --gpu 0,1,2,3 \
    --batch_size 4 \
    --seq_len 128 \
    --seq_len 75 \
    --acc_step 2 \
    --use_amp F \
    --lr 1e-5 \
    --ckpt "./results_log/EntangledQA/unifiedqa_11b/semeval20_all/ep_1_stp_6.0k_acc_89.5000_allenai_unifiedqa_t5_11b.pth" \
    # --ckpt "../Com2Sense/results_log/com2sense/unifiedqa_11b/semeval20_pretrained_train100/ep_3_stp_9.0k_acc_93.9698_allenai_unifiedqa_t5_11b.pth" \
    # --ckpt "../commonsense_pretraining/results_log/semeval_2020_task4/unifiedqa_11b_reasons/unifiedqa_11b_reasons/model_500.pth" \
    # --ckpt "../Com2Sense/results_log/com2sense/unifiedqa_11b/semeval20_pretrained/ep_4_stp_2.1k_acc_78.2338_allenai_unifiedqa_t5_11b.pth" \
    # --ckpt "../commonsense_pretraining/results_log/semeval_2020_task4/unifiedqa_11b_reasons/bz4_acc2_seqlen75/model_3000.pth" \
    # --ckpt "../commonsense_pretraining/results_log/semeval_2020_task4/unifiedqa_11b_reasons/bz4_acc2_seqlen75/model_2000.pth" \
    # --ckpt "../commonsense_pretraining/results_log/semeval_2020_task4/unifiedqa_11b_reasons/bz4_acc2_seqlen128/model_10000.pth" \
    # --ckpt "../commonsense_pretraining/results_log/semeval_2020_task4/unifiedqa_11b/bz4_acc2_seqlen75/model_3000.pth" \
    # --ckpt "../commonsense_pretraining/results_log/semeval_2020_task4/unifiedqa_11b/bz4_acc2_seqlen128/model_2500.pth" \
