#!/usr/bin/env bash

# ** RoBerta-large || EntangledQA **
python3 main.py \
    --mode train \
    --expt_dir ./results_log/SemEval20 \
    --expt_dir ./results_log/SemEval20Comparative \
    --expt_dir ./results_log/EntangledQA \
    --expt_name unifiedqa_11b \
    --model "allenai/unifiedqa-t5-11b" \
    --train_dataset semeval_2020 \
    --train_dataset EntangledQA \
    --train_dataset semeval20_comparative \
    --train_file all \
    --train_file train_all \
    --train_file train \
    --dev_dataset semeval_2020 \
    --dev_dataset semeval20_comparative \
    --dev_dataset EntangledQA \
    --dev_file dev \
    --dev_file dev \
    --dev_file released-a \
    --dev_file train_dev-a \
    --test_dataset semeval_2020 \
    --test_dataset semeval20_comparative \
    --test_dataset EntangledQA \
    --test_file test \
    --test_file dev \
    --test_file released-b \
    --test_file train_dev-b \
    --run semeval20_all \
    --run semeval20_all_pretrained \
    --run semeval20_comparative_pretrained \
    --run semeval20_comparative_com2sense80_pretrained \
    --run semeval20_comparative_com2sense60_pretrained \
    --run com2sense60_pretrained \
    --run semeval20_all_dev_EQA \
    --run semeval20_dev_EQA \
    --seed 1000 \
    --gpu 0,1,2,3,4,5,6,7 \
    --gpu 0,1,2,3 \
    --gpu 0,1,2,3,4,5 \
    --batch_size 4 \
    --seq_len 64 \
    --seq_len 128 \
    --acc_step 16 \
    --acc_step 8 \
    --use_amp F \
    --lr 1e-5 \

    # --ckpt "./results_log/SemEval20Comparative/unifiedqa_11b/semeval20_comparative/ep_1_stp_6.1k_acc_92.5340_allenai_unifiedqa_t5_11b.pth" \
    # --ckpt "./results_log/SemEval20Comparative/unifiedqa_11b/semeval20_comparative/ep_2_stp_12.2k_acc_92.6313_allenai_unifiedqa_t5_11b.pth" \
    # --ckpt "./results_log/com2sense/unifiedqa_11b/semeval20_comparative_com2sense/ep_6_stp_20.2k_acc_81.5327_allenai_unifiedqa_t5_11b.pth" \
    # --ckpt "./results_log/com2sense/unifiedqa_11b/semeval20_comparative_com2sense60/ep_3_stp_14.6k_acc_81.2500_allenai_unifiedqa_t5_11b.pth" \
    # --ckpt "./results_log/com2sense/unifiedqa_11b/semeval20_comparative_com2sense60/ep_4_stp_15.8k_acc_81.9095_allenai_unifiedqa_t5_11b.pth" \
    # --ckpt "./results_log/com2sense/unifiedqa_11b/semeval20_com2sense60/ep_2_stp_2.4k_acc_75.7538_allenai_unifiedqa_t5_11b.pth" \

    # --ckpt "./results_log/SemEval20/unifiedqa_11b/semeval20_all/ep_2_stp_12.0k_acc_98.3704_allenai_unifiedqa_t5_11b.pth" \
    # --ckpt "./results_log/SemEval20/unifiedqa_11b/semeval20_all/ep_1_stp_6.0k_acc_95.1608_allenai_unifiedqa_t5_11b.pth" \
    # --ckpt "" \
    # --train_dataset semeval_2020,com2sense \
    # --train_file all,train \
