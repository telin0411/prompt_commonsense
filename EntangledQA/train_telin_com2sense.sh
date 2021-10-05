#!/usr/bin/env bash

# ** RoBerta-large || EntangledQA **
python3 main.py \
    --mode train \
    --expt_dir ./results_log/SemEval20 \
    --expt_dir ./results_log/SemEval20Comparative \
    --expt_dir ./results_log/EntangledQA \
    --expt_dir ./results_log/com2sense \
    --expt_name unifiedqa_11b \
    --model "allenai/unifiedqa-t5-11b" \
    --train_dataset semeval_2020 \
    --train_dataset semeval20_comparative \
    --train_dataset EntangledQA \
    --train_dataset com2sense \
    --train_file all \
    --train_file train \
    --train_file train-80 \
    --dev_dataset semeval_2020 \
    --dev_dataset semeval20_comparative \
    --dev_dataset com2sense \
    --dev_dataset EntangledQA \
    --dev_file dev \
    --dev_file dev \
    --dev_file released-a \
    --dev_file dev-80 \
    --dev_file train_dev-a \
    --test_dataset semeval_2020 \
    --test_dataset semeval20_comparative \
    --test_dataset com2sense \
    --test_dataset EntangledQA \
    --test_file test \
    --test_file dev \
    --test_file released-b \
    --test_file test-80 \
    --test_file train_dev-b \
    --run semeval20_all \
    --run semeval20_all_pretrained \
    --run semeval20_comparative_pretrained \
    --run semeval20_comparative_com2sense60 \
    --run semeval20_com2sense60 \
    --run semeval20_comparative_com2sense80 \
    --run semeval20_comparative_com2sense80_dev_EQA \
    --seed 1000 \
    --gpu 0,1,2,3,4,5,6,7 \
    --gpu 0,1,2,3 \
    --gpu 0,1,2,3,4,5 \
    --batch_size 4 \
    --seq_len 64 \
    --seq_len 128 \
    --acc_step 4 \
    --use_amp F \
    --lr 1e-5 \
    --ckpt "./results_log/SemEval20Comparative/unifiedqa_11b/semeval20_comparative/ep_1_stp_6.1k_acc_92.5340_allenai_unifiedqa_t5_11b.pth" \
    --ckpt "./results_log/SemEval20Comparative/unifiedqa_11b/semeval20_comparative/ep_2_stp_12.2k_acc_92.6313_allenai_unifiedqa_t5_11b.pth" \
    --ckpt "./results_log/SemEval20Comparative/unifiedqa_11b/semeval20_comparative/ep_3_stp_18.4k_acc_93.0447_allenai_unifiedqa_t5_11b.pth" \
    # --ckpt "./results_log/SemEval20/unifiedqa_11b/semeval20_all/ep_2_stp_12.0k_acc_98.3704_allenai_unifiedqa_t5_11b.pth" \
    # --ckpt "./results_log/SemEval20/unifiedqa_11b/semeval20_all/ep_1_stp_6.0k_acc_95.1608_allenai_unifiedqa_t5_11b.pth" \
    # --ckpt "" \
    # --train_dataset semeval_2020,com2sense \
    # --train_file all,train \
