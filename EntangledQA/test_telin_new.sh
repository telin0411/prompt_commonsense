#!/usr/bin/env bash

# ** RoBerta-large || EntangledQA **
python3 main.py \
    --mode dev-b \
    --mode dev-a \
    --mode hard_few \
    --mode released-b \
    --mode released-a \
    --mode new_test_a \
    --mode new_test_b \
    --expt_dir ./results_log/EntangledQA \
    --expt_name unifiedqa_11b \
    --model "allenai/unifiedqa-t5-11b" \
    --train_dataset EntangledQA \
    --dev_dataset EntangledQA \
    --test_dataset EntangledQA \
    --run com2sense60_pretrained_dev-b \
    --run com2sense60_pretrained_dev-a \
    --run com2sense60_pretrained_released-b \
    --run com2sense60_pretrained_released-a \
    --run semeval20_comparative_com2sense60_pretrained_dev-b \
    --run semeval20_comparative_com2sense60_pretrained_dev-a \
    --run semeval20_comparative_pretrained_dev-b \
    --run semeval20_comparative_pretrained_dev-a \
    --run semeval20_comparative_pretrained_released-a \
    --run semeval20_comparative_pretrained_released-b \
    --run semeval20_comparative_pretrained_no_training_dev-b \
    --run semeval20_comparative_pretrained_no_training_dev-a \
    --run semeval20_comparative_com2sense60_pretrained_released-b \
    --run semeval20_comparative_com2sense60_pretrained_released-a \
    --run semeval20_comparative_com2sense60_pretrained_no_training_released-b \
    --run semeval20_comparative_com2sense60_pretrained_no_training_released-a \
    --run semeval20_comparative_pretrained_no_training_train-b \
    --run semeval20_comparative_pretrained_no_training_train-a \
    --run semeval20_comparative_com2sense80_pretrained_no_training_released-b \
    --run semeval20_comparative_com2sense80_pretrained_no_training_released-a \
    --run semeval20_all_comparative_pretrained_no_training_released-b \
    --run semeval20_all_comparative_pretrained_no_training_released-a \
    --run semeval20_comparative_dev_EQA_no_training_released-b \
    --run semeval20_comparative_dev_EQA_no_training_released-a \
    --run semeval20_comparative_com2sense70_pretrained_no_training_hard_few \
    --run semeval20_comparative_com2sense70_pretrained_no_training_released-b \
    --run semeval20_comparative_com2sense70_pretrained_no_training_released-a \
    --run semeval20_comparative_com2sense70_pretrained_no_training_new_test-a \
    --run semeval20_comparative_com2sense70_pretrained_no_training_new_test-b \
    --run semeval20_comparative_com2sense80_pretrained_no_training_new_test-a \
    --run semeval20_comparative_com2sense80_pretrained_no_training_new_test-b \
    --run semeval20_comparative_no_training_released-a \
    --run semeval20_comparative_no_training_released-b \
    --seed 808 \
    --gpu 0 \
    --gpu 0,1,2,3,4,5,6,7 \
    --gpu 0,1,2,3,4,5 \
    --gpu 0,1,2,3 \
    --seq_len 128 \
    --seq_len 64 \
    --batch_size 1 \
    --acc_step 1 \
    --use_amp F \
    --lr 5e-6 \
    --ckpt "" \
    --ckpt "./results_log/EntangledQA/unifiedqa_11b/semeval20_comparative_pretrained/ep_2_stp_12.3k_acc_84.5519_allenai_unifiedqa_t5_11b.pth" \
    --ckpt "./results_log/EntangledQA/unifiedqa_11b/semeval20_comparative_com2sense60_pretrained/ep_4_stp_15.8k_acc_85.6132_allenai_unifiedqa_t5_11b.pth" \
    --ckpt "./results_log/com2sense/unifiedqa_11b/semeval20_comparative_com2sense60/ep_4_stp_15.8k_acc_81.9095_allenai_unifiedqa_t5_11b.pth" \
    --ckpt "./results_log/SemEval20Comparative/unifiedqa_11b/semeval20_comparative/ep_3_stp_18.4k_acc_93.0447_allenai_unifiedqa_t5_11b.pth" \
    --ckpt "./results_log/com2sense/unifiedqa_11b/semeval20_comparative_com2sense80/ep_4_stp_21.5k_acc_80.6533_allenai_unifiedqa_t5_11b.pth" \
    --ckpt "./results_log/SemEval20Comparative/unifiedqa_11b/semeval20_all_dev_EQA/ep_4_stp_28.5k_acc_88.3721_allenai_unifiedqa_t5_11b.pth" \
    --ckpt "./results_log/com2sense/unifiedqa_11b/semeval20_comparative_com2sense80_dev_EQA/ep_6_stp_24.7k_acc_90.4070_allenai_unifiedqa_t5_11b.pth" \
    --ckpt "./results_log/com2sense/unifiedqa_11b/semeval20_comparative_com2sense80_v2/ep_10_stp_31.1k_acc_81.5955_allenai_unifiedqa_t5_11b.pth" \
    --ckpt "./results_log/com2sense/unifiedqa_11b/semeval20_comparative_com2sense80_v2/ep_6_stp_24.7k_acc_81.3442_allenai_unifiedqa_t5_11b.pth" \
    --ckpt "./results_log/com2sense/unifiedqa_11b/semeval20_comparative_com2sense80_v2/ep_4_stp_21.5k_acc_79.9623_allenai_unifiedqa_t5_11b.pth" \
    --ckpt "./results_log/EntangledQA/unifiedqa_11b/semeval20_dev_EQA/ep_9_stp_55.1k_acc_90.1163_allenai_unifiedqa_t5_11b.pth" \
    --ckpt "./results_log/com2sense/unifiedqa_11b/semeval20_comparative_com2sense70/ep_4_stp_21.1k_acc_81.0746_allenai_unifiedqa_t5_11b.pth" \
    --ckpt "./results_log/com2sense/unifiedqa_11b/semeval20_comparative_com2sense80/ep_4_stp_21.5k_acc_80.6533_allenai_unifiedqa_t5_11b.pth" \
    --ckpt "./results_log/SemEval20Comparative/unifiedqa_11b/semeval20_comparative/ep_3_stp_18.4k_acc_93.0447_allenai_unifiedqa_t5_11b.pth" \
