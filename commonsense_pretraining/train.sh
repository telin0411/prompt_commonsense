python3 main.py \
    --mode train \
    --expt_dir /data1/telinwu/research/prompt_2021/prompt_commonsense/commonsense_pretraining/results_log/semeval_2020_task4 \
    --expt_dir "./results_log/semeval_2020_task4" \
    --expt_name unifiedqa_11b \
    --data_dir ./datasets/semeval_2020_task4 \
    --model "allenai/unifiedqa-t5-11b" \
    --run_name bz4_acc2_seqlen75 \
    --run_name bz4_acc2_seqlen128 \
    --lr 1e-5 \
    --batch_size 4 \
    --gpu_ids 0,1,2,3 \
    --gpu_ids 0,1,2,3,4,5,6,7 \
    --use_amp F \
    --acc_step 2 \
    --save_interval 1000 \
    --seq_len 128 \
    --use_reason F \
    # --batch_size 1 \
    # --ckpt "/data1/telinwu/research/prompt_2021/prompt_commonsense/commonsense_pretraining/results_log/semeval_2020_task4/unifiedqa_11b_reasons/unifiedqa_11b_reasons/model_500.pth" \
    # --ckpt "./results_log/semeval_2020_task4/unifiedqa_11b_reasons/unifiedqa_11b_reasons/model_500.pth" \
    # --mode "test" \
