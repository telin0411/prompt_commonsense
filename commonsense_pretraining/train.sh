python3 main.py \
    --mode train \
    --expt_dir results_log/semeval20 \
    --expt_name roberta_reasons \
    --data_dir ./datasets/semeval_2020_task4 \
    --model t5-large \
    --model roberta-large \
    --run_name demo \
    --lr 1e-6 \
    --batch_size 64
