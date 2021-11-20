# ** T5-large || SemEval **
python3 main.py \
--mode train \
--expt_dir ./results_log/com2sense \
--expt_name t5_large \
--model t5-large \
--dataset com2sense \
--run bs_4 \
--batch_size 4  \
--seq_len 128 \
--gpu 0 --use_amp F
