LST="no_pretrained_seqlen128_released"
LST="semeval20_pretrained_seqlen128_released"
LST="semeval20_com2sense20_pretrained_released"
LST="semeval20_com2sense40_pretrained_released"
LST="semeval20_com2sense60_pretrained_released"
LST="semeval20_com2sense80_pretrained_released"
LST="semeval20_com2sense100_pretrained_released"
LST="semeval20_comparative_pretrained_released"
LST="semeval20_comparative_com2sense60_pretrained_released"
LST="semeval20_comparative_com2sense60_pretrained_no_training_released"
LST="semeval20_comparative_com2sense80_pretrained_no_training_released"
LST="semeval20_comparative_com2sense80_pretrained_devEQA_no_training_released"
LST="semeval20_comparative_dev_EQA_no_training_released"
LST="semeval20_all_comparative_pretrained_no_training_released"
LST="semeval20_comparative_pretrained_no_training_released"
LST="semeval20_comparative_com2sense70_pretrained_no_training_released"

python3 evaluate_entangled.py \
    --cycic3a_preds=prediction_lsts/${LST}-a-pred.lst \
    --cycic3b_preds=prediction_lsts/${LST}-b-pred.lst \
    --cycic3a_labels=datasets/cycic3/cycic3a_released_labels.jsonl \
    --cycic3b_labels=datasets/cycic3/cycic3b_released_labels.jsonl \
    --map=datasets/cycic3/cycic3_released_map.csv \
    --cycic3a_data=datasets/cycic3/cycic3a_released_questions.jsonl \
    --cycic3b_data=datasets/cycic3/cycic3b_released_questions.jsonl \
