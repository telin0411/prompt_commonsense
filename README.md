# prompt_commonsense
## 2022/01/19/  init RL
Experientment 1

Stage 1
-   Training:  67% ECQA_train +  0% Com2Sense_train
- Validation: 100% ECQA_valid +  0% Com2Sense_valid
Find the best model on validation set

Stage 2
-   Training:  33% ECQA_train +  33% Com2Sense_train
- Validation: 100% ECQA_valid + 100% Com2Sense_valid
Find models whose accuracy are 5% lower than the best model that doesn't mask ou the rest of explanations and doesn't contain Com2Sense

Stage 3
-   Training: 0%  ECQA +  67% Com2Sense_train
- Validation: 0%  ECQA + 100% Com2Sense_valid
Find the best model on validation set

Test Set, test accuracy and ROUGE
- ECQA: 30%, fixed
- Com2Sense 30%, fixed

Train/Valid/Test = 60/10/30:
- ECQA_train/ECQA_valid/ECQA_test = 60/10/30
- Com2Sense_train/Com2Sense_valid/Com2Sense_test = 60/10/30

## Environment
- python-3.8.12
- torch-1.10.0 
- huggingface-hub-0.1.2 
- numpy-1.21.4 
- tokenizers-0.10.3 
- tqdm-4.62.3
- transformers-4.12.5
- pandas-1.3.4
- tensorboard-2.7.0
- sentencepiece-0.1.96 (very important)
