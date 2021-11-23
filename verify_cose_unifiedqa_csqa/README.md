# Verify Cos-E on CSQA with Unified QA

## step 0 Prepare datasets
Combine questions from csqa and explanations from cose

data file saved at ucla plus-a100 `/data1/yixiao/datasets/cose_for_generation`

file structure

```python
{
    "question": "fefefaefgg",
    "choices":{
        "A": "gery",
        "B": "frg",
        "C": "few"
    },
    "explanation": "gnerghi",
    "answer": "gnerghi"
}
```

## step 1
Train T5 on Cos-E to generate explanations for Common Sense QA

model saved at ucla plus-a100 `/data1/yixiao/ckpt/verify_cose_unified_csqa/t5_finetuned_cose.pth`

generated data saved at ucla plus-a100 `/data1/yixiao/datasets/cose_t5_generated/`

## step 2
* Fine-tune UnifiedQA-3B on CSQA + T5 generated Cos-E

* Fine-tune UnifiedQA-3B on CSQA only

model saved at ucla plus-a100 `/data1/yixiao/ckpt/verify_cose_unified_csqa/unifiedqa_sta_plus_exp.pth`

model saved at ucla plus-a100 `/data1/yixiao/ckpt/verify_cose_unified_csqa/unifiedqa_sta_only.pth`

