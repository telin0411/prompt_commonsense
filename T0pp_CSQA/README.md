# T0pp on CSQA

Goal: generate explanations by T0pp and see if the accuracy on CSQA increase

## Step 1
Mode: Inference

Model: T0pp

Input: QUESTIONS || CHOICES. My common sense tells my

Output: EXPLANATION

## Step 2
Mode: Training

Model: UnifiedQA

Input: EXPLANATION || QUESTIONS || CHOICES

Target: GT-ANSWER

## Dataset

```json
{
  "id":"1afa02df02c908a558b4036e80242fac",
  "question":"A revolving door is convenient for two direction travel, but it also serves as a security measure at a what?",
  "choices":
  {
    "A":"bank", 
    "B":"library", 
    "C":"department store", 
    "D":"mall",
    "E":"new york"
  },
  "explanation":"Rivers flow trough valleys.",
  "answer":"bank"
}
```

Train: `/local1/telinwu/yixiao/datasets/cose_for_generation/train.json`

Dev: `/local1/telinwu/yixiao/datasets/cose_for_generation/dev.json`

