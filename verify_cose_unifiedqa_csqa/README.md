

# Com2Sense


This repo contains the dataset and baseline model weights for 
[Com2Sense](https://arxiv.org/abs/2106.00969) Benchmark.

*TODO*:
 
We will create the Official Leaderboard by August 1st, 
and corresponding evaluation script (eval.py)

Add Link to BigBench

---

  
 
## Table of Contents

  

-  [Dataset](#dataset)

-  [Models](#Models)

-  [Training](#training)

-  [Evaluation](#evaluation)

-  [Leaderboard](#leaderboard)
  
---

  

## Dataset

The directory is structured as follows:
 
```
com2sense
├── train.json
├── dev.json
└── test.json
```

Each file has the following format:

```

[   
    {
        "id": "",
        "sent_1": "",
        "sent_2": "",
        "label_1": "",
        "label_2": "",
        "domain": "", 
        "scenario": "",
        "numeracy": ""
    },
    ...
  ]

```


---

## Models

| Model             | Std / Pair Accuracy | Weights  |
| ---------         | ------------------- | --------- |
| UnifiedQA-3B      | 71.31 / 51.26       | [Link](https://drive.google.com/file/d/1uQnxZAkSoDc8JEmESzTl0XVE8kHpm_10/view?usp=sharing)|
| DeBerta-large     | 63.53 / 45.30       | Link |



---

## Training

For training we provide a sample script, with custom arguments ([train.sh](./train.sh))
  

```bash
$ python3 main.py \
--mode train \
--dataset com2sense \
--model roberta-large \
--expt_dir ./results \
--expt_name roberta_large_experiment \
--run_name demo \
--seq_len 128 \
--epochs 100 \
--batch_size 16 \
--acc_step 4 \
--lr 1e-5 \
--log_interval 500 \
--gpu_ids 0,1,2,3 \
--use_amp T \
-data_parallel
```

The Train & Validation metrics are logged to TensorBoard.
 
```bash
$ tensorboard --logdir path-to-expt-log-directory
```



---

 
## Evaluation

  To evaluate trained models you can run the following script, with custom arguments:
  

```bash
$ python3 main.py \
--mode test \
--model roberta-large \
--dataset com2sense \
--ckpt ./path_to_model.pth
--test_file test \
--pred_file roberta_large_results.csv 
```
---
   

## Leaderboard

> TODO: August 1st

---
