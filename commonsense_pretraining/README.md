# CS-Data-Dev

Models for the Commonsense Dataset Construction project



---


## Table of Contents

- [Datasets](#datasets)
- [Models](#Models)
- [Demo](#Demo)
- [Inference](#inference)
- [Training](#training)
- [Results](#results)

<!---
- [Visualization](#visualization)
-->



---

## Datasets

- [SemEval'20 Task #4](https://github.com/wangcunxiang/SemEval2020-Task4-Commonsense-Validation-and-Explanation) 

    The directory is structured as follows:

    ```
    datasets
    └── semeval_2020_task4
        ├── dev.csv
        ├── test.csv
        └── train.csv
    ```



---
## Models


- RoBERTa-large




---

## Demo

Please download the model weights from [Google-drive](https://drive.google.com/drive/folders/1oVwdYxSOdTLtsA0FFwwgXBzu6OZxR0KV?usp=sharing),
and place it in <i>model_ckpt </i>, and set the `--ckpt` flag accordingly.

Run the following demo script:

```bash
$ python3 demo.py \
--inp_txt ./datasets/ours/inp.txt \
--ckpt model_ckpt/model.pth \
--model roberta-large --gpu 0
```

To save the model predictions to a <b>CSV</b> file, simply use the flag:  <br>
`--pred_csv ./output_pred.csv`

It also supports <b>RAM </b> mode -- i.e. users can load model to RAM once, 
and re-compute the results as they modify the input txt. <br>
Simply set the following flag `--use_ram true`.


> We will also maintain a sub-readme for details regarding model checkpoints & training logs.

---

## Inference

Run the following script for evaluation:

```bash
$ python3 main.py \
--mode test
```




---

## Training

Run the following script for training:

```bash
$ python3 main.py --mode train \
--expt_dir results_log/semeval20 --expt_name roberta_reasons \
--data_dir ./datasets/semeval_2020_task4 --model roberta-base \
--run_name demo --lr 1e-6 --batch_size 64
```




---

## Results

- SemEval'20 #4A

    | Models        | Val-acc       | Test-acc      |
    | :---          | :---          | :---          |
    | roberta-base  | 80.78         |               |
    | roberta-large | 87.30         |               |

<br>


---


## References
[1]  SemEval-2020 Task 4: Commonsense Validation and Explanation ([paper](https://arxiv.org/abs/2007.00236)) <br>