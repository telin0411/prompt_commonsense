# CycIC3: Common Sense Entangled Questions

## The dataset

This repository contains the CycIC3 dataset. CycIC3 is a common sense knowledge and reasoning generalization task consisting of 1,000 true/false questions divided across two test sets - CycIC3a and CycIC3b. For every question q in CycIC3a there is an “entangled” question q1 in CycIC3b such that if one knows the answer to q, and one has common sense, one should also know the answer to q1 . This entanglement relation is not symmetrical. While there may be many cases in which someone who knows the answer to q1 in the B set should also know the answer to question q in the A set, this will not necessarily be the case. It is also important to note that the entanglement relation is not the same as logical entailment: answers to questions in the A set do not in general logically imply answers to their entangled questions in the B set nor vice versa. Instead, the entanglement relation is one of likely underlying causal understanding; e.g., knowing that ripe apples fall from trees is entangled with knowing that ripe peaches fall from trees, and is entangled with knowing that fruit grows on trees, and so on. Also included here are two 100 question dev question sets, entangled in the same way as the test sets, and a 75 question training set. 

If you are an MCS participant who wishes to submit results for evaluation on the test sets, please send in 2 prediction files - one for each test set. The predictions should be in .lst format, containing one answer per line, in the same order as the corresponding dataset.

## Run the evaluation script
`evaluate_entangled.py` takes predictions for the Cycic3a and Cycic3b datasets and outputs three accuracy metrics:
1. The accuracy on dataset a
2. The accuracy on dataset b
3. The accuracy on questions in dataset b where the corresponding question in dataset a was answered correctly.
The predictions should be in `.lst` format. That is, they should contain one answer per line, in the same order as the corresponding dataset, like so:
~~~
1
0
0
1
0
~~~
The script also requires the label files and the `linked_questions.csv` file from the dataset. If the prediction files are named `preds_a.lst` and `preds_b.lst`, an example call might look like this:
```
python3 evaluate_entangled.py \
--cycic3a_preds=preds_a.lst \
--cycic3b_preds=preds_b.lst \
--cycic3a_labels=cycic3a_labels.jsonl \
--cycic3b_labels=cycic3b_labels.jsonl \
--map=cycic3_dev_question_map.csv
```
