# Expanded EntangledQA: Common Sense Entangled Questions

## The dataset

This repository contains the Expanded EntangledQA dataset. EntangledQA is a common sense knowledge and reasoning generalization task consisting of 5,000 true/false questions divided across two test sets - the A set and the B set. For every question q in the A set there is an “entangled” question q1 in the B set such that if one knows the answer to q, and one has common sense, one should also know the answer to q1. This entanglement relation is not symmetrical. While there may be many cases in which someone who knows the answer to q1 in the B set should also know the answer to question q in the A set, this will not necessarily be the case. It is also important to note that the entanglement relation is not the same as logical entailment: answers to questions in the A set do not in general logically imply answers to their entangled questions in the B set nor vice versa. Instead, the entanglement relation is one of likely underlying causal understanding; e.g., knowing that ripe apples fall from trees is entangled with knowing that ripe peaches fall from trees, and is entangled with knowing that fruit grows on trees, and so on. Also included here is a 75 question training set and two 425-question dev sets. The training set was the training set for CycIC3. The dev sets are subsets of the test sets for CycIC3.

If you are an MCS participant who wishes to submit results for evaluation on the test sets, please send in 2 prediction files - one for each test set. The predictions should be in .lst format, containing one answer per line, in the same order as the corresponding questions.

## Run the evaluation script
`evaluate_entangled.py` takes predictions for the A question set and the B question set and outputs five accuracy metrics:
1. The accuracy on dataset a
2. The accuracy on dataset b
3. B | A accuracy - the accuracy on questions in dataset b where the corresponding question in dataset a was answered correctly. 
4. The Pearson correlation between A and B accuracy and its p-value.
5. The different between chance on B | A and the ideal B | A score, which is calculated as (B|A accuracy - B accuracy) / (100 - B accuracy)
The predictions should be in .lst format. That is, they should contain one answer per line, in the same order as the corresponding dataset, like so:
~~~
1
0
0
1
0
~~~
PLEASE NOTE that 0 and 1 refer to the indices of the answers, as in a multiple-choice dataset, with 0 being "True" and 1 being "False." If this is counterintuitive, boolean-formatted answers are also acceptable:
~~~
False
True
True
False
True
~~~
The script also requires the label files and the entangledqa_dev_map.csv file from the dataset. If the prediction files are named preds_a.lst and preds_b.lst, an example call might look like this:
```
python3 evaluate_entangled.py \
--a_preds=preds_a.lst \
--b_preds=preds_b.lst \
--a_labels=entangledqa_A_dev_labels.jsonl \
--b_labels=entangledqa_B_dev_labels.jsonl \
--map=entangledqa_dev_map.csv
```

