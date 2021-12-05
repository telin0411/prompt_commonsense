# Pretrain on Large Concentrated Explanations

## dataset

### schema:
```python
{
    "statement": str,
    "positive_explanations": [str],  # can be empty list
    "negative_explanations": [str],  # can be empty list
    "answers": str, # can be None
    "data_resource": str,
}
```
### source

- eQASC
- entailment_trees
- CSQA + COS-E


## model
T5-large