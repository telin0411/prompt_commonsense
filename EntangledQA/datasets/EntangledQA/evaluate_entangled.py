import pandas as pd
import argparse
from sklearn.metrics import accuracy_score
from scipy.stats import pearsonr
import numpy as np

def compute_accuracy(args):
    mapping = load_origin_entailed_mapping(args.map)
    
    origin_preds = load_predictions(args.a_preds)
    entailed_preds = load_predictions(args.b_preds)
    
    origin_labels = load_dataset_file(args.a_labels)
    entailed_labels = load_dataset_file(args.b_labels)
    
    origin_accuracy = accuracy_score(origin_labels.correct_answer, origin_preds)
    entailed_accuracy = accuracy_score(entailed_labels.correct_answer, entailed_preds)
    accuracy_dataset = compute_accuracy_dataset(origin_labels, origin_preds, entailed_labels, entailed_preds, mapping)
    origin_correct_idx = accuracy_dataset.origin_prediction == accuracy_dataset.origin_label
    conditional_accuracy = accuracy_score(accuracy_dataset[origin_correct_idx]['entailed_label'],
                                          accuracy_dataset[origin_correct_idx]['entailed_prediction'])
    entailed_correct_idx = accuracy_dataset.entailed_prediction == accuracy_dataset.entailed_label
    r, p = pearsonr(origin_correct_idx, entailed_correct_idx) #note: this assumes a one-to-one mapping of origin to entailed. Could be extended to one-to-many by computing the average of entailed accuracy for each origin
    print("Dataset A accuracy:", origin_accuracy)
    print("Dataset B accuracy:", entailed_accuracy)
    print("Conditional accuracy (B correct | A correct): {:.3f}".format(conditional_accuracy))
    print("Pearson correlation r(A, B): {:.3f}".format(r))
    print("p-value: {:.3f}".format(p))
    percent_recovered = (conditional_accuracy - entailed_accuracy) / (1.0 - entailed_accuracy)
    print("Percent error recovered by entailment ((B|A accuracy) - B accuracy) / (1 - B accuracy): {:.3f}".format(percent_recovered))

def compute_accuracy_dataset(origin_labels, origin_preds, entailed_labels, entailed_preds, mapping):
    # get a mapping of run_id -> label and prediction for each dataset
    origin_results = pd.concat([origin_labels['run_id'], origin_labels['correct_answer'], origin_preds], axis=1).rename(columns={"prediction":"origin_prediction", "correct_answer":"origin_label"})
    entailed_results = pd.concat([entailed_labels['run_id'], entailed_labels['correct_answer'], entailed_preds], axis=1).rename(columns={"prediction":"entailed_prediction", "correct_answer":"entailed_label"})
    # now merge them using map as a key
    accuracy_dataset = mapping.merge(origin_results, how='left', left_on='origin', right_on='run_id').drop('run_id', 1)
    accuracy_dataset = accuracy_dataset.merge(entailed_results, how='left', left_on='entailed', right_on='run_id').drop('run_id', 1)
    return accuracy_dataset


def load_origin_entailed_mapping(filename):
    return pd.read_csv(filename)

def load_dataset_file(filename):
    # @hack: assume True is answer index 0 and False is index 1
    df = pd.read_json(filename, lines=True, dtype={'correct_answer':bool})
    df.correct_answer = ~df.correct_answer
    return df

def load_predictions(filename):
    df = pd.read_csv(filename, header=None, names=['prediction'])
    if df.prediction.dtype == int:
        df.prediction = df.prediction.astype(bool)
        df.prediction = ~df.prediction
    else:
        df.prediction = df.prediction.astype(bool)
    return df

def main():
    parser = argparse.ArgumentParser(description="Evaluate prediction accuracy for entangled datasets")
    parser.add_argument("--a_labels", type=str, help="Labels (correct) for the origin data", required=True)
    parser.add_argument("--a_preds", type=str, help="Predictions for the origin data, one per line", required=True)
    parser.add_argument("--b_labels", type=str, help="Labels (correct) for the entailed data", required=True)
    parser.add_argument("--b_preds", type=str, help="Predictions for the entailed dataset, one per line", required=True)
    parser.add_argument("--map", type=str, help="CSV file mapping origin run ids to entailed run ids.", required=True)
    parser.add_argument("--a_data", type=str, help="A jsonl dataset of commonsense questions", required=False)
    parser.add_argument("--b_data", type=str, help="JSONL dataset of commonsense questions that ought to be answerable if the origin set is answerable.", required=False)
    
    args = parser.parse_args()
    compute_accuracy(args)

main()
