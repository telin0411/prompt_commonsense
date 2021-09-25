import pandas as pd
import argparse
from sklearn.metrics import accuracy_score

def compute_accuracy(args):
    mapping = load_origin_entailed_mapping(args.map)
    
    origin_preds = load_predictions(args.cycic3a_preds)
    entailed_preds = load_predictions(args.cycic3b_preds)
    
    origin_labels = load_dataset_file(args.cycic3a_labels)
    entailed_labels = load_dataset_file(args.cycic3b_labels)
    
    origin_accuracy = accuracy_score(origin_labels.correct_answer, origin_preds)
    entailed_accuracy = accuracy_score(entailed_labels.correct_answer, entailed_preds)
    accuracy_dataset = compute_accuracy_dataset(origin_labels, origin_preds, entailed_labels, entailed_preds, mapping)
    origin_correct_idx = accuracy_dataset.origin_prediction == accuracy_dataset.origin_label
    conditional_accuracy = accuracy_score(accuracy_dataset[origin_correct_idx]['entailed_label'],
                                          accuracy_dataset[origin_correct_idx]['entailed_prediction'])
    print("Dataset A accuracy:", origin_accuracy)
    print("Dataset B accuracy:", entailed_accuracy)
    print("Conditional accuracy (B correct | A correct):", conditional_accuracy)

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
    return pd.read_json(filename, lines=True)

def load_predictions(filename):
    return pd.read_csv(filename, header=None, names=['prediction'])

def main():
    parser = argparse.ArgumentParser(description="Evaluate prediction accuracy for entangled datasets")
    parser.add_argument("--cycic3a_labels", type=str, help="Labels (correct) for the cycic3a data", required=True)
    parser.add_argument("--cycic3a_preds", type=str, help="Predictions for the cycic3a data, one per line", required=True)
    parser.add_argument("--cycic3b_labels", type=str, help="Labels (correct) for the cycic3b data", required=True)
    parser.add_argument("--cycic3b_preds", type=str, help="Predictions for the cycic3b dataset, one per line", required=True)
    parser.add_argument("--map", type=str, help="CSV file mapping cycic3a run ids to cycic3b run ids.", required=True)
    parser.add_argument("--cycic3a_data", type=str, help="A jsonl dataset of commonsense questions", required=False)
    parser.add_argument("--cycic3b_data", type=str, help="JSONL dataset of commonsense questions that ought to be answerable if the cycic3a set is answerable.", required=False)
    
    args = parser.parse_args()
    compute_accuracy(args)

main()
