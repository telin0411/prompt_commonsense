import pandas as pd
import argparse
from sklearn.metrics import accuracy_score
from scipy.stats import pearsonr
import numpy as np


def compute_accuracy(args):
    mapping = load_origin_entailed_mapping(args.map)

    origin_preds = load_predictions(args.cycic3a_preds)
    entailed_preds = load_predictions(args.cycic3b_preds)

    # For bugs in option0:True and option1:False
    for index, row in origin_preds.iterrows():
        if row['prediction'] == 'yes': row['prediction'] = 1
        if row['prediction'] == 'no': row['prediction'] = 0
        row['prediction'] = 1 - row['prediction']
    for index, row in entailed_preds.iterrows():
        if row['prediction'] == 'yes': row['prediction'] = 1
        if row['prediction'] == 'no': row['prediction'] = 0
        row['prediction'] = 1 - row['prediction']

    origin_labels = load_dataset_file(args.cycic3a_labels)
    entailed_labels = load_dataset_file(args.cycic3b_labels)

    origin_accuracy = accuracy_score(origin_labels.correct_answer.tolist(), origin_preds['prediction'].tolist())
    entailed_accuracy = accuracy_score(entailed_labels.correct_answer.tolist(), entailed_preds['prediction'].tolist())
    accuracy_dataset = compute_accuracy_dataset(origin_labels, origin_preds, entailed_labels, entailed_preds, mapping)
    origin_correct_idx = accuracy_dataset.origin_prediction == accuracy_dataset.origin_label
    conditional_accuracy = accuracy_score(accuracy_dataset[origin_correct_idx]['entailed_label'].tolist(),
                                          accuracy_dataset[origin_correct_idx]['entailed_prediction'].tolist())
    entailed_correct_idx = accuracy_dataset.entailed_prediction == accuracy_dataset.entailed_label
    r, p = pearsonr(origin_correct_idx,
                    entailed_correct_idx)  # note: this assumes a one-to-one mapping of origin to entailed. Could be extended to one-to-many by computing the average of entailed accuracy for each origin
    print("Dataset A accuracy:", origin_accuracy)
    print("Dataset B accuracy:", entailed_accuracy)
    print("Conditional accuracy (B correct | A correct): {:.3f}".format(conditional_accuracy))
    print("Pearson correlation r(A, B): {:.3f}".format(r))
    print("p-value: {:.3f}".format(p))
    # total percent where correct(A) matches correct(B)
    print("Percent matching correctness: {:.3f}".format(
        np.sum(origin_correct_idx == entailed_correct_idx) / len(accuracy_dataset)))


    origin_data = load_dataset_file(args.cycic3a_data)
    entailed_data = load_dataset_file(args.cycic3b_data)
    origin_categories = origin_data["categories"].tolist()
    entailed_categories = entailed_data["categories"].tolist()

    origin_labels_list = origin_labels.correct_answer.tolist()
    entailed_labels_list = entailed_labels.correct_answer.tolist()
    origin_preds_list = origin_preds['prediction'].tolist()
    entailed_preds_list = entailed_preds['prediction'].tolist()
    labels_list = origin_labels_list + entailed_labels_list
    preds_list = origin_preds_list + entailed_preds_list
    categories_list = origin_categories + entailed_categories

    import pprint
    assert len(labels_list) == len(preds_list) == len(categories_list)
    categories_perf = {}
    for i in range(len(labels_list)):
        lab = labels_list[i]
        pre = preds_list[i]
        cats = categories_list[i]
        for cat in cats:
            if cat not in categories_perf:
                categories_perf[cat] = {
                    "preds": [],
                    "labels": [],
                }
            categories_perf[cat]["preds"].append(pre)
            categories_perf[cat]["labels"].append(lab)

    pand_list = []
    for cat in sorted(categories_perf):
        cat_perf = categories_perf[cat]
        score = accuracy_score(cat_perf["preds"], cat_perf["labels"])
        pand_list.append([cat, score, len(cat_perf["preds"])])
    df = pd.DataFrame(pand_list, columns =["Category", "Accuracy", "Count"])
    print(df[df["Count"] > 10])
    pass

def compute_accuracy_dataset(origin_labels, origin_preds, entailed_labels, entailed_preds, mapping):
    # get a mapping of run_id -> label and prediction for each dataset
    origin_results = pd.concat([origin_labels['run_id'], origin_labels['correct_answer'], origin_preds], axis=1).rename(
        columns={"prediction": "origin_prediction", "correct_answer": "origin_label"})
    entailed_results = pd.concat([entailed_labels['run_id'], entailed_labels['correct_answer'], entailed_preds],
                                 axis=1).rename(
        columns={"prediction": "entailed_prediction", "correct_answer": "entailed_label"})
    # now merge them using map as a key
    accuracy_dataset = mapping.merge(origin_results, how='left', left_on='origin', right_on='run_id').drop('run_id', 1)
    accuracy_dataset = accuracy_dataset.merge(entailed_results, how='left', left_on='entailed', right_on='run_id').drop(
        'run_id', 1)
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
    parser.add_argument("--cycic3a_preds", type=str, help="Predictions for the cycic3a data, one per line",
                        required=True)
    parser.add_argument("--cycic3b_labels", type=str, help="Labels (correct) for the cycic3b data", required=True)
    parser.add_argument("--cycic3b_preds", type=str, help="Predictions for the cycic3b dataset, one per line",
                        required=True)
    parser.add_argument("--map", type=str, help="CSV file mapping cycic3a run ids to cycic3b run ids.", required=True)
    parser.add_argument("--cycic3a_data", type=str, help="A jsonl dataset of commonsense questions", required=False)
    parser.add_argument("--cycic3b_data", type=str,
                        help="JSONL dataset of commonsense questions that ought to be answerable if the cycic3a set "
                             "is answerable.",
                        required=False)

    args = parser.parse_args()
    compute_accuracy(args)


main()
