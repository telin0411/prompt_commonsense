import argparse
import csv
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path, default=Path("data_dir", "semeval"))
    p.add_argument("--output-dir", type=Path, default=Path("task_data", "semeval-t5"))
    p.add_argument("--subset", type=float, default=None)
    args = p.parse_args()

    for in_split, out_split in (("train", "train"), ("dev", "dev"), ("test", "test")):
        question_path = args.input_dir / f"semeval_{in_split}_data.csv"
        label_path = args.input_dir / f"semeval_{in_split}_answer.csv"

        datareader = csv.reader(question_path.open(encoding="utf-8-sig"))
        labelreader = csv.reader(label_path.open(encoding="utf-8-sig"))
        # if the label -> incorrect.

        merge_data = []
        merge_label = []
        for data, labels in zip(datareader, labelreader):
            if labels[1] == "0":
                merge_data.append(data[1])
                merge_label.append("no")

                merge_data.append(data[2])
                merge_label.append("yes")

            elif labels[1] == "1":
                merge_data.append(data[1])
                merge_label.append("yes")

                merge_data.append(data[2])
                merge_label.append("no")

        args.output_dir.mkdir(parents=True, exist_ok=True)
        # if we finetune model with these statement style data, then it is okay ?
        source_file = open(args.output_dir / f"{out_split}.source", "w")
        target_file = open(args.output_dir / f"{out_split}.target", "w")
        for question, label in zip(merge_data, merge_label):
            if question[-1] != ".":
                question = question + "."
            question = question[:-1] + "? \\n"
            source_file.write(f"{question}\n")
            target_file.write(f"{label}\n")


if __name__ == "__main__":
    main()
