from pathlib import Path

output_dir = Path("task_data", "semeval+comparative")
for split in ("train", "dev"):
    for convert in ("source", "target"):
        FILE_NAME = split + "." + convert

        with open("task_data/semeval-t5/" + FILE_NAME) as fp:
            data = fp.read()
        with open("task_data/cycic-real-comparative/" + FILE_NAME) as fp:
            data2 = fp.read()

        data += data2

        output_dir.mkdir(parents=True, exist_ok=True)
        with open("task_data/semeval+comparative/" + FILE_NAME, "w") as fp:
            fp.write(data)
