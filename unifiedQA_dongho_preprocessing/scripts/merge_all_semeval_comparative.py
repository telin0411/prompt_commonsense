from pathlib import Path

output_dir = Path("task_data", "semeval_comparative_all")
for convert in ("source", "target"):
    data = ""
    for split in ("train", "dev", "test"):
        print(split, convert)
        FILE_NAME = split + "." + convert

        data1 = ""
        data2 = ""

        with open("task_data/semeval-t5/" + FILE_NAME) as fp:
            data1 = fp.read()
        try:
            with open("task_data/cycic-real-comparative/" + FILE_NAME) as fp:
                data2 = fp.read()
        except:
            pass

        data += data1 + data2

    output_dir.mkdir(parents=True, exist_ok=True)
    with open("task_data/semeval_comparative_all/" + "train.{}".format(convert), "w") as fp:
        fp.write(data)
