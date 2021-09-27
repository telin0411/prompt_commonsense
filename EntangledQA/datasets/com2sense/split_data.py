import sys
import json

def read_json_file(path):
    data = json.load(open(path))
    print(len(data), type(data))
    return data


data = []

data += read_json_file("train.json")
data += read_json_file("dev.json")
data += read_json_file("test.json")

print(len(data))
data_len = len(data)

train_ratio_int = int(sys.argv[1])
train_ratio = train_ratio_int / 100.
print(train_ratio)

train_len = int(data_len * train_ratio)
dev_len = int(data_len * (1-train_ratio)/2)

train_data = data[:train_len]
dev_data = data[train_len:train_len+dev_len]
test_data = data[train_len+dev_len:]

print(len(train_data))
print(len(dev_data))
print(len(test_data))

with open("train-{}.json".format(train_ratio_int), "w") as outf:
    json.dump(train_data, outf)
with open("dev-{}.json".format(train_ratio_int), "w") as outf:
    json.dump(dev_data, outf)
with open("test-{}.json".format(train_ratio_int), "w") as outf:
    json.dump(test_data, outf)
