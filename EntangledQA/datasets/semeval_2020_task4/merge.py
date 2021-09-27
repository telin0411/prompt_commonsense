fo = open("all.csv", "w")

f_train = open("train.csv")
f_dev = open("dev.csv")
f_test = open("test.csv")

for f in [f_train, f_dev, f_test]:
    for line in f:
        fo.write(line)
    pass

fo.close()
f_train.close()
f_dev.close()
f_test.close()
