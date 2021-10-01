# >>> import csv
# >>> with open('eggs.csv', newline='') as csvfile:
# ...     spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
# ...     for row in spamreader:
# ...         print(', '.join(row)

import csv

dev = open('Dev Data/subtaskA_dev_data.csv', 'r')
dev_answer = open('Dev Data/subtaskA_gold_answers.csv', 'r')

train = open('Training  Data/subtaskA_data_all.csv', 'r')
train_answer = open('Training  Data/subtaskA_answers_all.csv', 'r')

test= open('Test Data/subtaskA_test_data.csv', 'r')
test_answer = open('Test Data/subtaskA_gold_answers.csv', 'r')

trial = open('Trial Data/taskA_trial_data.csv', 'r')
trial_answer = open('Trial Data/taskA_trial_answer.csv', 'r')


total = test.readlines()[1:]
total_answer = test_answer.readlines()

print(total)
print(total_answer)

total_file = open('semeval_test_data.csv', 'w')
for t in total:
    total_file.write(t)
total_answer_file = open('semeval_test_answer.csv', 'w')
for t in total_answer:
    total_answer_file.write(t)

