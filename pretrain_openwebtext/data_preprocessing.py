import os
import json
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
import random


def get_key_words_from(data_path, topk=5):
    with open(data_path, 'r') as fp:
        keywords_dict = json.load(fp)
        fp.close()
    keywords = {}
    for keyword in keywords_dict.keys():
        if keywords_dict[keyword] > topk:
            keywords[keyword] = []
    return keywords


def match(sent, word):
    sent = sent.lower()
    word = word.lower()

    if 'because' in sent:
        sent = sent.split('becuase')
        statement = sent[0]
        explanation = "".join(sent[1:])
        if word in statement and word in explanation:
            return {'statement': statement, 'explanation': explanation}
        else:
            return None
    else:
        return None


def select_k_samples(text: list, keywords: dict, k=500):
    sample_sents = random.choices(text, k=k)
    for sent in sample_sents:
        for keyword in keywords.keys():
            if match(sent, keyword):
                keywords[keyword].append(sent)
    return keywords


if __name__ == '__main__':
    keywords = get_key_words_from('/nas/home/yixiaoli/datasets/sample.json', 5)
    data_dir = '/nas/home/yixiaoli/datasets/test/'

    file_name_list = os.listdir(data_dir)
    file_path_list = [os.path.join(data_dir, file_name) for file_name in file_name_list]

    for file_path in tqdm(file_path_list):
        # print(file_path.split('/')[-1])
        with open(file_path, 'r') as fp:
            text = fp.read()
            sent_list = sent_tokenize(text)  # ["This is a sentence", "This is another sentence", ..., "Last sentence?"]
            for keyword in keywords.keys():
                for sent in sent_list:
                    state_exp = match(sent, keyword)
                    if state_exp:
                        keywords[keyword].append(state_exp)

    with open('./data.json', 'w') as fp:
        json.dump(keywords, fp, indent=4)
        fp.close()
