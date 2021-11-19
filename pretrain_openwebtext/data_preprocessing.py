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
    pattern_list = [f"because {word} ",
                    f"because the {word} ",
                    f"because a {word} ",
                    f"because an {word} "]
    is_match = False
    for pattern in pattern_list:
        if pattern in sent.lower():
            is_match = True
    return is_match


def select_k_samples(text: list, keywords: dict, k=500):
    sample_sents = random.choices(text, k=k)
    for sent in sample_sents:
        for keyword in keywords.keys():
            if match(sent, keyword):
                keywords[keyword].append(sent)
    return keywords


if __name__ == '__main__':
    keywords = get_key_words_from('./sample.json', 5)
    data_dir = './text/'

    file_name_list = os.listdir(data_dir)
    file_path_list = [os.path.join(data_dir, file_name) for file_name in file_name_list]

    for file_path in tqdm(file_path_list):
        # print(file_path.split('/')[-1])
        with open(file_path, 'r') as fp:
            text = fp.read()
            sent = sent_tokenize(text)  # ["This is a sentence", "This is another sentence", ..., "Last sentence?"]
            for keyword in keywords.keys():
                if match(sent, keyword):
                    keywords[keyword].append(sent)

    with open('./data.json', 'w') as fp:
        json.dump(keywords, fp, indent=4)
        fp.close()
