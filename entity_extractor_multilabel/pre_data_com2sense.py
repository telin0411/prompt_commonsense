import re
import allennlp_models.tagging
import pandas as pd
from allennlp.predictors.predictor import Predictor


def entity_is_pron(entity):
    pron = ['I', 'i', 'he', 'she', 'it', 'they',
            'me', 'him', 'her', 'it', 'them',
            'mine', 'his', 'her', 'its', 'there',
            'one', 'this', 'that', 'those', 'these',
            'something', 'anything']
    return entity in pron


def match_bracket(start, string):
    # initialize the index of string
    # and the number of left bracket
    # We assume the left of string[start] is the first '('
    index = start
    number = 1

    while (index < len(string) and number != 0):
        if string[index] == '(':
            number += 1
        elif string[index] == ')':
            number -= 1
        index += 1

    return index


def remove_dirt(string, types):
    string = string.replace(')', '')
    string = string.replace('(', '')
    string = string.split(' ')
    entity = []
    for word in string:
        if word not in types:
            entity.append(word)
    return ' '.join(entity)


def extract_entities(sentence):
    if 'instead of' not in sentence and 'rather than' not in sentence:
        return '', ''
    else:
        # step 0, parse the sentence
        out = predictor.predict(sentence=sentence)
        trees = out['trees']
        types = out['hierplane_tree']['nodeTypeToStyle']

        # step 1, find the 'rather than' or 'instead of' phrase
        rt_phrase = re.search(r'\([A-Z]+ instead\) \([A-Z]+ of\)', trees)
        if rt_phrase is None:
            rt_phrase = re.search(r'\([A-Z]+ rather\) \([A-Z]+ than\)', trees)
            if rt_phrase is None:
                return '', ''
        rt_phrase = rt_phrase.group()

        # step 2, split the tree by 'rather than' phrase
        left = trees.split(rt_phrase)[0]
        right = trees.split(rt_phrase)[1]

        # step 3,
        # find the left most NP in the right split
        # and the right most NP in the left split
        r_NP_start = right.find('NP')
        l_NP_start = [m.start() for m in re.finditer('NP', left)][-1]

        # step 4, match the bracket and get NP entities

        r_NP_end = match_bracket(r_NP_start, right)
        l_NP_end = match_bracket(l_NP_start, left)

        entity_l = left[l_NP_start:l_NP_end]
        entity_r = right[r_NP_start:r_NP_end]

        # step 5, remove '(', ')', and pos types like 'NP', 'PP'
        entity_l = remove_dirt(entity_l, types)
        entity_r = remove_dirt(entity_r, types)

        if entity_is_pron(entity_l) or entity_is_pron(entity_l):
            return '', ''
        else:
            return entity_l, entity_r


def isVicinity(words, sent):
    words = ' '.join(words)
    if words in sent:
        return True
    else:
        return False


def isInBlackList(words, blackList):
    is_in_black_list = False
    for word in words:
        if word in blackList:
            is_in_black_list = True
    return is_in_black_list


def isOneWordDiff(sent_1, sent_2):
    blackList = ['can', 'cannot', 'can\'t', 'do', 'not', 'don\'t', 'will', 'won\'t', 'less', 'more']
    rest_1 = list(set(sent_1.split()) - set(sent_2.split()))
    rest_2 = list(set(sent_2.split()) - set(sent_1.split()))

    is_qualified = len(rest_1) == 1 and len(rest_2) == 1

    is_qualified = is_qualified and not isInBlackList(rest_1, blackList)
    is_qualified = is_qualified and not isInBlackList(rest_2, blackList)

    if is_qualified:
        return True, ' '.join(rest_1), ' '.join(rest_2)

    else:
        return False, None, None


if __name__ == '__main__':
    predictor = Predictor.from_path(
        "https://storage.googleapis.com/allennlp-public-models/elmo-constituency-parser-2020.02.10.tar.gz")
        #cuda_device=0)

    df_dev = pd.read_json("../Com2Sense/datasets/com2sense/dev.json")
    df_train = pd.read_json("../Com2Sense/datasets/com2sense/train.json")
    df_test = pd.read_json("../Com2Sense/datasets/com2sense/test.json")

    df = pd.concat([df_dev, df_train, df_test])
    df_entity = []

    print(len(df_dev), len(df_train), len(df_test), len(df))

    for index, row in df.iterrows():
        if row['scenario'] == 'comparison':
            # get the entities from the comparative sentences
            entities_1 = extract_entities(row['sent_1'])
            entities_2 = extract_entities(row['sent_2'])

            # if the entities if valid (not a pronoun or smth), we move it to the front
            if '' not in entities_1 and '' not in entities_2:
                df_entity.append({'sent': row['sent_1'],
                                  'entity': [entities_1[0].split()[-1], entities_1[1].split()[-1]]})
                df_entity.append({'sent': row['sent_2'],
                                  'entity': [entities_2[0].split()[-1], entities_2[1].split()[-1]]})

        elif row['scenario'] == 'causal':
            is_one_word_diff, word_1, word_2 = isOneWordDiff(row['sent_1'], row['sent_2'])

            if is_one_word_diff:
                df_entity.append({'sent': row['sent_1'], 'entity': [word_1]})
                df_entity.append({'sent': row['sent_2'], 'entity': [word_2]})

    df_entity = pd.DataFrame(df_entity)
    df_entity.to_json("./datasets/com2sense/com2sense.json", orient='records')
