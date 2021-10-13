import pandas as pd


def clean_meaningless(words_set):
    meaningless_words = ['am', 'is', 'are', 'be', 'were', 'was', 'will', 'he', 'she', 'it', 'this', 'that'
                         'do', 'does', 'did', 'can', 'would', 'shall', 'those', 'these', 'a', 'an', 's']
    out_set = words_set.copy()
    for word in words_set:
        if word in meaningless_words:
            out_set.remove(word)
    return out_set


def clean_sentence(sent):
    special_char = ['.', ',', ';', '(', ')', '?', ':', '!', '[', ']', '{', '}', '\n']
    for char in special_char:
        try:
            sent = sent.replace(char, '')
        except AttributeError:
            return None
    sent = sent.lower()
    return sent


def find_entity(statement, reason):
    statement = clean_sentence(statement)
    reason = clean_sentence(reason)

    words_statement = set(statement.split()) if statement is not None else set([])
    words_reason = set(reason.split()) if reason is not None else set([])

    words_entity = words_statement.intersection(words_reason)
    words_entity = clean_meaningless(words_entity)
    return words_entity


def sent_entity(df):
    df_entity = []
    for index, row in df.iterrows():
        entities = set([])
        entities = entities.union(find_entity(row['Incorrect Statement'], row['Right Reason1']))
        entities = entities.union(find_entity(row['Incorrect Statement'], row['Right Reason2']))
        entities = entities.union(find_entity(row['Incorrect Statement'], row['Right Reason3']))
        entities = entities.union(find_entity(row['Incorrect Statement'], row['Confusing Reason1']))
        entities = entities.union(find_entity(row['Incorrect Statement'], row['Confusing Reason2']))
        if len(entities) > 0:
            df_entity.append({'sent': row['Incorrect Statement'], 'entity': list(entities)})
    return pd.DataFrame(df_entity)


if __name__ == '__main__':
    df_train = pd.read_csv('../EntangledQA/datasets/semeval_2020_task4/train.csv')
    df_test = pd.read_csv('../EntangledQA/datasets/semeval_2020_task4/test.csv')
    df_dev = pd.read_csv('../EntangledQA/datasets/semeval_2020_task4/dev.csv')

    df_train_entity = sent_entity(df_train)
    df_test_entity = sent_entity(df_test)
    df_dev_entity = sent_entity(df_dev)

    df_train_entity.to_json("./sem-eval_train.json", orient='records')
    df_test_entity.to_json("./sem-eval_test.json", orient='records')
    df_dev_entity.to_json("./sem-eval_dev.json", orient='records')

