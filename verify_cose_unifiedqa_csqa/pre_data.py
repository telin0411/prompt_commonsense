import pandas as pd


def combine_CSQA_and_COSE(file_csqa, file_cose, save_path):
    """
    :param save_path: the path to save the combined data
    :param file_csqa: file path for csqa
    :param file_cose: file path for cose
    :return: Void
    """
    """ cose
    {
    "id": "3a1b3c21a11f4ec53c166b0559df7369", 
    "explanation": {
                    "open-ended": "bums are well known to take up residence under bridges.", 
                    "selected": "a bum"
                    }
    }
    """
    """ csqa
        {
        "answerKey": "B", 
        "id": "3d0f8824ea83ddcc9ab03055658b89d3", 
        "question": {
                     "question_concept": "mold", 
                     "choices": [{"label": "A", "text": "carpet"}, 
                                 {"label": "B", "text": "refrigerator"}, 
                                 {"label": "C", "text": "breadbox"}, 
                                 {"label": "D", "text": "fridge"}, 
                                 {"label": "E", "text": "coach"}], 
                     "stem": "The forgotten leftovers had gotten quite old, he found it covered in mold in the back of his what?"
                     }
        }
    """
    """ target
        {
        "id": "3d0f8824ea83ddcc9ab03055658b89d3"
        "question": "fefefaefgg",
        "choices":{
                  "A": "gery",
                  "B": "frg",
                  "C": "ytf",
                  "D": "fw",
                  "E": "hrt"
                  },
        "explanation": "gnerghi",
        "answer": "gnerghi"
        }
    """
    df_csqa = pd.read_json(file_csqa, lines=True)
    df_cose = pd.read_json(file_cose, lines=True)
    print("Loaded csqa: ", file_csqa)
    print("Loaded cose: ", file_cose)
    data = []
    str2int = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
    for idx_csqa, row_csqa in df_csqa.iterrows():
        for idx_cose, row_cose in df_cose.iterrows():
            if row_cose['id'] == row_csqa['id']:
                question = row_csqa['question']['stem']
                choices = {k: row_csqa['question']['choices'][str2int[k]]['text'] for k in ['A', 'B', 'C', 'D', 'E']}
                explanation = row_cose['explanation']['open-ended']
                answer = row_csqa['question']['choices'][str2int[row_csqa['answerKey']]]['text']
                data.append({'id': row_csqa['id'],
                             'question': question,
                             'choices': choices,
                             'explanation': explanation,
                             'answer': answer})

    cose_for_generation = pd.DataFrame(data)
    cose_for_generation.to_json(save_path, orient='record', indent=4)
    print("Saved combined file at ", save_path)


if __name__ == '__main__':
    combine_CSQA_and_COSE(file_csqa='/data1/yixiao/datasets/csqa/train_rand_split.jsonl',
                          file_cose='/data1/yixiao/datasets/cose/v1.11/cose_train_v1.11_processed.jsonl',
                          save_path='/data1/yixiao/datasets/cose_for_generation/train.json')