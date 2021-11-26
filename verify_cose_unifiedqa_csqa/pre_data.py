import pandas as pd
from tqdm import tqdm


def format_t5_generated_explanation(raw_path, prediction_path, save_path):
    """raw
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
    """pred
    {
        "input": "fefwfw"
        "prediction": "fewfwegt"
        "ground_truth": "fq"
    }
        
    """

    df_raw = pd.read_json(raw_path)
    df_pre = pd.read_csv(prediction_path)
    data = []
    for _, row_raw, _, row_pre in zip(df_raw.iterrows(), df_pre.iterrows()):
        row_raw = row_raw.update({"explanation": row_pre['prediction']})
        data.append(row_raw)
    data = pd.DataFrame(data)
    data.to_json(save_path, orient='records', indent=4)


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
    for idx_csqa, row_csqa in tqdm(df_csqa.iterrows()):
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
    cose_for_generation.to_json(save_path, orient='records', indent=4)
    print("Saved combined file at ", save_path)


if __name__ == '__main__':
    format_t5_generated_explanation(raw_path='',
                                    prediction_path='',
                                    save_path='')
