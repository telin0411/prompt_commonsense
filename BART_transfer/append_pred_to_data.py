import pandas as pd
from tqdm import tqdm


def append_to_data(raw_path, prediction_path, save_path):
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
    for index, row_raw in df_raw.iterrows():
        row_pre = df_pre.iloc[index]
        row_pre = row_pre.to_dict()
        row_raw = row_raw.to_dict()
        row_raw.update({"explanation_gen": row_pre['prediction']})
        data.append(row_raw)
    data = pd.DataFrame(data)
    data.to_json(save_path, orient='records', indent=4)


def ecqa2schema(ecqa_file_path, csqa_file_path, save_file_path):
    """
    :param ecqa_file_path: the/path/to/ecqa
    :param csqa_file_path: the/path/to/csqa
    :param save_file_path: the/path/to/save
    :return:
    """
    """ecqa
    {"id": "e1f9e9a9768c9404e1d87cbeb46dbe46", 
    "positives": ["engage is a type of fight."], 
    "negatives": ["arrogate is not a type of fight.", 
                  "retain is not a type of fight.", 
                  "smile is not a type of fight.", 
                  "embrace is not a type of fight."], 
    "explanation": "Bill engaged in a fight with enemy. Other options are not a type of fights one takes with enemy."}
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
            "explanation": {
                        "A": "gery",
                        "B": "frg",
                        "C": "ytf",
                        "D": "fw",
                        "E": "hrt"
            }
            "answer": "gery"
            "answer_key": "A"
            }
        """
    df_csqa = pd.read_json(csqa_file_path, lines=True)
    df_ecqa = pd.read_json(ecqa_file_path, lines=True)
    print("Loaded csqa: ", csqa_file_path)
    print("Loaded ecqa: ", ecqa_file_path)
    data = {}
    str2int = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
    for idx_ecqa, row_ecqa in df_ecqa.iterrows():
        data.update({row_ecqa['id']: {}})

    for idx_csqa, row_csqa in df_csqa.iterrows():
        question = row_csqa['question']['stem']
        choices = {k: row_csqa['question']['choices'][str2int[k]]['text'] for k in ['A', 'B', 'C', 'D', 'E']}
        answer_key = row_csqa['answerKey']
        answer = row_csqa['question']['choices'][str2int[row_csqa['answerKey']]]['text']
        data[row_csqa['id']] = {'question': question,
                              'answer': answer,
                              'answer_key': answer_key,
                              'choices': choices}

    for idx_ecqa, row_ecqa in df_ecqa.iterrows():
        if data[row_ecqa['id']] == {}:
            continue
        answer_key = data[row_ecqa['id']]['answer_key']
        choices = data[row_ecqa['id']]['choices']

        explanations = {answer_key: row_ecqa['positives'][0]}
        for neg_expl in row_ecqa['negatives']:
            for choice_key in choices.keys():
                if choices[choice_key].lower() in neg_expl.lower():
                    explanations.update({choice_key: neg_expl})

        if len(explanations) == 4:
            continue

        data[row_ecqa['id']].update({'explanations': explanations})




    ecqa_processed = pd.DataFrame(data)
    ecqa_processed.to_json(save_file_path, orient='records', indent=4)
    print("Saved combined file at ", save_file_path)


if __name__ == '__main__':
    ecqa2schema("", "", "")
