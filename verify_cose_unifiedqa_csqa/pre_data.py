import pandas as pd
from tqdm import tqdm


def amazonReviewsToCorpus(file_path_list, save_path):
    """
{
        "statement": str,
        "positive_explanations": [str],  # can be empty list
        "negative_explanations": [str],  # can be empty list
        "answers": str, # can be None
        "data_resource": str,
}
    """
    data = []
    for file_path in file_path_list:
        df = pd.read_csv(file_path, sep='\t')
        for idx, row in df.iterrows():
            sent = row['review_body'].lower()
            if 'because of ' in sent:
                statement = sent.split('because of ')[0]
                positive_explanation = "".join(sent.split('because of ')[1:])
            elif 'because ' in sent:
                statement = sent.split('because ')[0]
                positive_explanation = "".join(sent.split('because ')[1:])
            else:
                continue
            data.append({'statement': statement,
                         'positive_explanations': [positive_explanation],
                         'answers': "",
                         "data_resource": file_path})
    data = pd.DataFrame(data)
    data.to_json(save_path, lines=True)


def entailmentToCorpus(file_path, save_path):
    """
    {
    "id":"Mercury_SC_LBS10351",
    "context":"sent1: earth is a kind of celestial object sent2: stars appear to move relative to the horizon during the night sent3: a star is a kind of celestial object /      celestial body sent4: the earth rotating on its axis causes stars to appear to move across the sky at night sent5: apparent motion is when an object appears to move relative to another object 's pos     ition",
    "question":"How does the appearance of a constellation change during the night?",
    "answer":"Its position appears to shift relative to the horizon.",
    "hypothesis":"the earth rotating on its axis causes stars to move relative to the horizon during the night",
    "proof":"sent1 & sent3 & sent5 -> int1: apparent motion of stars is when stars appear to move relative to earth's position; int1      & sent4 -> int2: the earth rotating on its axis causes apparent motion of stars; int2 & sent2 -> hypothesis; ",
    "full_text_proof":" [BECAUSE] earth is a kind of celestial object [AND] a star is a k     ind of celestial object / celestial body [AND] apparent motion is when an object appears to move relative to another object 's position [INFER] int1: apparent motion of stars is when stars appear to      move relative to earth's position [BECAUSE] int1 [AND] the earth rotating on its axis causes stars to appear to move across the sky at night [INFER] int2: the earth rotating on its axis causes appare     nt motion of stars [BECAUSE] int2 [AND] stars appear to move relative to the horizon during the night [INFER] int3: the earth rotating on its axis causes stars to move relative to the horizon during      the night",
    "depth_of_proof":3,
    "length_of_proof":3,
    "meta":{
        "question_text":"How does the appearance of a constellation change during the night?",
        "answer_text":"Its position appears to shift relative to the horizon.",
        "hypothesis_id":"int3",
        "triples":{
            "sent1":"earth is a kind of celestial object",
            "sent2":"stars appear to move relative to the horizon during the night",
            "sent3":"a star is a kind of celestial object celestial body",
            "sent4":"the earth rotating on its axis causes stars to appear to move across the sky at night",
            "sent5":"apparent motion is when an object appears to move relative to another object 's position"
        },
        "distractors":[

        ],
        "distractors_relevance":[

        ],
        "intermediate_conclusions":{
            "int1":"apparent motion of stars is when stars appear to move relative to earth's position",
            "int2":"the earth rotating on its axis causes apparent motion of stars",
            "int3":"the earth rotating on its axis causes stars to move relative to the horizon during the night"
        },
        "core_concepts":[
            "stars appear to move relative to the horizon during the night"
        ],
        "step_proof":"sent1 & sent3 & sent5 -> int1: apparent motion of stars is when stars appear to move relative to earth's position; int1 & sent4 -> int2: the earth rotating on its axis causes apparent motion of stars; int2 & sent2 -> hypothesis; ",
        "lisp_proof":"((((((sent1 sent3 sent5) -> int1) sent4) -> int2) sent2) -> int3)",
        "polish_proof":"# int3 & # int2 & # int1 & sent1 & sent3 sent5 sent4 sent2",
        "worldtree_provenance":{
            "sent1":{
                "uuid":"dbe8-e776-f804-99a0",
                "original_text":"a star is a kind of celestial object celestial body"
            },
            "sent2":{
                "uuid":"",
                "original_text":"earth is a kind of celestial object"
            },
            "sent3":{
                "uuid":"49f5-727d-92b5-03aa",
                "original_text":"the earth rotating on its axis causes stars / the moon to appear to move across the sky at night"
            },
            "sent4":{
                "uuid":"1d13-2365-305d-1e1b",
                "original_text":"apparent motion is when an object appears to move relative to another object 's perspective / another object 's position"
            },
            "sent5":{
                "uuid":"fc96-f0ff-1c18-337e",
                "original_text":"stars appear to move relative to the horizon during the night"
            }
        },
        "add_list":[
            {
                "sid":"sent2",
                "fact":"earth is a kind of celestial object"
            }
        ],
        "delete_list":[
            {
                "uuid":"db62-4c2e-6a9d-8b12",
                "fact":"a constellation contains stars"
            },
            {
                "uuid":"5304-a10e-5986-7f63",
                "fact":"if something moves then that something is in a different location"
            },
            {
                "uuid":"f0ff-da5a-7b9d-614d",
                "fact":"appearance is a property of objects / materials"
            },

        ]
    }
}
    """
    """
    {
        "statement": str,
        "positive_explanations": [str],  # can be empty list
        "negative_explanations": [str],  # can be empty list
        "answers": str, # can be None
        "data_resource": str,
}
    """
    df = pd.read_json(file_path)
    corpus = []
    for idx, row in df.iterrows():
        statement = row['question']
        positive_explanations = [v for k, v in row['meta']['triples'].items()]
        positive_explanations += [v for k, v in row['meta']['intermediate_conclusions'].items()]
        negative_explanations = [sample['fact'] for sample in row['meta']['negative_explanations']]
        answers = row['answer']
        data_resource = file_path
        corpus.append({'statement': statement,
                       'positive_explanations': positive_explanations,
                       'negative_explanations': negative_explanations,
                       'answers': answers,
                       'data_resource': data_resource})
    corpus = pd.DataFrame(corpus)
    corpus.to_json(save_path, lines=True, indent=4)


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
    for index, row_raw in df_raw.iterrows():
        row_pre_1 = df_pre.iloc[2*index]
        row_pre_2 = df_pre.iloc[2*index+1]

        row_pre_1 = row_pre_1.to_dict()
        row_pre_2 = row_pre_2.to_dict()
        row_raw = row_raw.to_dict()

        row_raw.update({"expl_1": row_pre_1['prediction']})
        row_raw.update({"expl_2": row_pre_2['prediction']})

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
    format_t5_generated_explanation(raw_path='/local1/telinwu/yixiao/datasets/com2sense/dev.json',
                                    prediction_path='../T0pp_CSQA/dev_com2sense_pred.csv',
                                    save_path='/local1/telinwu/yixiao/datasets/com2sense/dev_expl.json')
