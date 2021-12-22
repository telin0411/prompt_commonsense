"""
Preprocessing for statement and explanation pretraining
"""

import pandas as pd
import torch
from transformers import AutoTokenizer
from torch.utils.data import Dataset
import random


class T2TDataset(Dataset):
    def __init__(self, file_path, tokenizer, input_seq_len=128, target_seq_len=2):
        self.file_path = file_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.input_seq_len = input_seq_len
        self.target_seq_len = target_seq_len
        self.data = None

    def __getitem__(self, idx):
        record = self.data[idx]
        input_text, output_text = record['input_text'], record['output_text']

        # Tokenize
        input_encoded = self.tokenizer.encode_plus(text=input_text,
                                                   add_special_tokens=False,
                                                   padding='max_length',
                                                   max_length=self.input_seq_len,
                                                   truncation=True,
                                                   return_attention_mask=True)

        target_encoded = self.tokenizer.encode_plus(text=output_text,
                                                    add_special_tokens=False,
                                                    padding='max_length',
                                                    max_length=self.target_seq_len,
                                                    truncation=True,
                                                    return_attention_mask=True)

        input_token_ids = torch.tensor(input_encoded['input_ids'])
        input_attn_mask = torch.tensor(input_encoded['attention_mask'])

        target_token_ids = torch.tensor(target_encoded['input_ids'])
        target_attn_mask = torch.tensor(target_encoded['attention_mask'])

        # Output
        sample = {'input_token_ids': input_token_ids,
                  'input_attn_mask': input_attn_mask,
                  'target_token_ids': target_token_ids,
                  'target_attn_mask': target_attn_mask}

        return sample

    def __len__(self):
        return len(self.data)

    def get_tokenizer(self):
        return self.tokenizer


class COSE(T2TDataset):
    def __init__(self, file_path, tokenizer, mode, input_seq_len):
        super().__init__(file_path, tokenizer, input_seq_len, target_seq_len=20)
        assert mode == 'predict_first' or mode == 'explain_first', "mode should choose from [predict_first, " \
                                                                   "explain_first], but got others "
        self.mode = mode
        self.data_preprocessing()

    def data_preprocessing(self):
        df = pd.read_json(self.file_path)
        for idx, row in df.iterrows():
            question = row['question']
            choice0 = row['choices']['A']
            choice1 = row['choices']['B']
            choice2 = row['choices']['C']
            choice3 = row['choices']['D']
            choice4 = row['choices']['E']
            answer = row['answer']
            explanation = row['explanation']
            if self.mode == 'explain_first':
                text = f'{question} The choices are {choice0}, {choice1}, {choice2}, {choice3}, or {choice4}. My ' \
                       f'commonsense tells me '
            else:
                text = f'{question} The choices are {choice0}, {choice1}, {choice2}, {choice3}, or {choice4}. The ' \
                       f'answer is {answer} because '

            self.data.append({'input_text': text,
                              'output_text': explanation})


class CSQA(T2TDataset):
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

    def __init__(self, file_path, tokenizer, input_seq_len, has_explanation=False):
        super().__init__(file_path, tokenizer, input_seq_len, target_seq_len=8)

        self.has_explanation = has_explanation
        self.data_preprocessing()

    def data_preprocessing(self):
        df = pd.read_json(self.file_path, lines=True)
        str2int = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
        for idx, row in df.iterrows():
            question = row['question']['stem']
            choices = row['question']['choices']
            answer = row['question']['choices'][str2int[row['answerKey']]]['text']
            choices = [f"({choice['label']}) {choice['text']} " for choice in choices]
            choices = "".join(choices)

            if self.has_explanation:
                explanation = row['explanation']
                input_text = f"{question}\n{explanation}\n{choices}"
            else:
                input_text = f"{question}\n{choices}"

            self.data.append({'input_text': input_text,
                              'output_text': answer})


class COSE_T5_gen(T2TDataset):
    """
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
        "explanation_gt": "gnerghi",
        "explanation_gen": "gnerghfei",
        "answer": "gnerghi"
        }
    """

    def __init__(self, file_path, tokenizer, input_seq_len, has_explanation=False):
        super().__init__(file_path, tokenizer, input_seq_len, target_seq_len=8)
        self.has_explanation = has_explanation
        self.data_preprocessing()

    def data_preprocessing(self):
        df = pd.read_json(self.file_path)
        for idx, row in df.iterrows():
            question = row['question']
            choices = [f"({k}){v}" for k, v in row['choices'].items()]
            choices = "".join(choices)
            answer = row['answer']
            explanation = row['explanation_gen']
            if self.has_explanation:
                text = f'{explanation}\n{question}\n{choices}'
            else:
                text = f'{question}\n{choices}'

            self.data.append({'input_text': text,
                              'output_text': answer})


class Com2Sense(T2TDataset):
    def __init__(self, file_path, tokenizer, input_seq_len):
        super().__init__(file_path, tokenizer, input_seq_len, target_seq_len=20)

        self.data_preprocessing()

    def data_preprocessing(self):
        df = pd.read_json(self.file_path)
        for idx, row in df.iterrows():
            sent_1 = row['sent_1']
            sent_2 = row['sent_2']
            self.data.append({'input_text': f"{sent_1} My common sense tells me",
                              'output_text': ""})
            self.data.append({'input_text': f"{sent_2} My common sense tells me",
                              'output_text': ""})


class BartTransfer(T2TDataset):
    def __init__(self, file_path, tokenizer, input_seq_len, is_test=False):
        super().__init__(file_path, tokenizer, input_seq_len, target_seq_len=20)
        self.data = []
        if is_test:
            self.data_preprocessing_ecqa_test()
        elif 'ecqa' in file_path:
            self.data_preprocessing_ecqa()
        elif 'com2sense' in file_path:
            self.data_preprocessing_com2sense()

    def data_preprocessing_com2sense(self):
        df = pd.read_json(self.file_path)
        for idx, row in df.iterrows():
            sent_1, sent_2 = row['sent_1'], row['sent_2']
            label_1, label_2 = row['label_1'], row['label_2']

            self.data.append({'input_text': f"{sent_1} It is a {label_1} statement because",
                              'output_text': ""})
            self.data.append({'input_text': f"{sent_2} It is a {label_2} statement because",
                              'output_text': ""})

    def data_preprocessing_ecqa(self):
        df = pd.read_json(self.file_path)
        for idx, row in df.iterrows():
            question = row['question']
            choices = row['choices']
            explanations = row['explanations']

            for key in explanations.keys():
                if key == row['answer_key']:
                    text = f'{question} {choices[key]}. It is a true answer because'
                    expl = f'{explanations[key]}'
                    for _ in range(len(explanations) - 1):
                        self.data.append({'input_text': text,
                                          'output_text': expl})
                else:
                    text = f'{question} {choices[key]}. It is a false answer because'
                    expl = f'{explanations[key]}'
                    self.data.append({'input_text': text,
                                      'output_text': expl})

    def data_preprocessing_ecqa_test(self):
        df = pd.read_json(self.file_path)
        for idx, row in df.iterrows():
            question = row['question']
            choices = row['choices']
            explanations = row['explanations']

            for key in explanations.keys():
                if key == row['answer_key']:
                    text = f'{question} {choices[key]}. It is a true answer because'
                else:
                    text = f'{question} {choices[key]}. It is a false answer because'

                expl = f'{explanations[key]}'
                self.data.append({'input_text': text,
                                  'output_text': expl})


class GANPair(Dataset):
    def __init__(self, real_file, fake_file, tokenizer, input_seq_len, target_seq_len=32):
        self.real_file = real_file
        self.fake_file = fake_file
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.input_seq_len = input_seq_len
        self.target_seq_len = target_seq_len
        self.real_data = None
        self.fake_data = None
        self.pre_data()

    def __len__(self):
        return max(len(self.real_data), len(self.fake_data))

    def __getitem__(self, idx):
        real_record = self.real_data[idx]
        fake_record = self.fake_data[idx]
        print("read record and fake record", idx)
        # Tokenize
        real_token_input = self.tokenize(real_record['input_text'], self.input_seq_len)
        real_token_target = self.tokenize(real_record['output_text'], self.target_seq_len)

        fake_token_input = self.tokenize(fake_record['input_text'], self.input_seq_len)
        fake_token_target = self.tokenize(fake_record['output_text'], self.target_seq_len)

        # Output
        sample = {'real': {'input_token_ids': real_token_input['input_ids'],
                           'input_attn_mask': real_token_input['attention_mask'],
                           'target_token_ids': real_token_target['input_ids'],
                           'target_attn_mask': real_token_target['attention_mask']},
                  'fake': {'input_token_ids': fake_token_input['input_ids'],
                           'input_attn_mask': fake_token_input['attention_mask'],
                           'target_token_ids': fake_token_target['input_ids'],
                           'target_attn_mask': fake_token_target['attention_mask']}
                  }

        return sample

    def get_tokenizer(self):
        return self.tokenizer

    def tokenize(self, text, length):
        token_dict = self.tokenizer.encode_plus(text=text,
                                                add_special_tokens=False,
                                                padding='max_length',
                                                max_length=length,
                                                truncation=True,
                                                return_attention_mask=True)
        return token_dict

    def pre_data(self):
        self.real_data = []
        self.fake_data = []

        # load real data, i.e., ecqa
        df = pd.read_json(self.real_file)
        for idx, row in df.iterrows():
            question = row['question']
            choices = row['choices']
            explanations = row['explanations']

            for key in explanations.keys():
                if key == row['answer_key']:
                    text = f'{question} {choices[key]}. It is a true answer because'
                else:
                    text = f'{question} {choices[key]}. It is a false answer because'

                expl = f'{explanations[key]}'
                self.real_data.append({'input_text': text,
                                       'output_text': expl})
        # load fake data, i.e., com2sense
        df = pd.read_json(self.fake_file)
        for idx, row in df.iterrows():
            sent_1, sent_2 = row['sent_1'], row['sent_2']
            label_1, label_2 = row['label_1'], row['label_2']

            self.fake_data.append({'input_text': f"{sent_1} It is a {label_1} statement because",
                                   'output_text': ""})
            self.fake_data.append({'input_text': f"{sent_2} It is a {label_2} statement because",
                                   'output_text': ""})

        # align length
        size = max(len(self.real_data), len(self.fake_data))
        real_data_margin = size - len(self.real_data)
        fake_data_margin = size - len(self.fake_data)

        self.real_data += random.choices(self.real_data, k=real_data_margin)
        self.fake_data += random.choices(self.fake_data, k=fake_data_margin)

        # shuffle pair
        random.shuffle(self.real_data)
        random.shuffle(self.fake_data)

