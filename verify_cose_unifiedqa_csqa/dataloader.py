"""
Preprocessing for statement and explanation pretraining
"""

import json
import pandas as pd
import torch
from transformers import AutoTokenizer
from torch.utils.data import Dataset


class T5Dataset(Dataset):
    def __init__(self, file_path, tokenizer, input_seq_len=128, target_seq_len=20):
        self.file_path = file_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.input_seq_len = input_seq_len
        self.target_seq_len = target_seq_len
        self.data = []

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


class COSE(T5Dataset):
    def __init__(self, file_path, tokenizer, mode):
        super().__init__(file_path, tokenizer)
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


class CSQA(T5Dataset):
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

    def __init__(self, file_path, tokenizer, has_explanation=False):
        super().__init__(file_path, tokenizer)

        self.has_explanation = has_explanation
        self.data_preprocessing()

    def data_preprocessing(self):
        df = pd.read_json(self.file_path)
        str2int = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
        for idx, row in df.iterrows():
            question = row['question']['stem']
            choices = row['choices']['choices']
            answer = row['question']['choices'][str2int[row['answerKey']]]['text']
            choices = [f"({choice['label']}) {choice['text']} " for choice in choices]

            if self.has_explanation:
                explanation = row['explanation']
                input_text = f"{question}\n{explanation}\n{choices}"
            else:
                input_text = f"{question}\n{choices}"

            self.data.append({'input_text': input_text,
                              'output_text': answer})
