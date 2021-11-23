"""
Preprocessing for statement and explanation pretraining
"""

import json
import pandas as pd
import torch
from transformers import AutoTokenizer
from torch.utils.data import Dataset


class T5Dataset(Dataset):
    def __init__(self, file_path, tokenizer, max_seq_len=128):
        self.file_path = file_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.max_seq_len = max_seq_len
        self.data = []

    def __getitem__(self, idx):
        record = self.data[idx]
        input_text, output_text = record['input_text'], record['output_text']

        # Tokenize
        input_encoded = self.tokenizer.encode_plus(text=input_text,
                                                   add_special_tokens=False,
                                                   padding='max_length',
                                                   max_length=self.max_seq_len,
                                                   truncation=True,
                                                   return_attention_mask=True)

        target_encoded = self.tokenizer.encode_plus(text=output_text,
                                                    add_special_tokens=False,
                                                    padding='max_length',
                                                    max_length=self.max_seq_len,
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
    def __init__(self, file_path, tokenizer, mode, max_seq_len=128):
        super().__init__(file_path, tokenizer, max_seq_len=128)
        assert mode == 'predict_first' or mode == 'explain_first', "mode should choose from [predict_first, " \
                                                                   "explain_first], but got others "
        self.mode = mode
        self.data_preprocessing()

    def data_preprocessing(self):
        df = pd.read_json(self.file_path, lines=True)
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
