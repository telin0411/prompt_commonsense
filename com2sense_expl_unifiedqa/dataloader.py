"""
Preprocessing for statement and explanation pretraining
"""

import json
import pandas as pd
import torch
from transformers import AutoTokenizer
from torch.utils.data import Dataset
import random


class T5Dataset(Dataset):
    def __init__(self, file_path, tokenizer, input_seq_len=128, target_seq_len=128):
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


class Com2Sense(T5Dataset):
    def __init__(self, file_path, tokenizer, has_explanation=False, input_seq_len=128, target_seq_len=2):
        super().__init__(file_path, tokenizer, input_seq_len, target_seq_len)
        self.has_explanation = has_explanation
        self.pre_data()

    def pre_data(self):
        df = pd.read_json(self.file_path)
        int2str = {0: 'no', 1: 'yes'}
        for idx, row in df.iterrows():
            sent_1 = row['sent_1']
            sent_2 = row['sent_2']
            label_1 = int2str[row['label_1']]
            label_2 = int2str[row['label_2']]
            if self.has_explanation:
                expl_1 = "Because " + row['expl_1']
                expl_2 = "Because " + row['expl_2']
            else:
                expl_1 = ""
                expl_2 = ""
            self.data.append({'input_text': f"{expl_1} Is the following statement correct?\n{sent_1}",
                              'output_text': label_1})
            self.data.append({'input_text': f"{expl_2} Is the following statement correct?\n{sent_2}",
                              'output_text': label_2})
