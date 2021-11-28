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


class corpus(T5Dataset):
    def __init__(self, file_path, tokenizer, input_seq_len=128, target_seq_len=128):
        super().__init__(file_path, tokenizer, input_seq_len, target_seq_len)
        self.pre_data()
    """
    "statement": str,
    "positive_explanations": [str],  # can be empty list
    "negative_explanations": [str],  # can be empty list
    "answers": str, # can be None
    "data_resource": str,
    """
    def pre_data(self):
        df = pd.read_json(self.file_path, lines=True)
        for idx, row in df.iterrows():
            statement = row['statement']
            num_explanation = min(len(row['positive_explanations']), len(row['negative_explanations']))
            positive_explanations = random.choices(row['positive_explanations'], k=num_explanation)
            negative_explanations = random.choices(row['negative_explanations'], k=num_explanation)
            explanations = positive_explanations + negative_explanations
            input_text = f"{statement} {row['answers']}. Because "

            self.data += [{'input_text': input_text, 'target_text': explanation} for explanation in explanations]


