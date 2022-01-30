"""
Preprocessing for statement and explanation pretraining
"""

import pandas as pd
import torch
from transformers import AutoTokenizer
from torch.utils.data import Dataset


class T5Dataset(Dataset):
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
                  'target_attn_mask': target_attn_mask,
                  'target_text': output_text}

        return sample

    def __len__(self):
        return len(self.data)

    def get_tokenizer(self):
        return self.tokenizer


class ECQA(T5Dataset):
    def __init__(self, file_path, tokenizer, input_seq_len):
        super().__init__(file_path, tokenizer, input_seq_len, target_seq_len=24)
        self.file_path = file_path
        self.processing_data()

    def processing_data(self):
        self.data = []
        df = pd.read_json(self.file_path)
        for index, row in df.iterrows():
            input_text = f"{row['statement']}"
            if 'ecqa' in self.file_path:
                expl_part = "".join(row['explanation'].split()[0:4])
                expl_rest = "".join(row['explanation'].split()[4:])
                output_text = f"{row['label']}, because {expl_part} "
            elif 'com2sense' in self.file_path:
                expl_part = ""
                expl_rest = ""
                output_text = f"{row['label']}, because "
            else:
                raise NameError("data must from either ecqa or com2sense")
            self.data.append({'input_text': input_text, 'output_text': output_text})
