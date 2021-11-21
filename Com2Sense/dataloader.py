"""
Preprocessing for statement and explanation pretraining
"""

import json
import torch
from transformers import AutoTokenizer
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self, text_path, tokenizer, max_seq_len=128):
        self.text_path = text_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.max_seq_len = max_seq_len
        self.data = []

        self.data_process()

    def __getitem__(self, idx):
        record = self.data[idx]
        loc_because = record.lower().find('because ')
        statement, explanation = record[0: loc_because], record[loc_because+8:]

        # Tokenize
        input_encoded = self.tokenizer.encode_plus(text=statement+", because",
                                                   add_special_tokens=False,
                                                   padding='max_length',
                                                   max_length=self.max_seq_len,
                                                   truncation=True,
                                                   return_attention_mask=True)

        target_encoded = self.tokenizer.encode_plus(text=explanation,
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

    def data_process(self):
        with open(self.text_path, "r") as fp:
            data = json.load(fp)
            self.data.append(data)
            fp.close()
