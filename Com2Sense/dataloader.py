"""
Preprocessing for statement and explanation pretraining
"""

import json
import torch
from transformers import AutoTokenizer
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self, text_path, data_name, tokenizer, max_seq_len=128):
        self.text_path = text_path
        self.data_name = data_name
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.max_seq_len = max_seq_len
        self.data = []

        self.data_process()

    def __getitem__(self, idx):
        record = self.data[idx]
        statement, explanation = record['statement'], record['explanation']

        # Tokenize
        input_encoded = self.tokenizer.encode_plus(text=statement + ", because",
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
        if self.data_name == 'openwebtext':
            data = []
            with open(self.text_path, "r") as fp:
                text = json.load(fp)
                for sentence in text:
                    sentence = sentence.lower()
                    loc_because = sentence.find('because ')
                    statement = sentence[0: loc_because]
                    explanation = sentence[loc_because+8:]
                    self.data.append({'statement': statement,
                                      'explanation': explanation})
                fp.close()

        elif self.data_name == 'sem-eval':
            pass
