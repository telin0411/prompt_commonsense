import os
import json
import torch
from torch.nn import functional
import random
import pandas as pd
from transformers import AutoTokenizer
from torch.utils.data import Dataset, ConcatDataset


class ExDataset(Dataset):
    def __init__(self, data_path, split, tokenizer, max_seq_len=128):
        """
        Processes raw dataset

        :param str data_path: path of a data file, e.g., ./datasets/com2sense/train.json
        :param str tokenizer: tokenizer name (e.g. 'roberta-base', 't5-3b', etc.)
        :param int max_seq_len: tokenized sequence length (padded)
        """
        self.data_path = data_path
        self.split = split
        self.max_seq_len = max_seq_len
        self.tok_name = tokenizer

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.tok_name)

        # Process dataset (in subclass)
        self.data = None
        self.get_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        record = self.data[idx]

        text, label = record['text'], record['label']

        text = f"{text}"

        text_tokens = self.tokenizer(text=text,
                                     padding='max_length',
                                     max_length=self.max_seq_len,
                                     truncation=True,
                                     add_special_tokens=False,
                                     return_attention_mask=True)
        label_tokens_ids = self.tokenizer(label, add_special_tokens=False)

        target = torch.zeros(self.max_seq_len)
        print(text_tokens, label_tokens_ids)
        for token_id in label_tokens_ids:
            target[text_tokens == token_id] = 1

        # Output
        sample = {'input_ids': torch.tensor(text_tokens['input_ids']),
                  'attention_mask': torch.tensor(text_tokens['attention_mask']),
                  'label_binary': target,
                  'label_string': label}
        return sample

    def get_tokenizer(self):
        return self.tokenizer

    def get_data(self):
        self.data = []
        df = pd.read_json(self.data_path)
        for index, row in df.iterrows():
            self.data.append({'text': row['sent'], 'label': " ".join(row['entity'])})
