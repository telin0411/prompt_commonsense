import torch
import random
import pandas as pd
from transformers import AutoTokenizer
from torch.utils.data import Dataset


class ExDataset(Dataset):
    def __init__(self, data_path, split, tokenizer, max_seq_len=128, num_entity=3):
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
        self.num_entity = num_entity

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.tok_name)

        # Process dataset (in subclass)
        self.data = None
        self.get_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        record = self.data[idx]

        sent, entity = record['sent'], record['entity']

        mask = self.tokenizer.mask_token

        text = sent.replace(entity, mask)

        sent_token = self.tokenizer(text=text,
                                    padding='max_length',
                                    max_length=self.max_seq_len,
                                    truncation=True,
                                    add_special_tokens=False,
                                    return_attention_mask=True)

        label = self.tokenizer(text=entity,
                               padding='max_length',
                               max_length=1,
                               truncation=True,
                               add_special_tokens=False,
                               return_attention_mask=True)[0]
        # Output
        sample = {'input_ids': torch.tensor(sent_token['input_ids']),
                  'attention_mask': torch.tensor(sent_token['attention_mask']),
                  'label': label}
        return sample

    def get_tokenizer(self):
        return self.tokenizer

    def get_data(self):
        self.data = []
        df = pd.read_json(self.data_path)
        for index, row in df.iterrows():
            if row['label'] == 1:
                for entity in row['entity']:
                    self.data.append({'sent': row['sent'], 'entity': entity})
