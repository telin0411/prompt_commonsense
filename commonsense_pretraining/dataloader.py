"""
Preprocessing Commonsense Datasets
"""

import os
import torch
import random
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


class SemEval20Dataset(Dataset):
    """SemEval2020 - Task #4"""

    def __init__(self, data_dir, split, tokenizer, max_seq_len=64, use_reason=False):
        """
        Loads raw dataset for the given fold.

        :param str data_dir: path to dataset directory
        :param str split: train/dev/test
        :param tokenizer: tokenizer name (e.g. 'roberta-base', 't5-3b', etc.)
        :param int max_seq_len: tokenized sequence length (padded)
        :param bool use_reason: if set, includes reasons as statements
        """
        self.split = split
        self.max_seq_len = max_seq_len
        self.use_reason = use_reason

        # Prepare data
        self.data = self.preprocess(data_dir)

        # Setup tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)

    def preprocess(self, data_dir):
        """
        Prepares dataset from raw csv.

        Columns = ['Correct Statement', 'Incorrect Statement',
                    'Right Reason1', 'Confusing Reason1',
                    'Confusing Reason2', 'Right Reason2',
                    'Right Reason3']

        :returns: list of text, label
        :rtype: list[dict]
        """
        path = os.path.join(data_dir, '{}.csv'.format(self.split))

        df = pd.read_csv(path).dropna()

        correct = df['Correct Statement'].tolist()
        incorrect = df['Incorrect Statement'].tolist()

        if self.use_reason:
            # Include reasons as statements
            correct += df['Right Reason1'].tolist()
            correct += df['Right Reason2'].tolist()
            correct += df['Right Reason3'].tolist()

            incorrect += df['Confusing Reason1'].tolist()
            incorrect += df['Confusing Reason2'].tolist()

        data = [{'text': x, 'label': 1} for x in correct]
        data += [{'text': x, 'label': 0} for x in incorrect]

        # Shuffle with fixed seed
        random.seed(0)
        random.shuffle(data)

        return data

    def __len__(self):
        return len(self.data)

    def get_tokenizer(self):
        return self.tokenizer

    def __getitem__(self, idx):
        record = self.data[idx]

        assert type(record['text']) == str, 'TypeError: idx: record - {}: {}'.format(idx, record)

        # TODO: Replace [CLS] --> Avg-Pool-Seq || ask Hope
        # Tokenized format: [CLS] [text] [PAD]
        tokens = [self.tokenizer.cls_token]
        tokens += self.tokenizer.tokenize(record['text'])

        tokens = self.tokenizer.encode_plus(tokens,
                                            padding='max_length',
                                            max_length=self.max_seq_len,
                                            add_special_tokens=False,
                                            return_attention_mask=True)

        token_ids = torch.tensor(tokens['input_ids'])
        attn_mask = torch.tensor(tokens['attention_mask'])
        label = record['label']

        # Output
        sample = {'tokens': token_ids,
                  'attn_mask': attn_mask,
                  'label': label}
        return sample


if __name__ == '__main__':
    path_ = './datasets/semeval_2020_task4'

    dataset = SemEval20Dataset(path_, 'dev', tokenizer='roberta-base')
    dataloader = DataLoader(dataset, batch_size=2)

    batch = next(iter(dataloader))
    print('Sample batch: \n{}\n{}'.format(batch['tokens'], batch['attn_mask']))
