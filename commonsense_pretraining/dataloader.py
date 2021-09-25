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

    def __init__(self, data_dir, split, tokenizer, max_seq_len=64,
                 use_reason=False, text2text=True, uniqa = False):
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
        self.text2text = text2text
        self.uniqa = uniqa

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

    @staticmethod
    def _prepare_text2text(record):
        """
        Input:
            {'text': __, 'label': 1/0}

        Output:
            text: 'c2s sentence: __' \n
            label: 'true' or 'false'

        :returns: text, label
        :rtype: tuple[str]
        """
        input_text = record['text']
        answer = 'true' if record['label'] else 'false'

        # Text-to-Text
        text = f'semeval sentence: {input_text} </s>'
        label = f'{answer} </s>'

        return text, label

    def __getitem__(self, idx):
        record = self.data[idx]

        assert type(record['text']) == str, 'TypeError: idx: record - {}: {}'.format(idx, record)

        if self.text2text:
            # Format input & label
            text, label = self._prepare_text2text(record)
            if self.uniqa:
              text = text.split(':')[1][1:]
              text = 'Is the following sentence correct?\n' + text
              label = label.replace('false', 'no')
              label = label.replace('true', 'yes')
            target_len = 2
            # Tokenize
            input_encoded = self.tokenizer.encode_plus(text=text,
                                                       add_special_tokens=False,
                                                       padding='max_length',
                                                       max_length=self.max_seq_len,
                                                       truncation=True,
                                                       return_attention_mask=True)

            target_encoded = self.tokenizer.encode_plus(text=label,
                                                        add_special_tokens=False,
                                                        padding='max_length',
                                                        max_length=target_len,
                                                        return_attention_mask=True)

            input_token_ids = torch.tensor(input_encoded['input_ids'])
            input_attn_mask = torch.tensor(input_encoded['attention_mask'])

            target_token_ids = torch.tensor(target_encoded['input_ids'])
            target_attn_mask = torch.tensor(target_encoded['attention_mask'])

            # Output
            sample = {'input_tokens': input_token_ids,
                      'input_attn_mask': input_attn_mask,
                      'target_tokens': target_token_ids,
                      'target_attn_mask': target_attn_mask}

            return sample

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

    dataset = SemEval20Dataset(path_, 'dev', tokenizer='roberta-base', use_reason=True)
    dataloader = DataLoader(dataset, batch_size=1)
    print(len(dataloader))

    batch = next(iter(dataloader))
    print('Sample batch: \n{}\n{}'.format(batch['tokens'], batch['attn_mask']))
