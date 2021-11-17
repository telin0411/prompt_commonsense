"""
Preprocessing Commonsense Datasets
"""
import os
import json
import torch
import random
import pandas as pd
from transformers import AutoTokenizer
from torch.utils.data import Dataset, ConcatDataset


class BaseDataset(Dataset):
    """
    Base class for Datasets
    """

    def __init__(self, split, tokenizer, max_seq_len=128, text2text=True, uniqa=False):
        """
        Processes raw dataset

        :param str split: train/dev/test; (selects `dev` if no `test`)
        :param str tokenizer: tokenizer name (e.g. 'roberta-base', 't5-3b', etc.)
        :param int max_seq_len: tokenized sequence length (padded)
        :param bool text2text: parse dataset in T5 format.
        :param bool uniqa: format dataset in unifiedQA format
        """
        self.split = split
        self.max_seq_len = max_seq_len
        self.text2text = text2text
        self.tok_name = tokenizer
        self.uniqa = uniqa
        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.tok_name)

        # Process dataset (in subclass)
        self.data = None

    def _preprocess(self, data_dir):
        """
        Called within the __init__() of subclasses as follows: \n

        `self.data = self._preprocess(data_dir)`

        :returns: list of text & label dicts
        """
        pass

    def __len__(self):
        return len(self.data)

    def get_tokenizer(self):
        return self.tokenizer

    @staticmethod
    def _get_dataset(name, **kwargs):
        """Datasets to concatenate"""

        datasets = {'com2sense': Com2SenseDataset}

        dataset = datasets[name](**kwargs)
        return dataset

    @staticmethod
    def _get_path(name):
        """Relative paths"""

        paths = {'com2sense': './datasets/com2sense'}

        return paths[name]

    def get_classname(self):
        return self.__class__.__name__

    def concat(self, dataset_names):

        args = {'split': self.split,
                'tokenizer': self.tok_name,
                'max_seq_len': self.max_seq_len,
                'text2text': self.text2text,
                'uniqa': self.uniqa}

        datasets = []
        for name in dataset_names:
            ds = self._get_dataset(name, **args)
            datasets.append(self._get_dataset(name, **args))

        datasets = ConcatDataset(datasets)

        return datasets

    @staticmethod
    def _prepare_text2text(record):
        """
        Formats input text & label in `__getitem__()`

        This is particularly desirable for text-to-text format,
        as it varies with datasets.

        Override this method for datasets (subclass)

        **Note**: The target text is given with length=2,
                e.g. `label </s>`

        :param record: single sample
        :returns: text, label
        """
        text, label = record['text'], record['label']

        return text, label

    def max_len_tokenized(self):
        """
        Max tokenized sequence length, assuming text-to-text format
        TODO: Revise it
        """
        return max([len(self.tokenizer.encode(''.join(d.values()))) for d in self.data])

    def __getitem__(self, idx):
        record = self.data[idx]
        if self.text2text:
            # Format input & label
            int2str = {0: 'no', 1: 'yes'}
            statement, explanation, label = record['statement'], record['explanation'], record['label']
            text = f'{statement}'

            text = 'Is the following statement correct?\n' + text
            # text = explanation + '\nBased on the previous reason, is the following statement correct?\n' + text
            label = int2str[label]
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
        else:

            statement, explanation, label = record['statement'], record['explanation'], record['label']

            cls = self.tokenizer.cls_token

            text = f'{cls} {statement}'

            tokens = self.tokenizer(text=text,
                                    padding='max_length',
                                    max_length=self.max_seq_len,
                                    truncation=True,
                                    add_special_tokens=False,
                                    return_attention_mask=True)

            token_ids = torch.tensor(tokens['input_ids'])
            attn_mask = torch.tensor(tokens['attention_mask'])

            # Output
            sample = {'tokens': token_ids,
                      'attn_mask': attn_mask,
                      'label': label}
        return sample


class Com2SenseDataset(BaseDataset):
    """
    Complementary Commonsense Benchmark

    [True]  It's more comfortable to sleep on a mattress than the floor.
    [False] It's more comfortable to sleep on the floor than a mattress.
    """

    def __init__(self, split, tokenizer, max_seq_len, text2text, uniqa=False):

        super().__init__(split, tokenizer, max_seq_len, text2text)

        self.uniqa = uniqa
        self.text2text = text2text
        # Read dataset
        data_dir = self._get_path('com2sense')

        self.data = self._preprocess(data_dir)

    def _preprocess(self, data_dir):
        """
        Parses raw dataset file (jsonl). \n
        The complementary sentences are unpaired and treated as independent samples.

        Input:
            [
                {_id: _, 'sent_1': ___, 'label_1': _, 'sent_2': ___, 'label_2': _},
                ...
                {_id: _, 'sent_1': ___, 'label_1': _, 'sent_2': ___, 'label_2': _}
            ]

        Output:
            [
                {_id: _, 'sent_1': ___, 'label_1': 1/0},
                {_id: _, 'sent_2': ___, 'label_2': 1/0},
                ...
                {_id: _, 'sent_1': ___, 'label_1': 1/0},
                {_id: _, 'sent_2': ___, 'label_2': 1/0}
            ]

        :param str data_dir: path to dataset dir
        :returns: sentence, label
        :rtype: list[dict]
        """
        path = os.path.join(data_dir, f'{self.split}.json')

        # Read data
        df = pd.read_json(path)

        raw_data = df.to_dict(orient='records')

        data = []
        for pair in raw_data:
            sample = dict(_id=pair['id'], statement=pair['statement'],
                          label=pair['label'], explanation=pair['explanation'])
            data.append(sample)

        if self.split == 'train':
            random.seed(0)
            random.shuffle(data)

        return data

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
        text = f'com2sense sentence: {input_text} </s>'
        label = f'{answer} </s>'

        return text, label
