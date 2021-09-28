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
    def __init__(self, split, tokenizer, max_seq_len=128, text2text=True, uniqa = False):
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
        else:

            text, label = record['text'], record['label']
      
            cls = self.tokenizer.cls_token
 
            text = f'{cls} {text}'
  
            tokens = self.tokenizer(text=text,
                              padding='max_length',
                              max_length=self.max_seq_len,
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
    def __init__(self, split, tokenizer, max_seq_len, text2text, uniqa = False):

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

        # Map labels
        label2int = {'True': 1, 'False': 0}

        df['label_1'] = df['label_1'].apply(lambda l: label2int[l])
        df['label_2'] = df['label_2'].apply(lambda l: label2int[l])

        raw_data = df.to_dict(orient='records')

        # add index for pairs       # TODO: Remove this, and use the database ID
        for i, pair in enumerate(raw_data):
            pair['_id'] = i

        data = []
        for pair in raw_data:
            if pair['scenario'] == 'comparison':
                pass
            elif pair['scenario'] == 'causal':
                sent_1 = pair['sent_1']
                sent_2 = pair['sent_2']
                is_qualified, entity_1, entity_2 = self._isOneWordDiff(sent_1, sent_2)
                if not is_qualified:
                    continue
                else:
                    # Get the index of the entity, ready to transform to comparison
                    insert_loc_1 = sent_1.find(entity_1) + len(entity_1)
                    insert_loc_2 = sent_2.find(entity_2) + len(entity_2)

                    # Insert template concatenated by masks
                    sent_1 = sent_1[:insert_loc_1] + " instead of" + " <mask>" + sent_1[insert_loc_1:]
                    sent_2 = sent_2[:insert_loc_2] + " instead of" + " <mask>" + sent_2[insert_loc_2:]

                    pair['sent_1'] = sent_1
                    pair['sent_2'] = sent_2

            sample_1 = dict(_id=pair['_id'], text=pair['sent_1'], label=pair['label_1'])
            sample_2 = dict(_id=pair['_id'], text=pair['sent_2'], label=pair['label_2'])
            if pair['label_1'] == 1: 
                correct_sentence = pair['sent_1']
                other_one = pair['sent_2']
            else:
                correct_sentence = pair['sent_2']
                other_one = pair['sent_1']
            # if self.mc:
            #     sample = dict(_id = pair['_id'], correct= correct_sentence, incorrect = other_one)
            #     data.append(sample)
            # else:
            data += [sample_1, sample_2]
        
        if self.split == 'train':
            random.seed(0)
            random.shuffle(data)

        return data

    @staticmethod
    def _isOneWordDiff(sent_1, sent_2):
        blackList = ['can', 'cannot', 'can\'t', 'do', 'not', 'don\'t', 'will', 'won\'t', 'less', 'more']

        sent_1 = sent_1.replace('.', '')
        sent_2 = sent_2.replace('.', '')

        rest_1 = list(set(sent_1.split()) - set(sent_2.split()))
        rest_2 = list(set(sent_2.split()) - set(sent_1.split()))
        if len(rest_1) == 1 and len(rest_2) == 1 and rest_1[0] not in blackList and rest_2[0] not in blackList:
            return True, rest_1[0], rest_2[0]
        else:
            return False, '', ''

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
