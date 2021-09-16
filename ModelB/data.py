"""
Preprocessing Commonsense Datasets
"""
import os
import pandas as pd
import torch
from transformers import AutoTokenizer
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    """
    Base class for Datasets
    """

    def __init__(self, split, tokenizer, data_dir, max_seq_len=128):
        """
        Processes raw dataset

        :param str split: train/dev/test; (selects `dev` if no `test`)
        :param str tokenizer: tokenizer name (e.g. 'roberta-base', 't5-3b', etc.)
        :param int max_seq_len: tokenized sequence length (padded)
        """
        self.data_dir = data_dir
        self.split = split
        self.max_seq_len = max_seq_len
        self.tok_name = tokenizer

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.tok_name)

        # Process dataset (in subclass)
        self.data = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        record = self.data[idx]

        text, label = record['text'], record['label']

        cls = self.tokenizer.cls_token

        text = f'{cls} {text}'

        tokens = self.tokenizer(text=text,
                                padding='max_length',
                                max_length=self.max_seq_len,
                                truncation=True,
                                add_special_tokens=False,
                                return_attention_mask=True)

        # Output
        sample = {'input_ids': torch.tensor(tokens['input_ids']),
                  'attention_mask': torch.tensor(tokens['attention_mask']),
                  'label': label}
        return sample


class Com2SenseDataset(BaseDataset):
    """
    Complementary Commonsense Benchmark

    [True]  It's more comfortable to sleep on a mattress than the floor.
    [False] It's more comfortable to sleep on the floor than a mattress.
    """

    def __init__(self, split, tokenizer, data_dir, max_seq_len, template="rather than", mask_len=1):

        super().__init__(split, tokenizer, data_dir, max_seq_len)

        self.data = self.preprocess(data_dir, template, mask_len)

    def preprocess(self, data_dir, template, mask_len):

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
            try:
                is_qualified, pair = self.data_preprocessing(pair, template, mask_len)
            except TypeError:
                print(pair)
                is_qualified = False
            if not is_qualified:
                continue
            sample_1 = dict(_id=pair['_id'], text=pair['sent_1'], label=pair['label_1'])
            sample_2 = dict(_id=pair['_id'], text=pair['sent_2'], label=pair['label_2'])
            data += [sample_1, sample_2]

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

    def data_preprocessing(self, sample: dict, template: str, mask_len=1):
        """
        :param   sample: a piece of data sample in Com2Sense dataset
        :param template: template for transformation from causality to comparison, e.g., instead of, rather than, etc
        :param mask_len: The length of mask, default 1
        :return: is_qualified, sample
        """
        if sample['scenario'] == 'comparison':
            return False, None
        elif sample['scenario'] == 'causal':
            sent_1 = sample['sent_1']
            sent_2 = sample['sent_2']
            is_qualified, entity_1, entity_2 = self._isOneWordDiff(sent_1, sent_2)
            if not is_qualified:
                return False, None
            else:
                # Get the index of the entity, ready to transform to comparison
                insert_loc_1 = sent_1.find(entity_1) + len(entity_1)
                insert_loc_2 = sent_2.find(entity_2) + len(entity_2)

                # Insert template concatenated by masks
                sent_1 = sent_1[:insert_loc_1] + f" {template}" + mask_len*" <mask>" + sent_1[insert_loc_1:]
                sent_2 = sent_2[:insert_loc_2] + f" {template}" + mask_len*" <mask>" + sent_2[insert_loc_2:]

                sample['sent_1'] = sent_1
                sample['sent_2'] = sent_2

                return True, sample
