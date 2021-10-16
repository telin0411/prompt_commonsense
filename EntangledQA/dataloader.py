"""
Preprocessing EntangledQA Datasets
"""
import os
import json
import torch
import random
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer
from torch.utils.data import Dataset, ConcatDataset, DataLoader, RandomSampler


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

        datasets = {
            'com2sense': Com2SenseDataset,
            'EntangledQA': EntangledQADataset,
            'semeval_2020': SemEval20Dataset,
            'semeval20_comparative': SemEval20ComparativeDataset,
        }

        dataset = datasets[name](**kwargs)
        return dataset

    @staticmethod
    def _get_path(name):
        """Relative paths"""

        paths = {
            'com2sense': './datasets/com2sense',
            'cycic3': './datasets/cycic3',
            'semeval_2020': './datasets/semeval_2020_task4',
            'semeval20_comparative': './datasets/semeval20_comparative',
        }

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
        for i in range(len(dataset_names)):
            name = dataset_names[i]
            args['split'] = self.split[i]
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
                # text = 'Is the following sentence correct?\n' + text
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


class EntangledQADataset(BaseDataset):
    """
    Cycic3 EntangledQA Benchmark

    [True]  Most winged animals can fly.
    [False] Abraham Lincoln was killed in the Vietnam War.
    """

    def __init__(self, split, tokenizer, max_seq_len, text2text,
                 uniqa=False, strip_sentence_prefix=False):

        assert split in ["train", "hard_few",
                         "dev-a", "dev-b",
                         "test-a", "test-b",
                         "released-a", "released-b",
                         "train_dev-a", "train_dev-b"]

        self.split2dataset_prefix = {
            "train": "training",
            "dev-a": "dev_a",
            "dev-b": "dev_b",
            "train_dev-a": "train_dev_a",
            "train_dev-b": "train_dev_b",
            "test-a": "test_a",
            "test-b": "test_b",
            "released-a": "cycic3a_released",
            "released-b": "cycic3b_released",
            "hard_few": "hard_few",
        }

        super().__init__(split, tokenizer, max_seq_len, text2text)

        self.uniqa = uniqa
        self.text2text = text2text
        self.strip_sentence_prefix = strip_sentence_prefix
        if self.uniqa or self.text2text:
            self.strip_sentence_prefix = True

        # Read dataset
        data_dir = self._get_path('cycic3')

        self.data = self._preprocess(data_dir)

    def _preprocess(self, data_dir):
        """
        Parses raw dataset file (jsonl). \n

        Input:
            [
                {'guid': _, 'run_id': _, 'question': ___, 'categories': ___,
                 'answer_option0': _, 'answer_option1': _},
                ...
                {'guid': _, 'run_id': _, 'question': ___, 'categories': ___,
                 'answer_option0': _, 'answer_option1': _},
            ]

        * guid will be discarded as only run_id is important for entanglement.
        * categories are not used for now.

        Output:
            [
                {_id: _, 'sent': ___, 'label': 1/0},
                ...
                {_id: _, 'sent': ___, 'label': 1/0}
            ]

        :param str data_dir: path to dataset dir
        :returns: sentence, label
        :rtype: list[dict]
        """
        q_path = os.path.join(data_dir,
                              f'{self.split2dataset_prefix[self.split]}_questions.jsonl')
        l_path = os.path.join(data_dir,
                              f'{self.split2dataset_prefix[self.split]}_labels.jsonl')

        # Read data
        questions = []
        labels = []
        data = []

        fq = open(q_path)
        for line in fq:
            questions.append(json.loads(line.strip()))
        fq.close()

        fl = open(l_path)
        for line in fl:
            labels.append(json.loads(line.strip()))
        fl.close()

        assert len(questions) == len(labels)

        # Map labels
        label2int = {'True': 1, 'False': 0}

        for i in range(len(questions)):
            qi = questions[i]
            li = labels[i]
            assert qi["guid"] == li["guid"]
            assert qi["run_id"] == li["run_id"]
            correct_answer = li["correct_answer"]
            correct_label = qi["answer_option{}".format(correct_answer)]
            label = label2int[correct_label]
            run_id = qi["run_id"]
            question = qi["question"]
            if self.strip_sentence_prefix:
                question = question.split("True or False:")[-1].strip()
            datum = dict(
                _id=run_id,
                text=question,
                label=label
            )
            data.append(datum)

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

        if input_text[-1] == ".":
            input_text = input_text[:-1] + "?"
        elif input_text[-1] == "?":
            pass
        else:
            input_text = input_text + "?"

        text = f'EntangledQA sentence: {input_text} </s>'
        label = f'{answer} </s>'

        return text, label


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


class SemEval20Dataset(BaseDataset):
    """SemEval2020 - Task #4"""

    def __init__(self, split, tokenizer, max_seq_len=64,
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
        data_dir = self._get_path('semeval_2020')
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

        # data = [{'text': x[:-1]+"?" if x[-1] == "." else x, 'label': 1} for x in correct]
        data = [{'text': x, 'label': 1} for x in correct]
        # data += [{'text': x[:-1]+"?" if x[-1] == "." else x, 'label': 0} for x in incorrect]
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

        if input_text[-1] == ".":
            input_text = input_text[:-1] + "?"
        elif input_text[-1] == "?":
            pass
        else:
            input_text = input_text + "?"

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
              # text = 'Is the following sentence correct?\n' + text
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


class SemEval20ComparativeDataset(BaseDataset):
    """
    SemEval20 and Cycic Comparative Benchmark

    [True]  Most winged animals can fly.
    [False] Abraham Lincoln was killed in the Vietnam War.
    """

    def __init__(self, split, tokenizer, max_seq_len, text2text,
                 uniqa=False, strip_sentence_prefix=False):

        super().__init__(split, tokenizer, max_seq_len, text2text)

        self.uniqa = uniqa
        self.text2text = text2text
        self.strip_sentence_prefix = strip_sentence_prefix

        # Read dataset
        data_dir = self._get_path('semeval20_comparative')

        self.data = self._preprocess(data_dir)

    def _preprocess(self, data_dir):
        """
        Parses raw dataset file (jsonl). \n

        Input:
            [
                {'guid': _, 'run_id': _, 'question': ___, 'categories': ___,
                 'answer_option0': _, 'answer_option1': _},
                ...
                {'guid': _, 'run_id': _, 'question': ___, 'categories': ___,
                 'answer_option0': _, 'answer_option1': _},
            ]

        * guid will be discarded as only run_id is important for entanglement.
        * categories are not used for now.

        Output:
            [
                {_id: _, 'sent': ___, 'label': 1/0},
                ...
                {_id: _, 'sent': ___, 'label': 1/0}
            ]

        :param str data_dir: path to dataset dir
        :returns: sentence, label
        :rtype: list[dict]
        """
        q_path = os.path.join(data_dir,
                              f'{self.split}.source')
        l_path = os.path.join(data_dir,
                              f'{self.split}.target')

        # Read data
        questions = []
        labels = []
        data = []

        fq = open(q_path)
        for line in fq:
            questions.append(line.split("\\n")[0].strip())
        fq.close()

        fl = open(l_path)
        for line in fl:
            labels.append(line.strip())
        fl.close()

        assert len(questions) == len(labels)

        # Map labels

        for i in range(len(questions)):
            question = questions[i]
            label = labels[i]
            datum = dict(
                _id=str(i),
                text=question,
                label=label
            )
            data.append(datum)

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
        answer = record['label']

        text = f'SemEval sentence: {input_text} </s>'
        label = f'{answer} </s>'

        return text, label



# Testing.
if __name__ == "__main__":
    split = "train"  # Can choose from "train", "dev-a/b", "test-a/b", "released-a/b"

    dataset = EntangledQADataset(
        split=split,
        tokenizer="roberta-large",
        max_seq_len=100,
        text2text=True,
        uniqa=True,
        strip_sentence_prefix=True,
    )

    dataset = SemEval20ComparativeDataset(
        split=split,
        tokenizer="roberta-large",
        max_seq_len=100,
        text2text=True,
        uniqa=True,
        strip_sentence_prefix=True,
    )

    sampler = RandomSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=2)
    epoch_iterator = tqdm(dataloader, desc="Iteration")
    for step, batch in enumerate(epoch_iterator):
        print(step)
        print(batch)
        break

    
    split = ["train-60", "train", "train"]
    dataset = BaseDataset(split, tokenizer="roberta-large", max_seq_len=100, text2text=True, uniqa=True)
    train_datasets = dataset.concat(["com2sense", "EntangledQA", "semeval_2020"])

    sampler = RandomSampler(train_datasets)
    dataloader = DataLoader(train_datasets, sampler=sampler, batch_size=2)
    epoch_iterator = tqdm(dataloader, desc="Iteration")
    for step, batch in enumerate(epoch_iterator):
        print(step)
        print(batch)
        break
