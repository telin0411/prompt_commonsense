"""
Preprocessing for statement and explanation pretraining
"""

import json
import pandas as pd
import torch
from transformers import AutoTokenizer
from torch.utils.data import Dataset
import random


class T5Dataset(Dataset):
    def __init__(self, file_path, tokenizer, input_seq_len=128, target_seq_len=2):
        self.file_path = file_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.input_seq_len = input_seq_len
        self.target_seq_len = target_seq_len
        self.data = None

    def __getitem__(self, idx):
        record = self.data[idx]
        input_text, output_text = record['input_text'], record['output_text']

        # Tokenize
        input_encoded = self.tokenizer.encode_plus(text=input_text,
                                                   add_special_tokens=False,
                                                   padding='max_length',
                                                   max_length=self.input_seq_len,
                                                   truncation=True,
                                                   return_attention_mask=True)

        target_encoded = self.tokenizer.encode_plus(text=output_text,
                                                    add_special_tokens=False,
                                                    padding='max_length',
                                                    max_length=self.target_seq_len,
                                                    truncation=True,
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

    def get_tokenizer(self):
        return self.tokenizer


class COSE(T5Dataset):
    def __init__(self, file_path, tokenizer, input_seq_len):
        super().__init__(file_path, tokenizer, input_seq_len, target_seq_len=24)

        self.data_preprocessing()

    def data_preprocessing(self):
        self.data = []
        df = pd.read_json(self.file_path)
        for idx, row in df.iterrows():
            question = row['question']
            choice0 = row['choices']['A']
            choice1 = row['choices']['B']
            choice2 = row['choices']['C']
            choice3 = row['choices']['D']
            choice4 = row['choices']['E']
            choices = f"(A){choice0} (B){choice1} (C){choice2} (D){choice3} (E){choice4}"
            answer = row['answer']
            explanation = row['explanation']

            input_text = f"{question}\nPick a choice and answer why.\n{choices}"
            target_text = f"{answer}, because {explanation}"

            self.data.append({'input_text': input_text,
                              'output_text': target_text})


class ECQA(T5Dataset):
    def __init__(self, file_path, tokenizer, input_seq_len):
        super().__init__(file_path, tokenizer, input_seq_len, target_seq_len=24)
        self.data_pos = None
        self.data_neg = None

        self.data_preprocessing()

    def data_preprocessing(self):
        self.data_pos = []
        self.data_neg = []
        df = pd.read_json(self.file_path)
        for idx, row in df.iterrows():
            question = row['question']
            answer = row['answer']
            answer_key = row['answer_key']
            choices = row['choices']
            # positive explanations and the true answer
            explanation_pos = row['explanations'][answer_key]
            input_text = f"{question}\n{choices[answer_key]}"
            target_text = f"True, because {explanation_pos}"
            self.data_pos.append({'input_text': input_text,
                                  'target_text': target_text})
            # negative explanations and the false answer
            row['explanations'].pop(answer_key)  # pop the true answer and remain the false
            for k, v in row['explanations'].items():
                input_text = f"{question}\n{choices[k]}"
                target_text = f"False, because {v}"
                self.data_neg.append({'input_text': input_text,
                                      'target_text': target_text})
        self.data = self.data_neg + self.data_pos

    def __getitem__(self, index):
        if index >= len(self.data_pos) + len(self.data_neg):
            index = random.randint(len(self.data_pos), len(self.data_pos) + len(self.data_neg) - 1)

        record = self.data[index]
        input_text, output_text = record['input_text'], record['output_text']

        # Tokenize
        input_encoded = self.tokenizer.encode_plus(text=input_text,
                                                   add_special_tokens=False,
                                                   padding='max_length',
                                                   max_length=self.input_seq_len,
                                                   truncation=True,
                                                   return_attention_mask=True)

        target_encoded = self.tokenizer.encode_plus(text=output_text,
                                                    add_special_tokens=False,
                                                    padding='max_length',
                                                    max_length=self.target_seq_len,
                                                    truncation=True,
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

    def len(self):
        return 2 * max(len(self.data_pos), len(self.data_neg))
