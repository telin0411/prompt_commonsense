from model import ModelA
from transformers import AutoTokenizer
from transformers import RobertaForMaskedLM
import torch
from data import BaseDataset
from torch.utils.data import DataLoader
import torch.nn as nn

if __name__ == '__main__':

    model_name = 'roberta-base'
    data_dir = './datasets/com2sense'

    model = ModelA(prompt_model_name=model_name, task_model_name=model_name)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), 1e-5)

    dataset = BaseDataset('dev', tokenizer=model_name, data_dir=data_dir, max_seq_len=32)
    train_datasets = dataset.concat('com2sense')
    train_loader = DataLoader(train_datasets, 8, shuffle=True, drop_last=True, num_workers=2)

    for e in range(10):
        for t, sample in enumerate(train_loader):
            logits = model(sample)
            loss = criterion(logits, sample['label'])
            optimizer.zero_grad()
            loss.backward()
            print(loss, t)


    """
    model = RobertaForMaskedLM.from_pretrained('roberta-base')
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')

    text = ["The <mask> of China is <mask>.",
            "The <mask> of Japan is <mask>.",
            "Water can dissolve <mask>.",
            "Fire can melt <mask>."]

    text_token = tokenizer(text=text,
                           add_special_tokens=False,
                           max_length=16,
                           padding='max_length',
                           return_tensors='pt',
                           truncation=True)

    mask_id = tokenizer.mask_token_id
    pos_mask = (text_token.data["input_ids"] == mask_id).long()
    pos_mask_ex = pos_mask.unsqueeze(2).repeat((1, 1, model.config.vocab_size))

    out = model(input_ids=text_token.data['input_ids'])

    mask_embedding = out['logits'] * pos_mask_ex

    token = mask_embedding.argmax(dim=2)
    txt = [tokenizer.decode(x) for x in token]
    print(txt)
    """