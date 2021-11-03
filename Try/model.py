import torch.nn as nn
from transformers import AutoModel


class model(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)

        hidden_dim = self.model.config.hidden_size
        vocab_size = self.model.config.vocab_size
        self.logit_layer = nn.Linear(hidden_dim, vocab_size)

    def forward(self, inp):

        x = self.model(inp['input_ids'],
                       inp['attention_mask'])[0]  # [B, L, D]
        cls_emb = x[:, 0, :]  # [B, D]

        logit = self.logit_layer(cls_emb)  # [B, C]

        return logit
