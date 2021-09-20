"""
Transformer Models for Binary classification
"""
import torch
import torch.nn as nn
from transformers import AutoModel


class Transformer(nn.Module):
    def __init__(self, model_name, num_cls, is_pretrained=True):
        """
        Transformer model with classification layer (over [CLS] token)

        :param model_name: transformer model (e.g. roberta-base)
        :param num_cls: model num of classes (logits layer)
        :param is_pretrained: load pretrained weights (official)
        """
        super().__init__()

        # Transformer
        self.model = AutoModel.from_pretrained(model_name)      # ToDo: Grad-Ckpt = True (config)
        hidden_dim = self.model.config.hidden_size

        # Classification
        self.logit_layer = nn.Linear(hidden_dim, num_cls)

    def forward(self, tokens, mask):
        # tokens: [B, L]
        # mask: [B, L]
        x = self.model(tokens, mask)[0]     # [B, L, D]
        cls_emb = x[:, 0, :]                # [B, D]

        # logits
        logit = self.logit_layer(cls_emb)   # [B, C]

        return logit


if __name__ == '__main__':
    n_cls = 2
    t = torch.randint(20, [1, 8])           # [B, L]
    a = torch.randint(n_cls, [1, 8])        # [B, L]
    m = Transformer('roberta-base', n_cls)

    res = m(t, a)                           # [B, C]

    print(res.shape)
