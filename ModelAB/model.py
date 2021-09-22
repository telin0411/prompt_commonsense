"""
Model A and Model B
"""
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, RobertaForMaskedLM


class ModelA(nn.Module):
    def __init__(self,
                 prompt_model_name,
                 task_model_name,
                 num_cls=2,
                 num_prompt_model_layer=-1,
                 num_task_model_layer=-1,
                 example_file=None):
        """
        :param      prompt_model_name: The model used for prompt model, e.g., RoBERTa
        :param        task_model_name: The model used for task model, e.g., RoBERTa
        :param                num_cls: The number of class, default 2, because it's a T/F question
        :param num_prompt_model_layer: The layer number of prompt model. Set -1 to use the pretrain model's default
        :param   num_task_model_layer: The layer number of task model. Set -1 to use the pretrain model's default
        :param           example_file: If it is not None, create a file storing prompt sentences
        """
        super().__init__()
        # configure Prompt Model
        if num_prompt_model_layer > -1:
            self.prompt_model = RobertaForMaskedLM.from_pretrained(prompt_model_name,
                                                                   num_hidden_layers=num_prompt_model_layer)
        else:
            self.prompt_model = RobertaForMaskedLM.from_pretrained(prompt_model_name)

        self.prompt_tokenizer = AutoTokenizer.from_pretrained(prompt_model_name)

        # configure Task Model
        if num_task_model_layer > -1:
            self.task_model = AutoModel.from_pretrained(task_model_name, num_hidden_layers=num_task_model_layer)
        else:
            self.task_model = AutoModel.from_pretrained(task_model_name)
        self.task_classifier = nn.Linear(self.task_model.config.hidden_size, num_cls)

        # set for example file
        self.example_file = './' + example_file + '.txt' if example_file is not None else None

    def forward(self, inp):
        """
        :param inp: The input of the Model B, a dict having 'input_ids' and 'attention_mask' keys,
                    input_ids shape (batch_size, seq_len), i.e. (B, L)
        :return: The logits, of shape (batch_size, num_cls)
        """
        # locate the <mask> in each text
        mask_id = self.prompt_tokenizer.mask_token_id
        pos_mask = (inp["input_ids"] == mask_id).long() if self.pos_mask == -1 else self.pos_mask

        # Extend the dimension from (B, L) to (B, L, D)
        pos_mask_ex = pos_mask.unsqueeze(2).repeat((1, 1, self.prompt_model.config.hidden_size))

        # PROMPT MODEL
        prompt_embeds = self.prompt_model(inp["input_ids"], inp['attention_mask'])["logits"]              # (B, L, V)
        prompt_tokens = prompt_embeds.argmax(dim=2) * pos_mask_ex + inp["input_ids"] * (1 - pos_mask_ex)  # (B, L)

        # Show some examples
        if self.example_file:
            with open(self.example_file, "a") as fp:
                for tokens in prompt_tokens:
                    sentence = self.prompt_tokenizer.decode(tokens, skip_special_tokens=True)
                    fp.write(sentence + '\n')
                fp.close()

        # TASK MODEL
        out_embeds = self.task_model(prompt_tokens, inp['attention_mask'])
        cls_embeds = out_embeds[0][:, 0, :]
        logits = self.task_classifier(cls_embeds)

        return logits


class ModelB(nn.Module):
    def __init__(self, prompt_model_name,
                 task_model_name,
                 num_cls=2,
                 pos_mask=-1,
                 num_prompt_model_layer=-1,
                 num_task_model_layer=-1,
                 example_file=None):
        """
        :param      prompt_model_name: The model used for prompt model, e.g., RoBERTa
        :param        task_model_name: The model used for task model, e.g., RoBERTa
        :param                num_cls: The number of class, default 2, because it's a T/F question
        :param               pos_mask: The position of <mask>. Set -1 to find the last '1' in attention mask
        :param num_prompt_model_layer: The layer number of prompt model. Set -1 to use the pretrain model's default
        :param   num_task_model_layer: The layer number of task model. Set -1 to use the pretrain model's default
        :param           example_file: If it is not None, create a file storing prompt sentences
        """
        super().__init__()
        # configure Prompt Model
        if num_prompt_model_layer > -1:
            self.prompt_model = AutoModel.from_pretrained(prompt_model_name, num_hidden_layers=num_prompt_model_layer)
        else:
            self.prompt_model = AutoModel.from_pretrained(prompt_model_name)

        self.prompt_tokenizer = AutoTokenizer.from_pretrained(prompt_model_name)

        # configure Task Model
        if num_task_model_layer > -1:
            self.task_model = AutoModel.from_pretrained(task_model_name, num_hidden_layers=num_task_model_layer)
        else:
            self.task_model = AutoModel.from_pretrained(task_model_name)
        self.task_embedding_layer = self.task_model.get_input_embeddings()
        self.task_classifier = nn.Linear(self.task_model.config.hidden_size, num_cls)

        # others
        self.pos_mask = pos_mask
        self.example_file = './' + example_file + '.txt' if example_file is not None else None

    def forward(self, inp):
        """
        :param inp: The input of the Model B, a dict having 'input_ids' and 'attention_mask' keys,
                    input_ids shape (batch_size, seq_len), i.e. (B, L)
        :return: The logits, of shape (batch_size, num_cls)
        """
        # locate the <mask> in each text
        mask_id = self.prompt_tokenizer.mask_token_id
        pos_mask = (inp["input_ids"] == mask_id).long() if self.pos_mask == -1 else self.pos_mask

        # Extend the dimension from (B, L) to (B, L, D)
        pos_mask_ex = pos_mask.unsqueeze(2).repeat((1, 1, self.prompt_model.config.hidden_size))

        # results of prompt embedding
        out_prompt = self.prompt_model(inp["input_ids"], inp["attention_mask"])[0]  # (B, L, D)

        # mask embedding, of shape (B, L, D); zeros except on mask position
        mask_embedding = out_prompt * pos_mask_ex

        # TASK MODEL
        words_embedding = self.task_embedding_layer(inp["input_ids"])
        input_embedding = words_embedding * (1 - pos_mask_ex) + mask_embedding

        # Trace back the KNN tokens
        if self.example_file:
            device = next(self.prompt_model.parameters()).device
            B, L, D = mask_embedding.shape
            V = self.task_model.config.vocab_size
            prompt_tokens = torch.clone(inp["input_ids"])

            vocab_token = torch.arange(V).to(device)  # (V, 1)
            vocab_embedding = self.task_embedding_layer(vocab_token)   # (V, D)

            for b in range(B):
                for l in range(L):
                    input_embedding_ = input_embedding[b, l, :].unsqueeze(0)   # (1, D)
                    r = torch.norm(input_embedding_ - vocab_embedding, dim=1)  # (V, 1)
                    _, token = r.topk(k=10, largest=False)
                    # Choose one token from the top 10
                    prompt_tokens[b][l] = token[0]

            recovered_token = inp["input_ids"] * (1 - pos_mask) + prompt_tokens * pos_mask

            with open(self.example_file, "a") as fp:
                for tokens in recovered_token:
                    sentence = self.prompt_tokenizer.decode(tokens)
                    fp.write(sentence + '\n')
                fp.close()

        out_embedding = self.task_model(inputs_embeds=input_embedding, attention_mask=inp['attention_mask'])
        cls_embedding = out_embedding[0][:, 0, :]
        logits = self.task_classifier(cls_embedding)

        return logits
