import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertPreTrainedModel, BertModel, BertConfig


class Model(nn.Module):
    def __init__(self, config, n_filters, filter_sizes):
        super(Model, self).__init__()
        model_config = BertConfig.from_pretrained(config.bert_path, num_labels=config.num_classes)
        self.bert = BertModel.from_pretrained(config.bert_path, config=model_config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        for param in self.bert.parameters():
            param.requires_grad = True
        self.classifier = nn.Linear(config.hidden_size, config.num_classes)
        self.W_w = nn.Parameter(torch.Tensor(config.hidden_size, config.hidden_size))
        self.u_w = nn.Parameter(torch.Tensor(config.hidden_size, 1))
        nn.init.uniform_(self.W_w, -0.1, 0.1)
        nn.init.uniform_(self.u_w, -0.1, 0.1)

    def forward(self, x):
        context = x[0]
        mask = x[1]
        token_type_ids = x[2]
        pooled, _ = self.bert(context, attention_mask=mask, token_type_ids=token_type_ids)
        encoded_layers = self.dropout(pooled)
        score = torch.tanh(torch.matmul(encoded_layers, self.W_w))
        attention_weighs = F.softmax(torch.matmul(score, self.u_w), dim=1)
        scored_x = encoded_layers * attention_weighs
        feat = torch.sum(scored_x, dim=1)

        out = self.classifier(feat)
        return out
