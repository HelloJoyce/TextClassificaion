import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig, BertPreTrainedModel


class Model(nn.Module):
    def __init__(self, config, n_filters, filter_size):
        super(Model, self).__init__()
        model_config = BertConfig.from_pretrained(config.bert_path, num_labels=config.num_classes)
        self.bert = BertModel.from_pretrained(config.bert_path, config=model_config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.rnn = nn.LSTM(config.hidden_size,
                           config.hidden_size,
                           3,
                           bidirectional=True,
                           batch_first=True,
                           dropout=0.1)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.classifier = nn.Linear(config.hidden_size * 2, config.num_classes)

    def forward(self, x):
        context = x[0]
        mask = x[1]
        token_type_ids = x[2]
        pooled, _ = self.bert(context, attention_mask=mask, token_type_ids=token_type_ids)
        encoded_layers = self.dropout(pooled)
        _, (hidden, cell) = self.rnn(encoded_layers)
        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]),
                                        dim=1))  # 连接最后一层的双向输出

        out = self.classifier(hidden)
        return out


class BertLSTM(BertPreTrainedModel):
    def __init__(self, config, num_labels, rnn_hidden_size, num_layers, bidirectional, dropout):
        super(BertLSTM, self).__init__()
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.rnn = nn.LSTM(config.hidden_size,
                           rnn_hidden_size,
                           num_layers,
                           bidirectional=bidirectional,
                           batch_first=True,
                           dropout=dropout)
        self.classifier = nn.Linear(rnn_hidden_size * 2, num_labels)

        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        encoded_layers, _ = self.bert(input_ids,
                                      token_type_ids,
                                      attention_mask,
                                      output_all_encoded_layers=False)
        encoded_layers = self.dropout(encoded_layers)

        _, (hidden, cell) = self.rnn(encoded_layers)
        # outputs: [batch_size, seq_len, rnn_hidden_size * 2]
        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]),
                                        dim=1))  # 连接最后一层的双向输出
        logits = self.classifier(hidden)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits
