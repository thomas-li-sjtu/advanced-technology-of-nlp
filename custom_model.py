# coding: utf-8
# Copyright (c) Antfin, Inc. All rights reserved.
# @Time : 2021/11/21 下午9:08
# @Author : shuheng.zsh
import torch.nn as nn
from transformers import BertModel, BertConfig

class ModelParamConfig:
    def __init__(self):
        self.num_classes = 2
        self.dropout_prob = 0.1


class CustomModel(nn.Module):
    def __init__(self, pretrain_model_path, model_param_config):
        super(CustomModel, self).__init__()
        self.config = BertConfig.from_pretrained(pretrain_model_path, output_hidden_states=True)
        self.bert = BertModel.from_pretrained(pretrain_model_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.dropout = nn.Dropout(model_param_config.dropout_prob)
        self.fc = nn.Linear(self.config.hidden_size, model_param_config.num_classes)

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None, loss_fn=None):
        sequence_out, cls_out = self.bert(input_ids=input_ids, token_type_ids=token_type_ids,
                                          attention_mask=attention_mask, return_dict=False)
        cls_out = self.dropout(cls_out)
        logits = self.fc(cls_out)
        if loss_fn is not None:
            loss = loss_fn(logits, labels)
            return logits, loss
        else:
            return logits
