# coding: utf-8
# Copyright (c) Antfin, Inc. All rights reserved.
# @Time : 2021/11/21 下午9:02
# @Author : shuheng.zsh
import os
import json
import argparse
import logging
import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from transformers import BertTokenizer, DataCollatorWithPadding
from custom_model import ModelParamConfig, CustomModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('--max_len', type=int, help='pair文本最大长度', default=32)
parser.add_argument('--batch_size', type=int, help='batch_size', default=512)
parser.add_argument('--model_dir', type=str, help='加载模型的地址', default='./pretrained_model/bert')
parser.add_argument('--output_dir', type=str, help='测试结果写入文件夹', default='test_results')
parser.add_argument('--run_mode', type=str, help='运行模式normal or test', default='test')
args = parser.parse_args()

# 处理数据
if args.run_mode == 'test':
    data_files = {"test": "data/oppo_test_tiny"}
else:
    data_files = {"test": "data/oppo_test"}
dataset = load_dataset('csv', data_files=data_files, delimiter='\t', quoting=3)
vocab_file_dir = './data/vocab.txt'
tokenizer = BertTokenizer.from_pretrained(vocab_file_dir)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding='max_length', max_length=args.max_len)


# 定义处理函数
def tokenize_function(example):
    return tokenizer(example["text_a"], example["text_b"], truncation=True, max_length=args.max_len)


tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(["text_a", "text_b"])
tokenized_datasets.rename_column("label", "labels")
test_dataloader = DataLoader(
    tokenized_datasets["test"], shuffle=False, batch_size=args.batch_size, collate_fn=data_collator
)

# 加载模型
logger.info('\n------------loading model------------')
model_param_config = ModelParamConfig()
model = CustomModel(args.model_dir, model_param_config)
model.load_state_dict(torch.load(os.path.join(args.model_dir, 'pytorch_model.bin')))


# 模型指定GPU
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
model.to(device)
model.eval()

# 开始预测
test_y_true = []
test_predictions = []
test_label_ids = []
test_metrics = 0.0
logger.info('\n------------start to predict------------')
with torch.no_grad():
    for step, batch_data in enumerate(test_dataloader):
        for key in batch_data.keys():
            batch_data[key] = batch_data[key].to(device)
        labels = batch_data['labels']
        test_y_true.extend(labels.cpu().numpy())
        logits = model(**batch_data)
        predict_scores = logits.softmax(-1)
        test_label_ids.extend(predict_scores.argmax(dim=1).detach().to("cpu").numpy())
        # 预测为1的概率值
        predict_scores = predict_scores[:, 1]
        test_predictions.extend(predict_scores.detach().cpu().numpy())


# 计算metric
test_f1 = f1_score(test_y_true, test_label_ids, average='macro')
logger.info('test f1: %.8f' %(test_f1))

# 保存预测结果
output_dir = os.path.join(args.output_dir)
if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)
np.savetxt(os.path.join(args.output_dir, 'test_predictions'), np.array(test_predictions).reshape(-1, 1))
np.savetxt(os.path.join(args.output_dir, 'test_label_ids'), np.array(test_label_ids).reshape(-1, 1))