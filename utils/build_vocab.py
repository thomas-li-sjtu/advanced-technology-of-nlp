# coding: utf-8
# Copyright (c) Antfin, Inc. All rights reserved.
# @Time : 2021/11/19 下午4:24
# @Author : shuheng.zsh
from datasets import load_dataset
from collections import Counter
dataset = load_dataset('csv', data_files=['../data/oppo_train', '../data/oppo_validation', '../data/oppo_test'], delimiter='\t', quoting=3)
all_text = []
token_count = Counter()
token_min_freq = 3
for data in dataset['train']:
    token_count.update(data['text_a'].split(' '))
    token_count.update(data['text_b'].split(' '))


# 整理词表
prev = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']
tail = []
for k, v in token_count.items():
    if v >= token_min_freq:
        tail.append(k)
vocab = prev + tail

# 写入词表
with open('../data/vocab.txt', "w", encoding="utf-8") as f:
    for i in vocab:
        f.write(str(i) + '\n')