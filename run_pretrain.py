# coding: utf-8
# Copyright (c) Antfin, Inc. All rights reserved.
# @Time : 2021/11/19 下午3:45
# @Author : shuheng.zsh
import os
import argparse
from datasets import load_dataset, GenerateMode
from transformers.trainer_utils import SchedulerType
from transformers import BertTokenizer, BertConfig, BertForMaskedLM, DataCollatorForLanguageModeling, TrainingArguments, \
    Trainer

# disable wandb
os.environ["WANDB_DISABLED"] = "true"

parser = argparse.ArgumentParser()
parser.add_argument('--max_len', type=int, help='pair文本最大长度', default=32)
parser.add_argument('--batch_size', type=int, help='batch_size', default=4)
parser.add_argument('--pretrained_model_dir', type=str, help='预训练模型地址', default='./pretrained_model/bert')
parser.add_argument('--mlm_probability', type=float, help='mask概率', default=0.15)
parser.add_argument('--num_train_epochs', type=int, help='训练epoch数', default=100)
parser.add_argument('--max_steps', type=int, help='最大训练步数，如果设置了，则覆盖num_train_epochs', default=-1)
parser.add_argument('--warmup_ratio', type=float, help='warmup比例', default=0.05)
parser.add_argument('--learning_rate', type=float, help='学习率', default=5e-5)
parser.add_argument('--save_strategy', type=str, help='保存模型策略', default='epoch')
parser.add_argument('--save_steps', type=int, help='多少步保存模型，如果save_strategy=epoch，则失效', default=100)
parser.add_argument('--save_total_limit', type=int, help='checkpoint数量', default=1)
parser.add_argument('--logging_steps', type=int, help='多少步日志打印', default=100)
parser.add_argument('--output_model_dir', type=str, help='模型保存地址', default='./pretrain_saved_model')
parser.add_argument('--tar_name_prefix', type=str, help='最终模型打包名称', default='oppo_pretrained_bert')
parser.add_argument('--run_mode', type=str, help='运行模式normal or test', default='test')
parser.add_argument('--seed', type=int, help='随机种子', default=2021)
args = parser.parse_args()

# 读取数据
if args.run_mode == 'test':
    dataset = load_dataset('csv',
                           data_files=['data/oppo_train_tiny', 'data/oppo_validation_tiny', 'data/oppo_test_tiny'],
                           delimiter='\t',
                           quoting=3)
else:
    dataset = load_dataset('csv', data_files=['data/oppo_train', 'data/oppo_validation', 'data/oppo_test'],
                           delimiter='\t',
                           quoting=3)

# 加载词表
vocab_file_dir = './data/vocab.txt'
tokenizer = BertTokenizer.from_pretrained(vocab_file_dir)


# 定义处理函数
def tokenize_function(example):
    return tokenizer(example["text_a"], example["text_b"], truncation=True, padding="max_length",
                     max_length=args.max_len, )


# tokenize数据
tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=['label'])

# MLM
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=args.mlm_probability)

# 加载模型
model = BertForMaskedLM.from_pretrained(args.pretrained_model_dir)

# 修改embedding大小， 本教程中可以不修改
model.resize_token_embeddings(len(tokenizer))

# 配置训练参数
training_args = TrainingArguments(
    num_train_epochs=args.num_train_epochs,
    max_steps=args.max_steps,
    per_device_train_batch_size=args.batch_size,
    learning_rate=args.learning_rate,
    lr_scheduler_type=SchedulerType.LINEAR,
    warmup_ratio=args.warmup_ratio,
    output_dir=args.output_model_dir,
    overwrite_output_dir=True,
    save_strategy=args.save_strategy,
    save_total_limit=args.save_total_limit,
    logging_steps=args.logging_steps,
    logging_first_step=True,
    seed=args.seed,
)

# 从trainer的第1115行以及651行代码：train_dataloader = self.get_train_dataloader()，train_sampler = self._get_train_sampler()，可以看出内部做了shuffle
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets['train'],
)

# 训练模型
trainer.train()
trainer.save_model(args.output_model_dir)
trainer.state.save_to_json(os.path.join(training_args.output_dir, "trainer_state.json"))
# 保存词表
os.system("cp %s %s" % (vocab_file_dir, training_args.output_dir))
