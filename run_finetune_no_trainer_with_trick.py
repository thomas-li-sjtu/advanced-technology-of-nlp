# coding: utf-8
# Copyright (c) Antfin, Inc. All rights reserved.
# @Time : 2021/11/21 下午1:53
# @Author : shuheng.zsh
import os
import time
import random
import argparse
import numpy as np
import logging
from distutils.util import strtobool
from sklearn.metrics import accuracy_score, f1_score
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import BertModel, BertConfig, BertTokenizer, DataCollatorWithPadding, AdamW, \
    get_linear_schedule_with_warmup

from custom_model import ModelParamConfig, CustomModel
from utils.attack import FGM, PGD
from utils.focal_loss import focal_loss
from utils.ema import EMA

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('--max_len', type=int, help='pair文本最大长度', default=32)
parser.add_argument('--batch_size', type=int, help='batch_size', default=512)
parser.add_argument('--pretrained_model_dir', type=str, help='预训练模型地址', default='./pretrained_model/bert')
parser.add_argument('--num_train_epochs', type=int, help='训练epoch数', default=20)
parser.add_argument('--max_steps', type=int, help='最大训练步数，如果设置了，则覆盖num_train_epochs', default=-1)
parser.add_argument('--warmup_ratio', type=float, help='warmup比例', default=0.05)
parser.add_argument('--learning_rate', type=float, help='学习率', default=2e-5)
parser.add_argument('--adam_epsilon', type=float, help='adam_epsilon', default=1e-6)
parser.add_argument('--max_grad_norm', type=int, help='max_grad_norm', default=1)
parser.add_argument('--loss_type', type=str, help='loss类型', default='ce')  # focal loss
parser.add_argument('--focal_loss_alpha', type=float, help='focal_loss_alpha', default=0.25)
parser.add_argument('--focal_loss_gamma', type=float, help='focal_loss_gamma', default=2)
parser.add_argument('--attack_mode', type=str, help='对抗方法，可选fgm, pdg, none', default='none')
parser.add_argument('--attack_eps', type=float, help='fgm_eps', default=1.0)
parser.add_argument('--attack_num', type=int, help='对抗次数，仅限于当attack_model=pgd时生效', default=3)
parser.add_argument('--use_ema', type=lambda x: bool(strtobool(x)), help='是否使用ema', default=False)
parser.add_argument('--ema_decay', type=float, help='ema_decay', default=0.999)
parser.add_argument('--early_stopping_patience', type=int, help='early_stopping_patience', default=3)
parser.add_argument('--save_steps', type=int, help='多少步保存模型', default=100)
parser.add_argument('--save_total_limit', type=int, help='checkpoint数量', default=1)
parser.add_argument('--logging_steps', type=int, help='多少步日志打印', default=100)
parser.add_argument('--output_model_dir', type=str, help='模型保存地址', default='./finetune_saved_model')
parser.add_argument('--tar_name_prefix', type=str, help='最终模型打包名称', default='oppo_finetune_no_trainer_bert')
parser.add_argument('--use_fp16', type=lambda x: bool(strtobool(x)), nargs='?', const=True, help='是否使用fp16',
                    default=True)
parser.add_argument('--run_mode', type=str, help='运行模式normal or test', default='normal')
parser.add_argument('--seed', type=int, help='随机种子', default=2021)
args = parser.parse_args()


def seed_everything(seed):
    '''
    固定随机种子
    :param random_seed: 随机种子数目
    :return:
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def save_model(model):
    output_dir = os.path.join(args.output_model_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    logger.info(f'Saving model to {output_dir}')
    torch.save(model.state_dict(), os.path.join(output_dir, 'pytorch_model.bin'))


# 处理数据
if args.run_mode == 'test':
    data_files = {"train": "data/oppo_train_tiny", "validation": "data/oppo_validation_tiny", "test": "data/oppo_test_tiny"}
else:
    data_files = {"train": "data/oppo_train", "validation": "data/oppo_validation", "test": "data/oppo_test"}
dataset = load_dataset('csv', data_files=data_files, delimiter='\t', quoting=3)

vocab_file_dir = './data/vocab.txt'
tokenizer = BertTokenizer.from_pretrained(vocab_file_dir)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding='max_length', max_length=args.max_len)


def tokenize_function(example):
    return tokenizer(example["text_a"], example["text_b"], truncation=True, max_length=args.max_len)


tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(["text_a", "text_b"])
tokenized_datasets.rename_column("label", "labels")

train_dataloader = DataLoader(
    tokenized_datasets["train"], shuffle=True, batch_size=args.batch_size, collate_fn=data_collator
)
eval_dataloader = DataLoader(
    tokenized_datasets["validation"], batch_size=args.batch_size, collate_fn=data_collator
)

# 模型训练
model_param_config = ModelParamConfig()
# 模型指定GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = CustomModel(args.pretrained_model_dir, model_param_config).to(device)

# 优化器定义
param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
total_steps = args.num_train_epochs * len(train_dataloader)

num_warmup_steps = int(total_steps * args.warmup_ratio)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                            num_training_steps=total_steps)

# 损失函数定义
if args.loss_type == "ce":
    loss_fn = nn.CrossEntropyLoss()
else:  # args.loss_type == focal_loss
    loss_fn = focal_loss(alpha=args.focal_loss_alpha, gamma=args.focal_loss_gamma)

# 对抗方法
fgm, pgd = None, None
if args.attack_mode == 'fgm':
    fgm = FGM(model=model, eps=args.attack_eps)
elif args.attack_mode == 'pgd':
    pgd = PGD(model=model, eps=args.attack_eps)

# 使用ema
if args.use_ema:
    ema = EMA(model, decay=args.ema_decay)
    ema.register()


# 模型训练
global_step = 0
save_steps = total_steps // args.num_train_epochs
eval_steps = save_steps
log_loss_steps = args.logging_steps
avg_loss = 0.

scaler = None
if args.use_fp16:
    scaler = torch.cuda.amp.GradScaler()

best_f1 = 0.
for epoch in range(args.num_train_epochs):
    train_loss = 0.0
    logger.info('\n------------epoch:{}------------'.format(epoch))
    last = time.time()
    for step, batch_data in enumerate(tqdm.tqdm(train_dataloader)):
        model.train()
        batch_data = {k: v.to(device) for k, v in batch_data.items()}

        if args.use_fp16:
            with autocast():
                loss = model(**batch_data, loss_fn=loss_fn)[1]
        else:
            loss = model(**batch_data, loss_fn=loss_fn)[1]

        # 反向传播，得到正常的grad
        if args.use_fp16:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # 对抗
        if fgm is not None:
            # 在embedding上添加对抗扰动
            fgm.attack()
            if args.use_fp16:
                with autocast():
                    loss_adv = model(**batch_data, loss_fn=loss_fn)[1]
            else:
                loss_adv = model(**batch_data, loss_fn=loss_fn)[1]

            # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
            if args.use_fp16:
                scaler.scale(loss_adv).backward()
            else:
                loss_adv.backward()
            # 恢复embedding参数
            fgm.restore()

        elif pgd is not None:
            pgd.backup_grad()
            for _t in range(args.attack_num):
                pgd.attack(is_first_attack=(_t == 0))
                if _t != args.attack_num - 1:
                    model.zero_grad()
                else:
                    pgd.restore_grad()

                if args.use_fp16:
                    with autocast():
                        loss_adv = model(**batch_data, loss_fn=loss_fn)[1]
                else:
                    loss_adv = model(**batch_data, loss_fn=loss_fn)[1]

                if args.use_fp16:
                    scaler.scale(loss_adv).backward()
                else:
                    loss_adv.backward()

            pgd.restore()

        train_loss += loss

        if args.use_fp16:
            scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        # 梯度下降，更新参数
        if args.use_fp16:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        scheduler.step()
        if args.use_ema:
            ema.update()

        model.zero_grad()

        global_step += 1
        if global_step % log_loss_steps == 0:
            avg_loss /= log_loss_steps
            logger.info('Step: %d / %d ----> total loss: %.5f' % (global_step, total_steps, avg_loss))
            avg_loss = 0.
        else:
            avg_loss += loss.item()

    logger.info(f"微调第{epoch}轮耗时：{time.time() - last}")

    if args.use_ema:
        ema.apply_shadow()

    eval_loss = 0
    eval_acc = 0
    y_true = []
    y_predict = []
    y_predict_target = []

    model.eval()
    with torch.no_grad():
        for step, batch_data in enumerate(tqdm.tqdm(eval_dataloader)):
            for key in batch_data.keys():
                batch_data[key] = batch_data[key].to(device)
            labels = batch_data['labels']
            y_true.extend(labels.cpu().numpy())

            logits, loss = model(**batch_data, loss_fn=loss_fn)
            predict_scores = F.softmax(logits)
            y_predict_target.extend(predict_scores.argmax(dim=1).detach().to("cpu").numpy())
            predict_scores = predict_scores[:, 1]
            y_predict.extend(predict_scores.detach().to("cpu").numpy())

            acc = ((logits.argmax(dim=-1) == labels).sum()).item()
            eval_acc += acc / logits.shape[0]
            eval_loss += loss

    eval_loss = eval_loss / len(eval_dataloader)
    eval_acc = eval_acc / len(eval_dataloader)
    eval_f1 = f1_score(y_true, y_predict_target, average='macro')

    if best_f1 < eval_f1:
        early_stop = 0
        best_f1 = eval_f1
        save_model(model)
    else:
        early_stop += 1

    logger.info(
        'epoch: %d, train loss: %.8f, eval loss: %.8f, eval acc: %.8f, eval f1: %.8f, best_f1: %.8f\n' %
        (epoch, train_loss, eval_loss, eval_acc, eval_f1, best_f1))

    torch.cuda.empty_cache()  # 每个epoch结束之后清空显存，防止显存不足
    # 检测早停
    if early_stop >= args.early_stopping_patience:
        break

# 保存词表
os.system("cp %s %s" % (vocab_file_dir, args.output_model_dir))
# 保存config
os.system("cp %s %s" % (os.path.join(args.pretrained_model_dir, 'config.json'), args.output_model_dir))


# # 上传模型
# from utils.upload_model import upload
#
# model_name = args.tar_name_prefix
# upload(args.output_model_dir, model_name)