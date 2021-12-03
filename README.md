# 1. 环境准备
1. 建议使用miniconda创建虚拟环境：https://docs.conda.io/en/latest/miniconda.html
2. 创建虚拟环境oppo: conda create -n oppo python=3.7
3. 激活环境：conda activate oppo
4. 安装torch(mac os): pip install torch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
5. 安装torch(windows): pip install torch==1.8.0+cpu torchvision==0.9.0+cpu torchaudio==0.8.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
6. 安装额外包：
    7. pip install transformers -i https://pypi.tuna.tsinghua.edu.cn/simple
    8. pip install datasets -i https://pypi.tuna.tsinghua.edu.cn/simple
    9. pip install sklearn -i https://pypi.tuna.tsinghua.edu.cn/simple

# 2. 测试
## 2.1 预训练模型
1. mac：sh run_pretrain.sh
2. windows：python run_pretrain.py \
  --num_train_epochs 2 \
  --batch_size 128 \
  --output_model_dir ./pretrain_bert_model \
  --tar_name_prefix oppo_pretrained_on_google_bert

## 2.2 使用trainer finetune模型
## 2.2.1 从头开始finetune
1. mac: sh run_finetune.sh
2. windows：python run_finetune.py \
  --num_train_epochs 2 \
  --batch_size 128 \
  --pretrained_model_dir ./pretrained_model/bert \
  --output_model_dir finetune_on_google_bert_with_trainer \
  --tar_name_prefix oppo_finetune_on_google_bert_with_trainer
## 2.2.2 加载2.1预训练模型finetune
1. python run_finetune.py \
  --num_train_epochs 2 \
  --batch_size 128 \
  --pretrained_model_dir ./pretrained_model/pretrained_bert_on_oppo_100epoch \
  --output_model_dir finetune_on_pretrained_bert_with_trainer \
  --tar_name_prefix oppo_finetune_on_pretrained_bert_with_trainer

## 2.3 不使用trainer finetune模型
## 2.3.1 从头开始finetune
1. mac: sh run_finetune_no_trainer.sh
2. windows：python run_finetune_no_trainer.py \
  --num_train_epochs 2 \
  --batch_size 128 \
  --pretrained_model_dir ./pretrained_model/bert/ \
  --output_model_dir ./finetune_on_google_bert_no_trainer \
  --tar_name_prefix oppo_finetune_on_google_bert_no_trainer
## 2.3.2 加载2.1预训练模型finetune
1. python run_finetune_no_trainer.py \
  --num_train_epochs 2 \
  --batch_size 128 \
  --pretrained_model_dir ./pretrained_model/pretrained_bert_on_oppo_100epoch \
  --output_model_dir finetune_on_pretrained_bert_no_trainer \
  --tar_name_prefix oppo_finetune_on_pretrained_bert_no_trainer

## 2.4 使用模型进行预测
1. mac: sh run_predict.sh
2. windows：python run_predict.py \
  --batch_size 128 \
  --model_dir finetune_on_pretrained_bert_no_trainer