python run_finetune_no_trainer.py \
  --run_mode test \
  --batch_size 128 \
  --num_train_epochs 2 \
  --pretrained_model_dir ./pretrained_model/bert/ \
  --output_model_dir ./finetune_on_google_bert_no_trainer \
  --tar_name_prefix oppo_finetune_on_google_bert_no_trainer