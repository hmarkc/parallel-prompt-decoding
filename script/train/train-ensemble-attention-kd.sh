#! /bin/bash
accelerate launch --num_processes 4 prompt/train/train.py --model_name_or_path lmsys/vicuna-7b-v1.3 \
    --dataset_path "./ShareGPT_training_dataset_2_distillation.pt" \
    --output_dir test/ \
    --num_train_epochs 1 \
    --save_steps 500 \
    --model_max_length 2048 \
    --num_special_tokens 3 \
    --virtual_tokens_per_special_token 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --learning_rate 1e-2 \
    --weight_decay 0.0 \
    --warmup_ratio 0.0 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --load_in_4bit \
    --vt_attention_type "ensemble" \
    --trainer_type "distillation_trainer"
    # --use_prefix_tuning \
    # --prefix_virtual_tokens 10 \
    # --size 100
    # --tf32 True \ requires at least Ampere
    # 2048 length not working for batch 1
