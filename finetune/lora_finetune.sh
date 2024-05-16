formatted_time=$(date +"%Y%m%d%H%M%S")
echo $formatted_time
export FORMATTED_TIME=$formatted_time

nohup deepspeed --num_gpus 2 finetune.py \
    --model_name_or_path /home/mnt/wyx/src/Finetune-MiniCPM/MiniCPM-2B-sft-bf16 \
    --output_dir output/AdvertiseGenLoRA/$formatted_time/ \
    --train_data_path data/AdvertiseGenChatML/train.json \
    --eval_data_path data/AdvertiseGenChatML/dev.json \
    --learning_rate 5e-5 --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 --model_max_length 384 --bf16 --use_lora \
    --gradient_accumulation_steps 1 --warmup_steps 100 \
    --max_steps 300 --weight_decay 0.01 \
    --evaluation_strategy steps --eval_steps 500 \
    --save_strategy steps --save_steps 500 --seed 42 \
    --log_level info --logging_strategy steps --logging_steps 10 \
    --deepspeed configs/ds_config_zero3_offload.json > logs/train_$formatted_time.log 2>&1 &


# deepspeed --num_gpus 2 finetune.py \
#     --model_name_or_path /home/mnt/wyx/src/Finetune-MiniCPM/MiniCPM-2B-sft-bf16 \
#     --output_dir output/AdvertiseGenLoRA/$formatted_time/ \
#     --train_data_path data/AdvertiseGenChatML/train.json \
#     --eval_data_path data/AdvertiseGenChatML/dev.json \
#     --learning_rate 5e-5 --per_device_train_batch_size 4 \
#     --per_device_eval_batch_size 4 --model_max_length 384 --bf16 --use_lora \
#     --gradient_accumulation_steps 1 --warmup_steps 100 \
#     --max_steps 3 --weight_decay 0.01 \
#     --evaluation_strategy steps --eval_steps 500 \
#     --save_strategy steps --save_steps 500 --seed 42 \
#     --log_level info --logging_strategy steps --logging_steps 10 \
#     --deepspeed configs/ds_config_zero3_offload.json > logs/train_$formatted_time.log 2>&1 &
