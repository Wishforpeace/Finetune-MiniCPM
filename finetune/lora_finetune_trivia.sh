# 确保日志目录存在
mkdir -p logs

# 通用的模型路径和配置
MODEL_PATH="/home/mnt/wyx/src/Finetune-MiniCPM/MiniCPM-2B-sft-fp32"
DEEPSPEED_CONFIG="configs/ds_config_zero3_offload.json"
# export MASTER_PORT='29501'

# 训练和评估函数
run_training () {
    local data_type=$1
    local precision=$2

    # 生成时间戳和日志文件名
    local formatted_time=$(date +"%Y%m%d%H%M%S")
    echo $formatted_time
    export FORMATTED_TIME=$formatted_time
    local log_file="logs/triviaqa_train_${data_type}_${precision}_${formatted_time}.log"

    # 构建命令
    nohup deepspeed --include localhost:1 finetune.py \
        --model_name_or_path $MODEL_PATH \
        --output_dir output/triviaqa/$formatted_time/ \
        --train_data_path /home/mnt/wyx/src/Finetune-MiniCPM/datasets/processed/triviaqa-rc/${data_type}-train_new.json \
        --eval_data_path /home/mnt/wyx/src/Finetune-MiniCPM/datasets/processed/triviaqa-rc/${data_type}-dev_new.json \
        --learning_rate 5e-5 --per_device_train_batch_size 2 \
        --per_device_eval_batch_size 2 --model_max_length 512 --$precision --use_lora \
        --gradient_accumulation_steps 1 --warmup_steps 100 \
        --max_steps 2000 --weight_decay 0.01 \
        --evaluation_strategy steps --eval_steps 500 \
        --save_strategy steps --save_steps 500 --seed 42 \
        --log_level info --logging_strategy steps --logging_steps 10 \
        --deepspeed $DEEPSPEED_CONFIG > $log_file 2>&1 &

    wait
}

# 执行所有组合的指令
# run_training "web" "fp16"
# run_training "web" "bf16"
run_training "wikipedia" "fp16"
# run_training "wikipedia" "bf16"
