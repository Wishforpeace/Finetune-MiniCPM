import csv
import json
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import torch.distributed as dist
import os
import sys

# 检查CUDA是否可用，并设置设备
torch.cuda.is_available()
print("Using GPUs:", torch.cuda.device_count())

# 初始化分布式环境
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

# 清理分布式环境
def cleanup():
    dist.destroy_process_group()

# 加载模型和tokenizer，将模型移动到指定的GPU
def load_model(device, model_path):
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    return model, tokenizer

# 分布式批量推理函数
def batch_inference(rank, world_size, data, model_path, batch_size=4):
    setup(rank, world_size)
    device = torch.device("cuda", rank)
    model, tokenizer = load_model(device, model_path)
    model = DDP(model, device_ids=[rank])

    # 数据分片处理
    start = rank * len(data) // world_size
    end = (rank + 1) * len(data) // world_size
    subset = data[start:end]

    predictions = []
    for i in tqdm(range(0, len(subset), batch_size), desc=f"Processing batches on GPU {rank}"):
        batch = subset[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        outputs = model.module.generate(input_ids, attention_mask=attention_mask, max_length=384, top_p=0.5, eos_token_id=2, pad_token_id=2)
        for output in outputs:
            result_text = tokenizer.decode(output, skip_special_tokens=True)
            number = extract_ai_response(result_text)  # 提取并处理输出文本以获取数字
            if number:  # 只有当找到数字时才添加到结果列表
                predictions.append(number)

    if rank == 0:
        with open('prediction.lst', 'w', encoding='utf-8') as file:
            for prediction in predictions:
                file.write(prediction + '\n')  # 每个数字单独一行
        print("Predictions have been saved to 'prediction.lst'")

    cleanup()

# 运行分布式推理
if __name__ == "__main__":
    rank = int(sys.argv[1])
    world_size = torch.cuda.device_count()
    with open('/home/mnt/wyx/src/Finetune-MiniCPM/datasets/processed/cosmosqa/test.json', 'r', encoding='utf-8') as file:
        data = json.load(file)
    batch_inference(rank, world_size, data, "/home/mnt/wyx/src/Finetune-MiniCPM/finetune/output/cosmosqa/20240515212751/")
