import csv
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import torch
from tqdm import tqdm
import sys
import re

# 检查CUDA是否可用，并设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 加载模型和tokenizer，将模型移动到GPU
model_path = "/home/mnt/wyx/src/Finetune-MiniCPM/finetune/output/cosmosqa/20240515212751/"
model = AutoModelForCausalLM.from_pretrained(model_path,trust_remote_code=True).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_path,trust_remote_code=True)

# 读取JSON文件
with open('/home/mnt/wyx/src/Finetune-MiniCPM/datasets/processed/cosmosqa/test.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

def extract_ai_response(output):
    # 查找 '<AI>' 标签，并提取其后的内容
    tag = '<AI>'
    start_index = output.find(tag)
    if start_index != -1:
        # 提取标签后的内容
        content_after_tag = output[start_index + len(tag):].strip()
        # 使用正则表达式提取第一个出现的数字
        match = re.findall(r'\d+', content_after_tag)
        return match[0] if match else None  # 返回第一个匹配的数字，如果没有则返回None
    return None  # 如果没有找到标签或数字，返回None


def batch_inference(data, model, tokenizer, device, batch_size=4):
    predictions = []
    # tqdm包裹循环，提供进度条
    for i in tqdm(range(0, len(data), batch_size), desc="Processing batches"):
        batch = data[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        outputs = model.generate(input_ids, attention_mask=attention_mask, max_length=384, top_p=0.5, eos_token_id=2, pad_token_id=2)
        for output in outputs:
            result_text = tokenizer.decode(output, skip_special_tokens=True)
            number = extract_ai_response(result_text)  # 提取并处理输出文本以获取数字
            if number:  # 只有当找到数字时才添加到结果列表
                predictions.append(number)
    return predictions


# 进行批量推理
predictions = batch_inference(data, model, tokenizer, device)

# 保存结果到prediction.lst文件
with open('prediction.lst', 'w', encoding='utf-8') as file:
    for prediction in predictions:
        file.write(prediction + '\n')  # 每个数字单独一行

print("Predictions have been saved to 'prediction.lst'")