import csv
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import torch
from tqdm import tqdm
import sys
import re

# 检查CUDA是否可用，并设置设备
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print("Using device:", device)

# 加载模型和tokenizer，将模型移动到GPU
model_path = "/hy-tmp/Finetune-MiniCPM/finetune/output/triviaqa/fp32_web_20240516204754"
model = AutoModelForCausalLM.from_pretrained(model_path,trust_remote_code=True).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_path,trust_remote_code=True)

# 读取JSON文件
with open('/hy-tmp/Finetune-MiniCPM/datasets/processed/verified-web-dev.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

def extract_ai_response(output):
    # 查找 '<AI>' 标签，并提取其后的内容
    tag = '<AI>'
    start_index = output.find(tag)
    if start_index != -1:
        # 提取标签后的所有内容
        content_after_tag = output[start_index + len(tag):].strip()
        return content_after_tag  # 返回标签后的全部内容
    return None  # 如果没有找到标签，返回None

def batch_inference(data, model, tokenizer, device, batch_size=4):
    predictions = {}
    # tqdm包裹循环，提供进度条
    for i in tqdm(range(0, len(data), batch_size), desc="Processing batches"):
        batch = data[i:i+batch_size]
        texts = [item['question'] for item in batch]
        labels = [item['question_label'] for item in batch]
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=4096)
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        outputs = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=100,  top_p=0.5, eos_token_id=2, pad_token_id=2)
        print(outputs)
        for label, output in zip(labels, outputs):
            result_text = tokenizer.decode(output, skip_special_tokens=True)
            result = extract_ai_response(result_text)  # 提取并处理输出文本以获取标签后的内容
            predictions[label] = result
    return predictions


# 进行批量推理
predictions = batch_inference(data, model, tokenizer, device)


with open('triviaqa_results.json', 'w', encoding='utf-8') as file:
    json.dump(predictions, file, ensure_ascii=False, indent=4)
    
print("Predictions have been saved to 'triviaqa_results.json'")