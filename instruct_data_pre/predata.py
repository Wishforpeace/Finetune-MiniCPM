import json
import os
import concurrent.futures
from tqdm import tqdm  # 引入tqdm

# 定义读取和处理每个文件的函数
def process_json_file(file_path, evidence_path, type):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        data_out_list = []

        # 使用tqdm显示处理进度
        for entry in data.get("Data", []):
            input_text = f"This is a reading comprehension task. Given the context:\n"
            output_text = entry["Answer"]["Value"]
            question = entry["Question"]
            key = "SearchResults" if type == "web" else "EntityPages"

            if key in entry:
                # 处理每个 SearchResult 或 EntityPage
                for index, result in enumerate(entry[key]):
                    filename = result.get("Filename")
                    e_path = os.path.join(evidence_path, filename)  
                    with open(e_path, 'r', encoding='utf-8') as file:
                        content = file.read()
                        content = f"Reading Material {index}: {content}\n"
                        input_text += content
            
            data_out = {
                "messages": [
                    {
                        "role": "user",
                        "content": input_text,
                        "question": question,
                    },
                    {
                        "role": "assistant",
                        "content": output_text,
                    },
                ]
            }
            data_out_list.append(data_out)

    output_path = '/home/mnt/wyx/src/Finetune-MiniCPM/datasets/processed/triviaqa-rc'
    dataset_name = f'{type}-train' if 'train' in file_path else f'{type}-dev'
    output_file_path = os.path.join(output_path, f'{dataset_name}.json')

    # 保存处理后的数据到 JSON 文件
    with open(output_file_path, 'w', encoding='utf-8') as fo:
        json.dump(data_out_list, fo, ensure_ascii=False, indent=4)

    print(f"{dataset_name} processed and saved.")

def process_file(filename):
    file_path = os.path.join(directory_path, filename)
    print(file_path)
    if 'web' in filename:
        evidence_path = "/home/mnt/wyx/src/Finetune-MiniCPM/datasets/raw/triviaqa-rc/evidence/web"
        process_json_file(file_path, evidence_path, 'web')
    elif 'wikipedia' in filename:
        evidence_path = '/home/mnt/wyx/src/Finetune-MiniCPM/datasets/raw/triviaqa-rc/evidence/wikipedia'
        process_json_file(file_path, evidence_path, 'wikipedia')

directory_path = '/home/mnt/wyx/src/Finetune-MiniCPM/datasets/raw/triviaqa-rc/qa'
filtered_files = ['web-train.json', 'wikipedia-dev.json', 'web-dev.json', 'wikipedia-train.json']

# 使用concurrent.futures进行并行处理
with concurrent.futures.ThreadPoolExecutor() as executor:
    executor.map(process_file, filtered_files)
