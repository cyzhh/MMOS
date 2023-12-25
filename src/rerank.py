import json
import os
import re
import random
import io
import shutil
import sys
import argparse
from collections import defaultdict

from sympy import symbols, Eq, solve
from collections import OrderedDict

def save_filtered_data(source_file_path, target_file_path):
    count = 0
    dedup = 0
    with open(target_file_path, 'w') as f:
        f.write("\n")
    with open(source_file_path, 'r', encoding='utf-8') as src_file, \
            open(target_file_path, 'w', encoding='utf-8') as tgt_file:
        for line in src_file:
            item = json.loads(line)
            codes = item['code']
            scores = item['score']
            for i, score in enumerate(scores):
                full_prompt = f"<|user|>\n{item['question']}\n<|assistant|>\n"
                # 找到 score 中所有的true case
                extracted_codes = []
                if score == True:
                    # 取出item所有的code
                    code = codes[i]
                    extracted_codes.append(code)

                count += len(extracted_codes)
                # 去重 简单版本

                extracted_codes = dedup_list(extracted_codes)
                for extracted_code in extracted_codes:
                    extracted_data = {
                        "idx": dedup,
                        "prompt": full_prompt,
                        "completion": extracted_code,
                        "source": item["idx"]
                    }
                    tgt_file.write(json.dumps(extracted_data, ensure_ascii=False) + '\n')
                    dedup += 1

        print(f'total count:{count}')
        print(f'total dedup:{dedup}')


def process_file(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f, open(output_file, 'w', encoding='utf-8') as out:
        for line in f:
            data = json.loads(line)

            # 更新question
            full_prompt = f"<|user|>\n{data['question']}\n<|assistant|>\n"
            data['prompt'] = full_prompt
            data['completion'] = data['code'][0]

            # 写回文件
            out.write(json.dumps(data, ensure_ascii=False) + '\n')


def extract_fields(input_file_path, output_file_path):
    with open(input_file_path, 'r', encoding='utf-8') as f_in, open(output_file_path, 'w', encoding='utf-8') as f_out:
        for i, line in enumerate(f_in):
            data = json.loads(line)

            # 提取需要的字段
            extracted_data = {
                "idx": i,
                "prompt": data["prompt"],
                "completion": data["completion"]
            }

            # 将提取出来的数据写入新文件
            json.dump(extracted_data, f_out, ensure_ascii=False)
            f_out.write('\n')


def show_example(file_path, count=-1):
    with open(file_path, 'r', encoding='utf-8') as file:
        for i, line in enumerate(file):
            continue

        print(f'Show case {file_path} Total num {i+1}')
    lines = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for i, line in enumerate(file):
            if count == -1 or i > count-1:
                break
            # print(line)
            lines.append(json.loads(line))
            print(json.loads(line))

        return lines


def move_key_to_first(dictionary, key):
    if key not in dictionary:
        raise KeyError(f"{key} not found in dictionary")
    ordered_dict = OrderedDict(dictionary)
    ordered_dict.move_to_end(key, last=False)
    return ordered_dict


def random_select(input_file_path, output_file, k,specified_column = True, seed=0):
    input_file_path = input_file_path[0]
    # 读取数据
    file_data = []
    with open(input_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            file_data.append(data)

    # 随机打乱数据
    random.seed(seed)
    random.shuffle(file_data)

    # 根据source分组
    source_data = defaultdict(list)
    for data in file_data:
        source = data['source']
        source_data[source].append(data)

    # 对每个source进行随机选择
    random_selected_data = {source: data[:min(k, len(data))] for source, data in source_data.items()}


    with open(output_file, 'w', encoding='utf-8') as out_f:
        idx = 0
        for source, data_list in random_selected_data.items():
            for data in data_list:
                specified_data = {}
                if specified_column:
                    data['idx'] = idx
                    specified_data['idx'] = data['idx']
                    specified_data['prompt'] = data['prompt']
                    specified_data['completion'] = data['completion']

                else:
                    specified_data = data
                json.dump(move_key_to_first(specified_data, 'idx'), out_f)
                out_f.write('\n')
                idx += 1


def merge_jsonl(files, rate, output_file,specified_column=True):


    all_data = []

    # 遍历所有文件
    for i,file in enumerate(files):
        file_data = []
        if os.path.exists(file) and os.path.isfile(file):
            with open(file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        data['file'] = i
                        file_data.append(data)
                    except json.JSONDecodeError:
                        print(f"Warning: Could not decode line in file {file}. Skipping...")
        else:
            print(f"Warning: File {file} does not exist or is not a file. Skipping...")


        if rate[i] > 1:
            print('Rate > 1 Take it as sample number')
            sample_number = rate[i]

        else:
            sample_number = int(rate[i]*len(file_data))

        random.seed(0)
        random.shuffle(file_data)
        file_data_rate = file_data[:sample_number]

        all_data += file_data_rate
        print(f'input file {i}: {file} Total data {len(file_data)} Rate {rate[i]}, Add data {len(file_data_rate)}')

    # 重新设置idx并写入新文件
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for idx, data in enumerate(all_data):
            data['idx'] = idx

            specified_data = {}
            if specified_column:
                specified_data['idx'] = data['idx']
                specified_data['prompt'] = data['prompt']
                specified_data['completion'] = data['completion']

            else:
                specified_data = data
            json.dump(move_key_to_first(specified_data, 'idx'), out_f)
            out_f.write('\n')

    print(f"Merge completed. Total data num {idx+1}. Data written to {output_file}")

def main(task, data, source_files, rate, rs, k, seed):
    dedup_dir = f'./{("train_data" if task == "train" else "data")}/{data}/'
    os.makedirs(dedup_dir, exist_ok=True)

    dedup_output_file_path = os.path.join(dedup_dir, f'{task}.jsonl')

    if task == 'train':
        if rs:
            random_select(source_files, dedup_output_file_path, k=k, specified_column=True, seed=seed)
        else:
            merge_jsonl(source_files, rate, dedup_output_file_path, specified_column=True)
    elif task == 'test':
        merge_jsonl(source_files, rate, dedup_output_file_path)

    show_example(dedup_output_file_path, 0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='create')
    parser.add_argument("--task", type=str, choices=['train', 'test'], required=True, help="Task to perform: 'train' or 'test'.")
    parser.add_argument("--data", type=str, required=True, help="Data folder name.")
    parser.add_argument("--source_files", nargs='+', required=True, help="List of source file paths to merge.")
    parser.add_argument("--rate", nargs='+', type=int, required=True, help="List of rates for each source file.")
    parser.add_argument('--rs', action="store_true", help="Random select flag.")
    parser.add_argument('--k', type=int, default=9, help="Number of items to select if random select is enabled.")
    parser.add_argument('--seed', type=int, default=42, help="Seed for random selection.")

    args = parser.parse_args()

    main(args.task, args.data, args.source_files, args.rate, args.rs, args.k, args.seed)