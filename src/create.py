import json
import os
import copy
import argparse
import timeout_decorator
import sys
import io
from tqdm import tqdm
import re
from sympy import symbols, Eq, solve, Rational, sympify
from contextlib import redirect_stdout
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed


def run_python_code(python_code):
    @timeout_decorator.timeout(5)
    def execute():
        program_io = io.StringIO()
        with redirect_stdout(program_io):
            exec(python_code, globals())
        program_io.seek(0)
        result = program_io.readlines()[-1]
        return result
    
    try:
        return execute()
    except Exception as e:
        print(e)
        return None


def filter_run_false(code):
    try:
        code_part, _ = code.split('```output\n', 1)
        substring = '```\n```output'
        code = code.split(substring)[0]
        python_code = code.strip('```python\n')
        # print(python_code)
        output = run_python_code(python_code)
        return output
    except Exception as e:
        return None


def extract_wrong_data(source_file_path, target_file_path):
    with open(target_file_path, 'w') as f:
        f.write("\n")
    with open(source_file_path, 'r', encoding='utf-8') as src_file, \
         open(target_file_path, 'w', encoding='utf-8') as tgt_file:
        for line in src_file:
            item = json.loads(line)
            for i, score in enumerate(item['score']):
                # 找到 score 中所有的wrong case
                if score != True:
                    extracted_data = copy.deepcopy(item)
                    extracted_data['code'] = item['code'][i]
                    extracted_data['pred'] = item['pred'][i]
                    extracted_data['source'] = item['idx'] if 'source' not in item.keys() else item['source']
                    keys = extracted_data.keys()
                    for key in ['report','score']:
                        if key in keys:
                            extracted_data.pop(key)


                    tgt_file.write(json.dumps(extracted_data, ensure_ascii=False) + '\n')

        print(f"Data written to {target_file_path}")



def extract_true_data(source_file_path, target_file_path):
    count = 0
    true_error = 0

    datas = []
    with open(source_file_path, 'r', encoding='utf-8') as src_file:
        for line in src_file:
            item = json.loads(line)
            datas.append(item)


    with open(target_file_path, 'w') as f:
        f.write("\n")
    with open(source_file_path, 'r', encoding='utf-8') as src_file, \
         open(target_file_path, 'w', encoding='utf-8') as tgt_file:
        for item in tqdm(datas):
            for i, score in enumerate(item['score']):
                # 找到 score 中所有的wrong case
                if score:
                    if not filter_run_false(item['code'][i]):
                        true_error += 1
                        continue
                    extracted_data = copy.deepcopy(item)
                    extracted_data['code'] = item['code'][i]
                    extracted_data['pred'] = item['pred'][i]
                    extracted_data["source"] = item["idx"] if "source" not in item.keys() else item["source"]
                    count += 1

                    keys = extracted_data.keys()
                    for key in ['report','score']:
                        if key in keys:
                            extracted_data.pop(key)

                    tgt_file.write(json.dumps(extracted_data, ensure_ascii=False) + '\n')

        print(f'total count: {count}')
        print(f'total error: {true_error}')
        print(f"Data written to {target_file_path}")




# def extract_true_train_data(source_file_path, target_file_path):
#     count = 0
#     true_error = 0

#     datas = []
#     with open(source_file_path, 'r', encoding='utf-8') as src_file:
#         for line in src_file:
#             item = json.loads(line)
#             datas.append(item)

#     with open(target_file_path, 'w') as f:
#         f.write("\n")
#     with open(source_file_path, 'r', encoding='utf-8') as src_file, \
#          open(target_file_path, 'w', encoding='utf-8') as tgt_file:
#         for item in tqdm(datas):
#             codes = item['code']
#             scores = item['score']
#             full_prompt = f"<|user|>\n{item['question']}\n<|assistant|>\n"
#             for i, score in enumerate(scores):
#                 # 找到 score 中所有的true case
#                 extracted_codes = []
#                 if score:
#                     if not filter_run_false(item['code'][i]):
#                         true_error += 1
#                         continue

#                     # 取出item所有的code
#                     code = codes[i]
#                     extracted_codes.append(code)

#                 count += len(extracted_codes)
#                 for extracted_code in extracted_codes:
#                     extracted_data = {
#                         # "idx": dedup,
#                         "prompt": full_prompt,
#                         "completion": extracted_code
#                     }
#                     extracted_data["source"] = item["idx"] if "source" not in item.keys() else item["source"]



#                     tgt_file.write(json.dumps(extracted_data, ensure_ascii=False) + '\n')

#         print(f'true_error: {true_error}')
#         print(f'total count: {count}')
#         print(f"Data written to {target_file_path}")

def extract_true_train_data(source_file_path, target_file_path):
    count = 0
    true_error = 0

    with open(source_file_path, 'r', encoding='utf-8') as src_file:
        datas = [json.loads(line) for line in src_file]

    # 创建一个字典，将 future 与其对应的 item 和 index 关联起来
    futures_dict = {}
    with ProcessPoolExecutor() as executor:
        for item in datas:
            for i, score in enumerate(item['score']):
                if score:
                    future = executor.submit(filter_run_false, item['code'][i])
                    futures_dict[future] = (item, i)

        # 使用 tqdm 来跟踪完成的任务
        with open(target_file_path, 'w', encoding='utf-8') as tgt_file:
            for future in tqdm(as_completed(futures_dict), total=len(futures_dict)):
                item, i = futures_dict[future]
                try:
                    # 设置超时时间，例如 1 秒
                    if not future.result(timeout=5):
                        true_error += 1
                        continue

                    # 正常处理代码
                    extracted_code = item['code'][i]
                    extracted_data = {
                        "prompt": f"<|user|>\n{item['question']}\n<|assistant|>\n",
                        "completion": extracted_code
                    }
                    extracted_data["source"] = item.get("source", item["idx"])
                    tgt_file.write(json.dumps(extracted_data, ensure_ascii=False) + '\n')
                    count += 1
                except TimeoutError:
                    true_error += 1
                    continue
                except Exception as exc:
                    print(f'Error processing item {item["idx"]}: {exc}')
                    continue

    print(f'true_error: {true_error}')
    print(f'total count: {count}')
    print(f"Data written to {target_file_path}")


def construct_wrong_data(source_file_path, target_file_path):
    with open(target_file_path, 'w') as f:
        f.write("\n")
    with open(source_file_path, 'r', encoding='utf-8') as src_file, \
         open(target_file_path, 'w', encoding='utf-8') as tgt_file:
        for line in src_file:

            item = json.loads(line)
            codes = item['code']
            full_prompt = f"<|user|>\n{item['question']}\n<|assistant|>\n"

            line_codes = codes.split('\n')
            line_codes = [line_code +'\n' for line_code in line_codes]
            for i in range(len(line_codes)-1):
                question = full_prompt
                for j in range(i):
                    question += line_codes[j]
                extracted_data = {
                    "question": question,
                    "gt": item["gt"],
                    "gt_cot": item["gt_cot"],
                    "source": item["source"]
                }
                tgt_file.write(json.dumps(extracted_data, ensure_ascii=False) + '\n')
    print(f"Data written to {target_file_path}")


if __name__ == "__main__":
    from .rerank import show_example

    parser = argparse.ArgumentParser(description='create')
    parser.add_argument(
        "--task",
        type=str,
        default='extract_wrong',
        help="The name of the dataset to use (via the datasets library).",
    )
    args = parser.parse_args()


    data = 'gsm8k_wrong'
    mid = 'u400'
    dir = f'/root/ToRA/src/data/{data}/'
    os.makedirs(dir, exist_ok=True)

    source_file_path = '/root/ToRA/src/data/gsm8k_u400/combine.jsonl'


    wrong_file_path = dir + mid + '_wrong.jsonl'
    nodup_file_path = dir + mid + '_wrong_nodup.jsonl'
    output_file_path = dir + mid + '_construct_wrong.jsonl'


    if args.task == 'extract_wrong':
        extract_wrong_data(source_file_path, wrong_file_path)
        show_example(wrong_file_path,0)

    if args.task == 'construct_wrong':
        show_example(nodup_file_path,0)

        construct_wrong_data(nodup_file_path,output_file_path)
        show_example(output_file_path,0)




