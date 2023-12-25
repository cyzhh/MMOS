import shutil
import json
import re
import string
import os
import ast
import astunparse
import itertools
from collections import Counter, defaultdict
import difflib
import Levenshtein
import textwrap
import argparse
import traceback
from .utils import show_example
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from .utils import fixed_code, merge_jsonl, build_paths, build_greedy_path_for_source, read_data


def hello():
    print('hello')


def add_idx(input_file_path, output_file_path):
    print('*' * 50, 'add_idx')

    with open(input_file_path, 'r') as input_file, open(output_file_path, 'w') as output_file:
        idx = 0
        for line in input_file:
            # print(idx)
            data = json.loads(line.strip())
            # print(data)

            output_data = data
            if 'source' not in output_data.keys():
                output_data['source'] = output_data['idx']
            output_data['idx'] = idx
            json.dump(output_data, output_file, ensure_ascii=False)
            output_file.write('\n')
            idx += 1

def process_data(input_file_path, output_file_path, args):
    count = 0
    print('*' * 50, 'process_data')
    datas = []
    with open(input_file_path, 'r') as input_file:
        for line in input_file:
            data = json.loads(line)
            datas.append(data)
    with open(output_file_path, 'w') as output_file:
        for data in tqdm(datas):
            completion = data[args.key].strip()
            updated_completion = re.sub(r'(```output[\s\S]+?\n```)[\s\S]+', r'\1', completion)
            completion = updated_completion
            completion = replace_var_names_in_code_blocks(completion, args)
            if completion != "1":
                output_data = {'idx': data['idx'],
                               args.key: completion,
                               'source': data['source']}
                json.dump(output_data, output_file, ensure_ascii=False)
                output_file.write('\n')
            else:
                count += 1
        print(count)

def remove_duplicates(input_file_path, output_file_path, args):
    print('*' * 50, 'remove_duplicates')
    seen_combinations = {}
    with open(input_file_path, 'r') as input_file, open(output_file_path, 'w') as output_file:
        for line in input_file:
            data = json.loads(line)
            completion = data[args.key].strip()
            source = data['source']
            if source is not None:
                if source not in seen_combinations:
                    seen_combinations[source] = set()
                if completion not in seen_combinations[source]:
                    seen_combinations[source].add(completion)
                    output_file.write(json.dumps(data, ensure_ascii=False) + '\n')

def extract_matching_data(input_file_path, target_file_path, output_file_path):
    print('*' * 50, 'extract_matching_data')
    # 读取target文件，提取所有的idx值并存储在集合中
    target_idxs = set()
    with open(target_file_path, 'r') as target_file:
        for line in target_file:
            data = json.loads(line.strip())
            idx = data['idx']
            if idx is not None:
                target_idxs.add(idx)

    with open(input_file_path, 'r') as input_file, open(output_file_path, 'w') as output_file:
        for line in input_file:
            data = json.loads(line.strip())
            idx = data['idx']
            if idx in target_idxs:
                output_data = data
                json.dump(output_data, output_file)
                output_file.write('\n')

def find_most_common_completion(input_file_path, output_file_path):
    # Step 1: Read the original file
    with open(input_file_path, 'r') as input_file:
        # Parse and count
        completion_counts = defaultdict(lambda: defaultdict(int))
        completion_idx = defaultdict(dict)
        for line in input_file:
            data = json.loads(line)
            source = data['source']
            completion = data['completion']
            # Update the count
            completion_counts[source][completion] += 1
            # Save the idx if this is the first occurrence
            if completion not in completion_idx[source]:
                completion_idx[source][completion] = data['idx']

    # Step 3: Find the most common completion
    max_completions = {}
    for source, completions in completion_counts.items():
        max_completion = max(completions, key=completions.get)
        max_completions[source] = {
            'idx': completion_idx[source][max_completion],
            'completion': max_completion,
            'count': completions[max_completion]
        }

    # Step 4: Save to new file
    with open(output_file_path, 'w') as output_file:
        for source, data in max_completions.items():
            new_data = {
                'idx': data['idx'],
                'completion': data['completion'],
                'source': source,
                'count': data['count']
            }
            output_file.write(json.dumps(new_data) + '\n')

def filter_completions(filtered_file_path, input_file_path, output_file_path):
    print('*'*50,'filter_completions')

    # Step 1: Read the filtered completions
    with open(filtered_file_path, 'r') as filtered_file:
        filtered_completions = set()
        for line in filtered_file:
            data = json.loads(line)
            filtered_completions.add((data['source'], data['completion']))

    # Step 2: Read the input file and filter the completions
    with open(input_file_path, 'r') as input_file, open(output_file_path, 'w') as output_file:
        for line in input_file:
            data = json.loads(line)
            # If the completion is not in the filtered set, write it to the output file
            if (data['source'], data['completion']) not in filtered_completions:
                output_file.write(line)

def count_source_occurrences(file_path, output_txt_path):
    # Initialize a dictionary to count the occurrences of each source value
    source_counts = defaultdict(int)

    # Read the file and count the source values
    with open(file_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            source = data['source']
            source_counts[source] += 1

    # Write the source counts to the output txt file
    with open(output_txt_path, 'w') as txt_file:
        for source, count in source_counts.items():
            txt_file.write(f'{source}: {count}\n')

    # Count the number of source values with a count less than 3
    less_than_three_count = sum(1 for count in source_counts.values() if count < 3)

    print(f'Total count of source values with less than 3 occurrences: {less_than_three_count}')

def compare_and_save_topk(main_path, remain_path, output_path, k=3):
    main_data, remain_data = read_data(main_path, remain_path)
    all_paths = build_paths(main_data, remain_data, k+1)

    with open(output_path, 'w') as out_file:
            for source, path in tqdm(all_paths.items(),desc='Writing'):
                for completion in path:
                    line = remain_data[source].get(completion, "")
                    if line:
                        out_file.write(line)

def replace_var_names(code, args):
    python_keywords = set([
        'False', 'None', 'True', 'and', 'as', 'assert', 'async', 'await',
        'break', 'class', 'continue', 'def', 'del', 'elif', 'else', 'except',
        'finally', 'for', 'from', 'global', 'if', 'import', 'in', 'is',
        'lambda', 'nonlocal', 'not', 'or', 'pass', 'raise', 'return',
        'try', 'while', 'with', 'yield', 'print'
    ])

    def replace_name(node, var_dict_stack, func_dict_stack, current_var_index, current_func_index):
        if isinstance(node, ast.FunctionDef):
            name = node.name
            if name not in python_keywords and name not in func_dict_stack[-1]:
                new_name = string.ascii_lowercase[current_func_index]  # 使用小写字母替换函数名
                func_dict_stack[-1][name] = new_name
                current_func_index = (current_func_index + 1) % 26
                node.name = new_name
            var_dict_stack.append({})
            func_dict_stack.append({})

        elif isinstance(node, ast.Call):
            func_name = getattr(node.func, 'id', None)
            if func_name:
                for func_dict in func_dict_stack[::-1]:
                    if func_name in func_dict:
                        node.func.id = func_dict[func_name]
                        break

        elif isinstance(node, ast.Name):
            name = node.id
            if isinstance(node.ctx, ast.Load):
                for var_dict in var_dict_stack[::-1]:
                    if name in var_dict:
                        node.id = var_dict[name]
                        break
            elif name not in python_keywords and name not in var_dict_stack[-1]:
                new_name = string.ascii_uppercase[current_var_index]  # 使用大写字母替换变量名
                var_dict_stack[-1][name] = new_name
                current_var_index = (current_var_index + 1) % 26
                node.id = new_name

        for child in ast.iter_child_nodes(node):
            current_var_index, current_func_index = replace_name(child, var_dict_stack, func_dict_stack,
                                                                 current_var_index, current_func_index)

        if isinstance(node, ast.FunctionDef):
            var_dict_stack.pop()
            func_dict_stack.pop()

        return current_var_index, current_func_index

    code = fixed_code(code)
    code = code.replace('\"\"\"\"', '\"\"\"')

    try:
        tree = ast.parse(code)
    except SyntaxError:
        print("Syntax error in the code. Skipping this block.")
        traceback.print_exc()

        if args.save:
            return code
        # with open('/root/cyz/nodup/tora-code-13b_u100/error_codes.jsonl', 'a') as file:
        #    json_record = json.dumps({"error_code": code})
        #    file.write(json_record + '\n')
        return 1

    var_dict_stack = [{}]
    func_dict_stack = [{}]
    current_var_index = 0
    current_func_index = 0

    replace_name(tree, var_dict_stack, func_dict_stack, current_var_index, current_func_index)
    return astunparse.unparse(tree)

def replace_var_names_in_code_blocks(text, args):
    # 找到所有的代码块
    code_blocks = re.findall(r'```python(.*?)```', text, re.DOTALL)

    # 处理每个代码块
    for i, code_block in enumerate(code_blocks):
        processed_code = replace_var_names(code_block.strip(), args)  # 确保去掉代码块两端的空白字符
        if processed_code != 1:
            text = text.replace(f'```python{code_block}```', f'```python\n{processed_code}\n```', 1)
        else:
            text = "1"

    return text


if __name__ == "__main__":


    parser = argparse.ArgumentParser(description='nodup')
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input JSONL file')
    parser.add_argument('--save', action="store_true")
    parser.add_argument(
        "--key",
        type=str,
        default='completion',
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=-1,
        help="The name of the dataset to use (via the datasets library).",
    )
    args = parser.parse_args()
    print(args)

    input_file_path = args.input_file
    dir = os.path.dirname(input_file_path)
    base_name = os.path.splitext(input_file_path)[0]
    # 创建一个临时目录
    tmpdirname = dir + '/tmp/'
    os.makedirs(tmpdirname, exist_ok=True)
    print(f'临时目录创建{tmpdirname}')


    input_file_path_idx = tmpdirname + 'idx.jsonl'
    process_data_file_path = tmpdirname + 'process_data.jsonl'
    main_path = tmpdirname + 'main.jsonl'
    dedup_path = tmpdirname + 'dedup.jsonl'
    remain_path = tmpdirname + 'remain.jsonl'
    filter_remain_path = tmpdirname + 'filter_remain.jsonl'


    main_final = base_name + '_main.jsonl'
    dedup_final = base_name + '_nodup.jsonl'
    remain_final = base_name + '_remain.jsonl'
    output_txt_path = base_name + '_counts.txt'
    filter_remain_final = base_name + f'_remain_filter_top{args.topk}.jsonl'
    filter_final = base_name + f'_all_filter_top{args.topk}.jsonl'

    if args.topk == -1:
        hello()
        add_idx(input_file_path,input_file_path_idx)
        process_data(input_file_path_idx, process_data_file_path,args)
        remove_duplicates(process_data_file_path, dedup_path,args)
        extract_matching_data(input_file_path_idx, dedup_path, dedup_final)

        show_example(input_file_path,0)
        show_example(dedup_final,0)


    else:
        add_idx(input_file_path,input_file_path_idx)
        process_data(input_file_path_idx, process_data_file_path,args)
        find_most_common_completion(process_data_file_path, main_path)
        remove_duplicates(process_data_file_path, dedup_path,args)
        filter_completions(main_path, dedup_path, remain_path)
        # source_counts = count_source_occurrences(remain_path, output_txt_path)
        compare_and_save_topk(main_path, remain_path, filter_remain_path,args.topk)


        extract_matching_data(input_file_path_idx, main_path, main_final)
        extract_matching_data(input_file_path_idx, dedup_path, dedup_final)
        extract_matching_data(input_file_path_idx, remain_path, remain_final)
        extract_matching_data(input_file_path_idx, filter_remain_path, filter_remain_final)
        merge_jsonl([filter_remain_final,main_final], filter_final)


        show_example(input_file_path,0)
        show_example(main_final,0)
        show_example(dedup_final,0)
        show_example(filter_remain_final,0)
        show_example(filter_final,0)

    shutil.rmtree(tmpdirname)
    print('临时目录已删除')
