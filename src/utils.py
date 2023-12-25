import os
import json
import re
import io
import numpy as np
import timeout_decorator
import logging
import Levenshtein
import ast
import astunparse
from contextlib import redirect_stdout
from tqdm import tqdm
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from .grader import math_equal

import copy
import argparse
import sys
from sympy import symbols, Eq, solve, Rational, sympify




def process(input_file, output_file, mu, sigma):
    with open(input_file, 'r', encoding='utf-8') as f, open(output_file, 'w', encoding='utf-8') as out:  
        count = 0
        count1 = 0
        count2 = 0
        count3 = 0
        list1 = []
        list2 = []
        list3 = []
        lines = f.readlines()
        for line in tqdm(lines, desc='Processing'):
            count += 1
            data = json.loads(line)
            data['prompt'], numbers = find_numbers(data['prompt'])
            number_mapping = {}
            for num in numbers:
                perturbed_num = distribute_perturb(float(num), mu, sigma)
                while perturbed_num < 0:
                    perturbed_num = distribute_perturb(float(num), mu, sigma)
                number_mapping[num] = str(int(perturbed_num))
            modified_code_blocks = []
            code_part, output_part = data['completion'].split('```output\n', 1)
            while '"""' in code_part:
                tmp1 = code_part.split('"""',2)[0]
                tmp2 = code_part.split('"""',2)[1]
                tmp3 = code_part.split('"""',2)[2]
                code_part = tmp1 + tmp3
            code_part = re.sub(r'#.*\n', '\n', code_part)
            numbers_set = find_numbers_code(code_part)
            modified_code_part = replace_numbers(code_part, number_mapping, numbers_set)
            
            data['prompt'] = replace_numbers(data['prompt'], number_mapping, numbers_set)
            data['completion'] = modified_code_part + '```output\n' + output_part
            substring = '```\n```output'
            data['completion'] = data['completion'].split(substring)[0]
            python_code = data['completion'].strip('```python\n')
            output = run_python_code(python_code, data['source'])
            if output is None:
                count1 += 1
                list1.append(data['source'])
                pass
            else:
                try:
                    output = float(output)
                    if output < 1e-4:
                        count2 += 1
                        list2.append(data['source'])
                        pass
                except:
                    try:
                        output = float(parse_latex(output).evalf())
                        if output < 1e-4:
                            count2 += 1
                            list2.append(data['source'])
                            pass
                    except:
                        count3 += 1
                        list3.append(data['source'])
                        pass

            data['completion'] = data['completion'] + "```output\n" + str(output) + "\n```"
            data["pred"] = output
            json.dump(data, out, ensure_ascii=False)
            out.write('\n')
        print("None:", count1, list1)
        print("<0:", count2, list2)
        print("ValueError:", count3, list3)
        print("total:", count)
    return list1, list2, list3

def process_change(input_file, output_file, mu, sigma):
    with open(input_file, 'r', encoding='utf-8') as f, open(output_file, 'w', encoding='utf-8') as out:  
        count = 0
        count1 = 0
        count2 = 0
        count3 = 0
        list1 = []
        list2 = []
        list3 = []
        lines = f.readlines()
        for line in tqdm(lines, desc='Processing'):
            count += 1
            data = json.loads(line)
            data['prompt'], numbers = find_numbers(data['prompt'])
            number_mapping = {}
            for num in numbers:
                perturbed_num = distribute_perturb1(float(num), mu, sigma)
                while perturbed_num < 0:
                    perturbed_num = distribute_perturb1(float(num), mu, sigma)
                number_mapping[num] = str(int(perturbed_num))
            modified_code_blocks = []
            code_part, output_part = data['completion'].split('```output\n', 1)
            while '"""' in code_part:
                tmp1 = code_part.split('"""',2)[0]
                tmp2 = code_part.split('"""',2)[1]
                tmp3 = code_part.split('"""',2)[2]
                code_part = tmp1 + tmp3
            code_part = re.sub(r'#.*\n', '\n', code_part)
            numbers_set = find_numbers_code(code_part)
            modified_code_part = replace_numbers(code_part, number_mapping, numbers_set)
            
            data['prompt'] = replace_numbers(data['prompt'], number_mapping, numbers_set)
            data['completion'] = modified_code_part + '```output\n' + output_part
            substring = '```\n```output'
            data['completion'] = data['completion'].split(substring)[0]
            python_code = data['completion'].strip('```python\n')
            python_code = replace_rational(python_code)
            output = run_python_code(python_code, data['source'])
            if output is None:
                count1 += 1
                list1.append(data['source'])
                continue
            else:
                try:
                    output = float(output)
                    if output < 1e-4:
                        count2 += 1
                        list2.append(data['source'])
                        continue
                except:
                    try:
                        output = float(parse_latex(output).evalf())
                        if output < 1e-4:
                            count2 += 1
                            list2.append(data['source'])
                            continue
                    except:
                        count3 += 1
                        list3.append(data['source'])
                        continue

            data['completion'] = data['completion'] + "```output\n" + str(output) + "\n```"
            data["pred"] = output
            json.dump(data, out, ensure_ascii=False)
            out.write('\n')
        print("None:", count1, list1)
        print("<0:", count2, list2)
        print("ValueError:", count3, list3)
        print("total:", count)
    return list1, list2, list3

def change(input_file_path1, input_file_path2, output_file_path):
    data1 = []
    data2 = []
    error_count = 0

    with open(input_file_path1, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in tqdm(lines, desc='Load input data1'):
            data1.append(json.loads(line))

    with open(input_file_path2, 'r', encoding='utf-8') as f: 
        lines = f.readlines()
        for line in tqdm(lines, desc='Load input data2'):
            data2.append(json.loads(line))
    
    count = 0
    set1 = get_idx_set(input_file_path1)
    for i in range(len(data2)):
        if data2[i]['source'] not in set1:
            count += 1
    print("Not match times:", count)
    
    count = 0
    with open(output_file_path, 'w', encoding='utf-8') as out:  
        
        for j in tqdm(range(len(data1)),desc='Change Processing'):
            for i in range(len(data2)):
                if data1[j]['source'] == data2[i]['source']:
                    question1 = data1[j]['prompt']
                    question2 = data2[i]['prompt']
                    question1, question2, number_mapping = find_numbers_group(question1, question2)
                    
                    code2 = data2[i]['completion']
                    code2 = code2.split('```python\n', 1)[1]
                    code2, tmp = code2.split('```\n```output', 1)
                    tmp = tmp.split("\n```\n",1)[1]
                    
                    
                    if "$\\boxed{" in tmp:
                        temp1, temp2 = tmp.split("$\\boxed{",1)
                        temp2 = temp2.split("}$",1)[1]
                    else: 
                        temp1 = 'The answer is'
                        temp2 = '.'
                    while '"""' in code2:
                        tmp1 = code2.split('"""',2)[0]
                        tmp2 = code2.split('"""',2)[1]
                        tmp3 = code2.split('"""',2)[2]
                        code2 = tmp1 + tmp3
                    code2 = re.sub(r'#.*\n', '\n', code2)
                    numbers_set = find_numbers_code(code2)
                    code2 = replace_numbers(code2, number_mapping, numbers_set)
                    temp1 = replace_numbers(temp1, number_mapping, numbers_set)
                    temp2 = replace_numbers(temp2, number_mapping, numbers_set)
                    
                    output = run_python_code(code2, data1[j]['source'])
                    if output:
                        output = output.strip()
                    
                    
                        data = {
                            "prompt": question1,
                            "completion": "```python\n" + code2 + "```\n```output\n" + str(output) + "\n```\n" + temp1 + "$\\boxed{" + str(output) + "}$" + temp2,

                            "pred": output,
                            "source": data2[i]["source"],
                            "idx": data2[i]["idx"]
                        }

                        try:
                            json.dump(data, out, ensure_ascii=False)
                            out.write('\n')
                            count += 1
                        except Exception as e:
                            logging.error("An error occurred while writing data to output file: %s", e)
                            error_count += 1
    print("remain times:", count)     
    print("error times:", error_count)              

def topk(main_path, remain_path, output_path, k=3):
    main_data, remain_data = read_data(main_path, remain_path)
    
    all_paths = build_paths(main_data, remain_data, k+1)

    with open(output_path, 'w') as out_file:
            for source, path in tqdm(all_paths.items(),desc='Writing'):
                for completion in path:
                    line = remain_data[source].get(completion, "")
                    if line:
                        out_file.write(line) 

def build_paths(main_data, remain_data, k):
    all_paths = {}
    with ProcessPoolExecutor() as executor:
        # 创建所有任务
        futures = {executor.submit(build_greedy_path_for_source, main_completion, remain_data[source], k): source 
                   for source, main_completion in main_data.items() if source in remain_data}

        # 获取结果
        for future in tqdm(futures, desc="Processing sources"):
            source = futures[future]
            path = future.result()
            all_paths[source] = path

    return all_paths

def build_greedy_path_for_source(main_completion, remain_completions_dict, k):
    path = [main_completion]
    path_distances = {comp: Levenshtein.distance(main_completion, comp) for comp in remain_completions_dict}

    while len(path) < k and remain_completions_dict:
        # 选择下一个点
        next_completion = max(remain_completions_dict, key=lambda comp: path_distances[comp])
        next_distance = path_distances[next_completion]
        
        # 更新路径和距离
        path.append(next_completion)
        del remain_completions_dict[next_completion]
        del path_distances[next_completion]

        # 仅更新剩余点的累积距离
        for comp in remain_completions_dict:
            path_distances[comp] += Levenshtein.distance(next_completion, comp)

    return path

def read_data(main_path, remain_path):
    main_data = {}
    remain_data = defaultdict(dict)

    with open(main_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            processed_code = process_data(data['completion'])
            main_data[data['source']] = processed_code

    with open(remain_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            processed_code = process_data(data['completion'])
            source = data['source']
            remain_data[source][processed_code] = line

    return main_data, remain_data

def process_data(code):
    # 使用 AST 替换变量和函数名
    try:
        updated_code = replace_var_names_in_code_blocks(code)
        if updated_code != "1":
            return updated_code
    except Exception as e:
        print(f"Error processing code: {e}")
    return code

def replace_var_names(code):
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
            current_var_index, current_func_index = replace_name(child, var_dict_stack, func_dict_stack, current_var_index, current_func_index)

        if isinstance(node, ast.FunctionDef):
            var_dict_stack.pop()
            func_dict_stack.pop()

        return current_var_index, current_func_index

    
    code = fixed_code(code)
    code = code.replace('\"\"\"\"','\"\"\"')
    
    
    try:
        tree = ast.parse(code)
    except SyntaxError:
        print("Syntax error in the code. Skipping this block.")
        with open('/data/cyz/nodup/tora-code-13b_u100/error_codes.jsonl', 'a') as file:
            json_record = json.dumps({"error_code": code})
            file.write(json_record + '\n')
        return 1
    
    var_dict_stack = [{}]
    func_dict_stack = [{}]
    current_var_index = 0
    current_func_index = 0

    replace_name(tree, var_dict_stack, func_dict_stack, current_var_index, current_func_index)
    return astunparse.unparse(tree)

def replace_var_names_in_code_blocks(text):
    # 找到所有的代码块
    code_blocks = re.findall(r'```python(.*?)```', text, re.DOTALL)
    
    # 处理每个代码块
    for i, code_block in enumerate(code_blocks):
        processed_code = replace_var_names(code_block.strip())  # 确保去掉代码块两端的空白字符
        if processed_code != 1:
            text = text.replace(f'```python{code_block}```', f'```python\n{processed_code}\n```', 1)
        else:
            text = "1"
        
    return text   

def fixed_code(code):
    # Regular expression pattern to match '\n   ' but not '\n    '
    pattern = re.compile(r'\n {3}(?! )')

    # Replace occurrences of '\n   ' with '\n    '
    fixed_code = re.sub(pattern, '\n    ', code)

    return fixed_code

def sameanswer(input_file_path,output_file_path):
    data = []
    with open(input_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in tqdm(lines, desc='Processing'):
            data.append(json.loads(line))
    filtered_data = []
    for i in data:
        current_source = i["source"]
        current_pred = i["pred"]
        matching_entry = None

        # 在 filtered_data 中查找具有相同 source 的匹配项
        for entry in filtered_data:
            if entry["source"] == current_source and entry["pred"] == current_pred and entry["pred"] != None:
                matching_entry = entry
                break

        if matching_entry is None:
            # 如果未找到匹配项，则将当前数据添加到 filtered_data 中
            filtered_data.append(i)
    
    seen_sources = set()  # 存储已经处理过的 source
    
    with open(output_file_path, 'w', encoding='utf-8') as out:
        for i in data:
            current_source = i["source"]
            current_pred = i["pred"]

            # 如果当前 source 已经处理过，则跳过
            if current_source in seen_sources:
                continue

            # 在遇到新的 source 时，输出数据并将其添加到已处理的 source 集合中
            seen_sources.add(current_source)
            json.dump(i, out, ensure_ascii=False)
            out.write('\n')

def match_true(input_file_path1, input_file_path2, output_file_path1, output_file_path2):
    data1 = []
    data2 = []
    with open(input_file_path1, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in tqdm(lines, desc='Load create'):
            data1.append(json.loads(line))
    with open(input_file_path2, 'r', encoding='utf-8') as f: 
        lines = f.readlines()
        for line in tqdm(lines, desc='Load same answer'):
            data2.append(json.loads(line))
    true1 = 0
    with open(output_file_path1, 'w', encoding='utf-8') as out1, open(output_file_path2, 'w', encoding='utf-8') as out2:  
        for i in tqdm(range(len(data1)), desc="Processing data1"):
            for j in range(len(data2)):
                if data2[j]['pred']:
                    pred_value = data2[j]['pred'].strip()
                if data1[i]['source'] == data2[j]['source']:
                    if math_equal(pred_value, data1[i]['pred']):
                        # print(pred_value, data1[i]['pred'])
                        true1 += 1
                        # print(data2[j]["idx"])
                        data = {
                            "pred": pred_value,
                            "source": data2[j]["source"],
                            "idx": data2[j]["idx"],
                            "score": True
                        }
                        json.dump(data, out1, ensure_ascii=False)
                        out1.write('\n')
                    else:
                        json.dump(data1[i], out2, ensure_ascii=False)
                        out2.write('\n')
            
    print(true1)    

def get_idx_set(file_path):
    idx_set = set()
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            idx_set.add(data['source'])
    return idx_set

def extract(input_file1, input_file2, output_file1, output_file2):
    set1 = get_idx_set(input_file2)
    data1 = []
    data2 = []
    with open(input_file1, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in tqdm(lines, desc='Extract Load 1'):
            data1.append(json.loads(line))
    with open(input_file2, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in tqdm(lines, desc='Extract Load 2'):
            data2.append(json.loads(line))   
    with open(output_file1, 'w', encoding='utf-8') as out1, open(output_file2, 'w', encoding='utf-8') as out2:
        for i in tqdm(range(len(data1)), desc='Extract Processing'):
            for j in range(len(data2)):
                if data1[i]['source'] == data2[j]['source']:
                    if math_equal(data1[i]['pred'],data2[j]['pred']):
                    # print(i)
                        json.dump(data1[i], out1, ensure_ascii=False)
                        out1.write('\n')
                    else:
                        json.dump(data1[i], out2, ensure_ascii=False)
                        out2.write('\n')

def merge_and_sort_jsonl(input_file_path, output_file_path, sort_key='source'):

    data = []
    if os.path.exists(input_file_path):
        with open(input_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data.append(json.loads(line.strip()))
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON in input file: {e}")

    if os.path.exists(output_file_path):
        with open(output_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data.append(json.loads(line.strip()))
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON in output file: {e}")

    sorted_data = sorted(data, key=lambda x: x.get(sort_key))

    with open(output_file_path, 'w', encoding='utf-8') as f:
        for item in sorted_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')      

def _fix_sqrt(string):
    _string = re.sub(r"\\sqrt(\w+)", r"\\sqrt{\1}", string)
    return _string

def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if len(substr) > 0 and substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string

def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        if "sqrt" not in a:
            a = int(a)
        if "sqrt" not in b:
            b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:
        return string

def strip_string(string):
    string = str(string).strip()
    # linebreaks
    string = string.replace("\n", "")

    # right "."
    string = string.rstrip(".")

    # remove inverse spaces
    string = string.replace("\\!", "")
    string = string.replace("\\ ", "")

    # replace \\ with \
    string = string.replace("\\\\", "\\")
    string = string.replace("\\\\", "\\")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove unit: miles, dollars if after is not none
    _string = re.sub(r"\\text{.*?}$", "", string).strip()
    if _string != "" and _string != string:
        # print("Warning: unit not removed: '{}' -> '{}'".format(string, _string))
        string = _string

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")
    string = string.replace("$", "")

    string = string.replace("\\text", "")
    string = string.replace("x\\in", "")

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")
    string = string.replace("%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")

    # cdot
    string = string.replace("\\cdot", "")

    # inf
    string = string.replace("infinity", "\\infty")
    if "\\infty" not in string:
        string = string.replace("inf", "\\infty")
    string = string.replace("+\\inity", "\\infty")

    # and 
    string = string.replace("and", "")
    string = string.replace("\\mathbf", "")

    # use regex to remove \mbox{...}
    string = re.sub(r"\\mbox{.*?}", "", string)

    # quote
    string.replace("'", "")
    string.replace("\"", "")
    
    # i, j
    if "j" in string and "i" not in string:
        string = string.replace("j", "i")

    # replace a.000b where b is not number or b is end, with ab, use regex
    string = re.sub(r"(\d+)\.0+([^\d])", r"\1\2", string)
    string = re.sub(r"(\d+)\.0+$", r"\1", string)

    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    string = _fix_sqrt(string)
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)

    return string

def replace_rational(text):
    def replacer(match):
        return f"{match.group(1)} / {match.group(2)}"
    return re.sub(r'Rational\((\d+),\s*(\d+)\)', replacer, text)

def find_numbers(text):
    text = re.sub(r'\$(\d+(,\d{3})(\.\d{2})?)', lambda x: x.group(1).replace(',', ''), text)
    number_list = re.findall(r'(?<![\w])\d+\s*x\s*\d+\s*x\s*\d+(?![\w-])|(?<![\w])\d+\s+[p][e][r][c][e][n][t](?![\w-])|(?<![\w])\d+\/\d+(?![\w-])|(?<![\w])\d+\:\d+(?![\w-])|(?<![\w])\d+[srnt][tdh](?![\w-])|(?<![\w])\d+\.\d+%(?![\w-])|(?<![\w])\d+\.?\d*%?(?![\w-])|(?<![\w])\.\d+(?![\w-])|(?<![\w])\d+(?![\w-])', text)
    numbers_set = set()
    for num in number_list:
        if num.endswith('%'):
            continue
        if "/" in num:
            continue
        if ":" in num:
            continue
        if "rd" in num or "th" in num or "nd" in num or "st" in num or "percent" in num:
            continue
        if "." in num:
            if num.endswith('.'):
                num = num.replace('.','')
            elif num[0] == ".":
                continue
        if ("2" == num and ("half" in text or "bicycle" in text) ) or ("3" == num and ("one-third" in text or "two-thirds" in text or "tricycle" in text)) or ("1" == num and "unicycle" in text):
            continue
        if "x" in num:
            continue
        num = strip_string(num)
        if num not in numbers_set:
            numbers_set.add(num)
        
    numbers = list(numbers_set)
    return text, numbers

def find_numbers_group(text1,text2):
    def extract_numbers(text):
        text = re.sub(r'\$(\d+(,\d{3})(\.\d{2})?)', lambda x: x.group(1).replace(',', ''), text)
        number_list = re.findall(r'(?<![\w])\d+\s*x\s*\d+\s*x\s*\d+(?![\w-])|(?<![\w])\d+\s+[p][e][r][c][e][n][t](?![\w-])|(?<![\w])\d+\/\d+(?![\w-])|(?<![\w])\d+\:\d+(?![\w-])|(?<![\w])\d+[srnt][tdh](?![\w-])|(?<![\w])\d+\.\d+%(?![\w-])|(?<![\w])\d+\.?\d*%?(?![\w-])|(?<![\w])\.\d+(?![\w-])|(?<![\w])\d+(?![\w-])', text)
        
        cleaned_numbers = []
        for num in number_list:
            if num.endswith('%'):
                continue
            if "/" in num:
                continue
            if ":" in num:
                continue
            if "rd" in num or "th" in num or "nd" in num or "st" in num or "percent" in num:
                continue
            if "." in num:
                if num.endswith('.'):
                    num = num.replace('.','')
                elif num[0] == ".":
                    continue
            if ("3" == num and ("one-third" in text or "two-thirds" in text or "tricycle" in text)) or ("1" == num and "unicycle" in text):
                continue
            if "x" in num:
                continue
            num = strip_string(num)
            cleaned_numbers.append(num)
            
        return text, cleaned_numbers

    text1, number_list1 = extract_numbers(text1)
    text2, number_list2 = extract_numbers(text2)

    mapping = {}

    for i in range(len(number_list1)):
        mapping[number_list2[i]] = number_list1[i]

    return text1, text2, mapping

def find_numbers_code(text):
    number_list = re.findall(r'(?<![\w])\d+\s*x\s*\d+\s*x\s*\d+(?![\w-])|(?<![\w])\d+\s+[p][e][r][c][e][n][t](?![\w-])|(?<![\w])\d+\/\d+(?![\w-])|(?<![\w])\d+\:\d+(?![\w-])|(?<![\w])\d+[srnt][tdh](?![\w-])|(?<![\w])\d+\.\d+%(?![\w-])|(?<![\w])\d+\.?\d*%?(?![\w-])|(?<![\w])\.\d+(?![\w-])|(?<![\w])\d+(?![\w-])', text)
    numbers_set = set()
    for num in number_list:
        if num.endswith('%'):
            continue
        if "/" in num:
            continue
        if ":" in num:
            continue
        if "rd" in num or "th" in num or "nd" in num or "st" in num or "percent" in num:
            continue
        if "." in num:
            if num.endswith('.'):
                num = num.replace('.','')
            elif num[0] == ".":
                continue
        if ("2" == num and ("half" in text or "bicycle" in text) ) or ("3" == num and ("one-third" in text or "two-thirds" in text or "tricycle" in text)) or ("1" == num and "unicycle" in text):
            continue
        if "x" in num:
            continue
        num = strip_string(num)
        if num not in numbers_set:
            numbers_set.add(num)
        else:
            numbers_set.remove(num)
    return numbers_set

def distribute_perturb(number, mean, std_dev, seed=42):
    if seed is not None:
        np.random.seed(seed)
    
    X = np.random.normal(mean, std_dev)
    perturbed_number = number + int(X)
    return perturbed_number

def distribute_perturb1(number, mean, std_dev):
    X = np.random.normal(mean, std_dev)
    perturbed_number = number + int(X)
    return perturbed_number

def replace_numbers(text, number_mapping, numbers_set):
    def replacer(match):
        num = match.group()
        if num.endswith('.'):
            num = num[:-1]
        if num in numbers_set:
            return str(number_mapping.get(num, num))
        else: 
            return str(num)
 
    return re.sub(r'(?<![\w])\d+\s*x\s*\d+\s*x\s*\d+(?![\w-])|(?<![\w])\d+\s+[p][e][r][c][e][n][t](?![\w-])|(?<![\w])\d+\/\d+(?![\w-])|(?<![\w])\d+\:\d+(?![\w-])|(?<![\w])\d+[srnt][tdh](?![\w-])|(?<![\w])\d+\.\d+%(?![\w-])|(?<![\w])\d+\.?\d*%?(?![\w-])|(?<![\w])\.\d+(?![\w-])|(?<![\w])\d+(?![\w-])', replacer, text)

def run_python_code(python_code, idx):
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

def extract_lines_by_idx(input_file, idx_list, output_file):
    with open(input_file, 'r', encoding='utf-8') as f, open(output_file, 'w', encoding='utf-8') as out:
        for line in f:
            data = json.loads(line)
            if data['source'] in idx_list:
                json.dump(data, out, ensure_ascii=False)
                out.write('\n')

def filter_and_save(input_file, idx_set, output_file):
    with open(input_file, 'r', encoding='utf-8') as f, open(output_file, 'w', encoding='utf-8') as out:
        for line in f:
            data = json.loads(line)
            if data['source'] not in idx_set:
                json.dump(data, out, ensure_ascii=False)
                out.write('\n')

def extract_prompt(input_file_path, output_file_path):
    with open(input_file_path, 'r', encoding='utf-8') as f_in, open(output_file_path, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            data = json.loads(line)
            question = data['prompt'].split("<|user|>\n")[-1]
            question = question.split("\n<|assistant|>\n")[0]

            extracted_data = {
                "question": question,
                "gt_cot": "0",
                "gt": data['pred'],
                "idx": data["source"]
            }
            json.dump(extracted_data, f_out, ensure_ascii=False)
            f_out.write('\n')

def fix(input_file1, input_file2, output_file):
    data1 = []
    idx = []
    with open(input_file1, 'r', encoding='utf-8') as file1:
        for line in file1:
            data = json.loads(line)
            data1.append(data)
            idx.append(data['idx'])

    with open(input_file2, 'r', encoding='utf-8') as file2:
        for line in file2:
            data = json.loads(line)
            gt = data['answer'].split("\n#### ")[-1]
            if data['idx'] not in idx:
                new_data = {
                    "question": data['question'],
                    "gt_cot": "0",
                    "gt": gt,
                    "idx": data["idx"]
                }
                data1.append(new_data)
    
    with open(output_file, 'w', encoding='utf-8') as out:
        for line in data1:
            json.dump(line, out)
            out.write('\n')

def merge_files(file_paths, output_file):
    idx_set = set()
    with open(output_file, 'w', encoding='utf-8') as out:
        for file_path in file_paths:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    if data['source'] not in idx_set:
                        idx_set.add(data['source'])
                        json.dump(data, out, ensure_ascii=False)
                        out.write('\n')
    return idx_set

def save_original_data(input_file, idx_set, output_file):
    with open(input_file, 'r', encoding='utf-8') as f, open(output_file, 'w', encoding='utf-8') as out:
        for line in f:
            data = json.loads(line)
            if data['source'] in idx_set:
                json.dump(data, out, ensure_ascii=False)
                out.write('\n')
    print('save complete!')
    
def merge_jsonl(files, output_file):
    all_data = []

    for file in files:
        if os.path.exists(file) and os.path.isfile(file):
            with open(file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        all_data.append(data)
                    except json.JSONDecodeError:
                        print(f"Warning: Could not decode line in file {file}. Skipping...")
        else:
            print(f"Warning: File {file} does not exist or is not a file. Skipping...")

    with open(output_file, 'w', encoding='utf-8') as out_f:
        for idx, data in enumerate(all_data):
            json.dump(data, out_f)
            out_f.write('\n')

    print(f"Merge completed. Total data num {idx+1}. Data written to {output_file}")
    

