## 制作attack 数据集

import json
import os
import re
import random
import sys
import io
from sympy import symbols, Eq, solve, Rational, sympify
import numpy as np
import timeout_decorator
from num2words import num2words
from collections import Counter
from sympy.parsing.latex import parse_latex
from contextlib import redirect_stdout
from tqdm import tqdm
import logging
import string
from utils import find_numbers, distribute_perturb, replace_numbers, run_python_code, find_numbers_code, remove, extract_lines_by_idx, extract_prompt, change, topk, sameanswer, match_true, get_idx_set, extract, fix, process, merge_and_sort_jsonl, merge_files, filter_and_save, save_original_data, process_change
import argparse
import shutil



def cleanup_files(*file_paths):
    if not args.save:
        for file_path in file_paths:
            if os.path.exists(file_path):
                os.remove(file_path)

def cleanup_directory(directory_path):
    if not args.save:
        if os.path.exists(directory_path) and os.path.isdir(directory_path):
            shutil.rmtree(directory_path)

if __name__ == "__main__":

    ## create distribution 


    parser = argparse.ArgumentParser(description='Process file based on method.')
    parser.add_argument('method', type=str, choices=['distribution', 'similar'], 
                    help='Method to process the file: "distribution" or "similar"')
    parser.add_argument('direction', type=str, help='Directory path')
    parser.add_argument('data', type=str, help='Data file name')
    parser.add_argument('--save', action='store_true', 
                        help='Set this flag to keep intermediate files')


    args = parser.parse_args()
    dirr = args.direction
    data = args.data
    os.makedirs(dirr, exist_ok=True)

    original_path = './data/gsm_test/extract_main.jsonl'
    distribution_path = f'{dirr}/{data}_distribution.jsonl' 
    error_file = f'{dirr}/Error.jsonl'
    wrong_file = f'{dirr}/Wrong.jsonl'
    valueerror_file = f'{dirr}/ValueError.jsonl'
    
    if args.method == 'distribution':
        mu = 1000
        sigma = 300
    else: 
        mu = 5
        sigma = 1
    l1, l2, l3 = process(original_path, distribution_path, mu, sigma)

    extract_lines_by_idx(distribution_path, l1, error_file)
    extract_lines_by_idx(distribution_path, l2, wrong_file)
    extract_lines_by_idx(distribution_path, l3, valueerror_file)


    
    ## 修改不合理数据
    merged_file_path = f'{dirr}/{data}_error.jsonl'
    filter_path = f'{dirr}/{data}_filter.jsonl' 
    match1 = f'{dirr}/original_extracted_list_Error.jsonl'
    match2 = f'{dirr}/original_extracted_list_Wrong.jsonl'
    match3 = f'{dirr}/original_extracted_list_ValueError.jsonl'
    match = f'{dirr}/original_extracted_list.jsonl'
    addnew = f'{dirr}/extracted_list_new.jsonl'
    sort_key = 'source'  
    
    extracted_files = [error_file, wrong_file, valueerror_file]    
    set1 = get_idx_set(error_file)
    set2 = get_idx_set(wrong_file)
    set3 = get_idx_set(valueerror_file)
    idx_set = merge_files(extracted_files, merged_file_path)
    filter_and_save(distribution_path, idx_set, filter_path)
    
    save_original_data(original_path, set1, match1)
    save_original_data(original_path, set2, match2)
    save_original_data(original_path, set3, match3)
    merge_and_sort_jsonl(match1, match, sort_key)
    merge_and_sort_jsonl(match2, match, sort_key)
    merge_and_sort_jsonl(match3, match, sort_key)
    save_original_data(original_path, idx_set, match)

    
    l1,l2,l3 = process_change(match, addnew, mu, sigma)  
    merge_and_sort_jsonl(addnew, filter_path, sort_key)
    extract_lines_by_idx(original_path, l1, match1)
    extract_lines_by_idx(original_path, l2, match2)
    extract_lines_by_idx(original_path, l3, match3)
    cleanup_files(match)
    merge_and_sort_jsonl(match1, match, sort_key)
    merge_and_sort_jsonl(match2, match, sort_key)
    merge_and_sort_jsonl(match3, match, sort_key)
    idx_set = set(l1 + l2 + l3)
    save_original_data(original_path, idx_set, match)
    
    
    count = 1
    
    while count < 2:       
        l1,l2,l3 = process_change(match, addnew, mu, sigma)  
        merge_and_sort_jsonl(addnew, filter_path, sort_key)
        extract_lines_by_idx(original_path, l1, match1)
        extract_lines_by_idx(original_path, l2, match2)
        extract_lines_by_idx(original_path, l3, match3)
        cleanup_files(match)
        merge_and_sort_jsonl(match1, match, sort_key)
        merge_and_sort_jsonl(match2, match, sort_key)
        merge_and_sort_jsonl(match3, match, sort_key)
        idx_set = set(l1 + l2 + l3)
        print("len = ",len(idx_set))
        save_original_data(original_path, idx_set, match)
        count+= 1
        print("attempt times:", count)
     
    ## 更改对应题目
    remain_path = "./data/gsm_test/extract_remain.jsonl"
    changed_path = f'{dirr}/{data}_remain_changed.jsonl'
    topk_path =  f'{dirr}/{data}_topk.jsonl'
    Same_answer_path = f'{dirr}/{data}_same_answer.jsonl'
    temp1 = f"{dirr}/tmp1.jsonl"
    temp2 = f"{dirr}/tmp2.jsonl"
    changed_true_path = f'{dirr}/{data}_main_changed_true.jsonl'
    new_main_changed_path = f"{dirr}/extract_main_new_changed.jsonl"

    change(filter_path, remain_path, changed_path)
    topk(filter_path, changed_path, topk_path, 3)
 
    sameanswer(topk_path, Same_answer_path)
    match_true(filter_path, Same_answer_path, temp1, temp2)
    set1 = get_idx_set(temp1)
    extract_lines_by_idx(filter_path, set1, changed_true_path)
    
    ## 转成test格式
    dir = f"./data/{data}/"
    dirrr = f'./data/{data}_all'
    os.makedirs(dir, exist_ok=True)
    os.makedirs(dirrr, exist_ok=True)
    test_file = f'{dir}/test.json'
    changed_true_path = f'{dirr}/{data}_main_changed_true.jsonl'
    extract_prompt(changed_true_path,test_file)
    
    tmp1 = "./data/gsm_test/test.json"
    tmp2 = f'{dirrr}/test.json'
    fix(test_file,tmp1,tmp2)
    
    cleanup_files(
        error_file, wrong_file, valueerror_file, 
        merged_file_path, filter_path, match1, match2, match3, match, addnew, 
        temp1, temp2, changed_true_path, topk_path, Same_answer_path, changed_path, distribution_path
    )
    
