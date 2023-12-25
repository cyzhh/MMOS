import json
import os
import re
import random
import sys
import io
from sympy import symbols, Eq, solve
from .utils import merge_jsonl
from .create import extract_true_train_data

if __name__ == "__main__":
    
    
    data = 'gsm_total'
    dir = f'./data/{data}/'
    os.makedirs(dir, exist_ok=True)


    source_file_path = ['/data/ToRA/src/outputs/pretrain_model/tora-70b/gsm8k/cz993_test_tora_-1_seed0_t0.9_s0_e1319_11-18_10-40.jsonl']


    output_file_path1 = dir + 'combine.jsonl'
    output_file_path2 = dir + 'extract.jsonl'
    merge_jsonl(source_file_path,output_file_path1)

    extract_true_train_data(output_file_path1,output_file_path2)



