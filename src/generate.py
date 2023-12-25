import json
import os
import re
import random
import sys
import io
from sympy import symbols, Eq, solve
from .utils import merge_jsonl
from .create import extract_true_train_data
import argparse

def main(data, source_file_paths):
    dir = f'./data/{data}/'
    os.makedirs(dir, exist_ok=True)

    output_file_path1 = dir + 'combine.jsonl'
    output_file_path2 = dir + 'extract.jsonl'
    merge_jsonl(source_file_paths, output_file_path1)
    extract_true_train_data(output_file_path1, output_file_path2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process and extract data.')
    parser.add_argument('--data', type=str, required=True, help='Directory to store the output file')
    parser.add_argument('--source_files', nargs='+', required=True, help='List of source file paths to merge')

    args = parser.parse_args()

    main(args.data, args.source_files)


