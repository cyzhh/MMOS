import argparse
import numpy as np
import ast
from tqdm import tqdm
from pebble import ProcessPool
from concurrent.futures import TimeoutError

from eval.grader import *
from utils.parser import *
from utils.utils import load_jsonl
from utils.python_executor import PythonExecutor


def evaluate(data_name, prompt_type, samples: list=None, file_path: str=None, max_num_samples=None, execute=False):
    gsm_hard = True if data_name == "gsm-hard" else False

    assert samples or file_path, "samples or file_path must be provided"
    if not samples:
        samples = list(load_jsonl(file_path))
    # dedup by idx
    if 'idx' in samples[0]:
        samples = {sample['idx']: sample for sample in samples}.values()
        samples = sorted(samples, key=lambda x: x['idx']) 
    else:
        samples = [dict(idx=idx, **sample) for idx, sample in enumerate(samples)]

    if max_num_samples:
        print(f"max_num_samples: {max_num_samples} / {len(samples)}")
        samples = samples[:max_num_samples]
    
    # parse gt
    for sample in samples:
        sample['gt_cot'], sample['gt'] = parse_ground_truth(sample, data_name)

    # execute
    if ('pred' not in samples[0]) or execute:
        if "pal" in prompt_type:
            executor = PythonExecutor(get_answer_expr="solution()")
        else:
            executor = PythonExecutor(get_answer_from_stdout=True)

        for sample in tqdm(samples, desc="Execute"):
            sample['pred'] = []
            sample['report'] = []
            for code in sample['code']:
                pred, report = run_execute(executor, code, prompt_type, execute=True, gsm_hard=gsm_hard)

                if not gsm_hard:
                    sample['pred'].append(pred)

                else:
                    sample['pred'] = ast.literal_eval(pred)

                    # print(sample['pred'])
                sample['report'].append(report)
            # print("sample['pred']", sample['pred'])
            # sample['pred'] = majority_vote(sample['pred'])
            # print("majority_vote_pred", sample['pred'])

    params = [(idx, pred, sample['gt']) for idx, sample in enumerate(samples) for pred in sample['pred']]


    scores = []
    timeout_cnt = 0 

    with ProcessPool() as pool:
        future = pool.map(math_equal_process, params, timeout=3)
        iterator = future.result()
        with tqdm(total=len(samples), desc="Evaluate") as progress_bar:
            while True:
                try:
                    result = next(iterator)
                    scores.append(result)
                except StopIteration:
                    break
                except TimeoutError as error:
                    print(error)
                    scores.append(False)
                    timeout_cnt += 1
                except Exception as error:
                    print(error.traceback)
                    exit()
                progress_bar.update(1) 

    idx = 0
    score_mat = []
    majority_vote_mat = []
    for sample in samples:
        sample['score'] = scores[idx: idx+len(sample['pred'])]

        if not gsm_hard:
            majority_index = majority_vote(sample['pred'])
            sample['majority_vote'] = scores[idx + majority_index]
            majority_vote_mat.append(sample['majority_vote'])
        else:
            sample['majority_vote'] = any(sample['score'])
            majority_vote_mat.append(sample['majority_vote'])


        assert len(sample['score']) == len(sample['pred'])
        score_mat.append(sample['score'])
        idx += len(sample['pred'])



    max_len = max([len(s) for s in score_mat])

    for i, s in enumerate(score_mat):
        if len(s) < max_len:
            score_mat[i] = s + [s[-1]] * (max_len - len(s)) # pad

    # output mean of each column of scores
    col_means= np.array(score_mat).mean(axis=0)
    mean_score = list(np.round(col_means * 100, decimals=1))

    majority_vote_mat_mean = np.array(majority_vote_mat).mean()
    majority_vote_mat_mean = round(majority_vote_mat_mean*100,1)
    result_str = f"Num samples: {len(samples)}\n" \
        f"Num scores: {len(scores)}\n" \
        f"Timeout samples: {timeout_cnt}\n" \
        f"Empty samples: {len([s for s in samples if not s['pred'][-1]])}\n" \
        f"Mean score: {mean_score}\n" \
        f"Majority score: {majority_vote_mat_mean}\n"
    # each type score
    if "type" in samples[0]:
        type_scores = {}
        for sample in samples:
            if sample['type'] not in type_scores:
                type_scores[sample['type']] = []
            type_scores[sample['type']].append(sample['majority_vote'])
        type_scores = {k: np.round(np.array(v).mean() * 100, decimals=1) for k, v in type_scores.items()}
        type_scores = {k: v for k, v in sorted(type_scores.items(), key=lambda item: item[0])}
        result_str += f"Type scores: {type_scores}\n"

    print(result_str)
    return result_str


def majority_vote(lst):
    # 计算每个元素出现的次数
    count_dict = {}
    max_count = 0
    majority_index = 0
    for i,element in enumerate(lst):
        if element not in count_dict:
            count_dict[element] = 1
        else:
            count_dict[element] += 1

        # 更新当前的多数元素
        if count_dict[element] > max_count:
            max_count = count_dict[element]
            majority_index = i

    return majority_index


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str, default="math")
    parser.add_argument("--prompt_type", type=str, default="tora")
    parser.add_argument("--file_path", type=str, default=None, required=True)
    parser.add_argument("--max_num_samples", type=int, default=None)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    evaluate(data_name=args.data_name, prompt_type=args.prompt_type, file_path=args.file_path,
             max_num_samples=args.max_num_samples, execute=args.execute)
