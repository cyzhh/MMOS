<h1 align="center">MMOS</h1>



<div align="center">
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/an-empirical-study-of-data-ability-boundary/arithmetic-reasoning-on-gsm8k)](https://paperswithcode.com/sota/arithmetic-reasoning-on-gsm8k?p=an-empirical-study-of-data-ability-boundary)  [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/an-empirical-study-of-data-ability-boundary/math-word-problem-solving-on-svamp)](https://paperswithcode.com/sota/math-word-problem-solving-on-svamp?p=an-empirical-study-of-data-ability-boundary)  [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/an-empirical-study-of-data-ability-boundary/math-word-problem-solving-on-asdiv-a)](https://paperswithcode.com/sota/math-word-problem-solving-on-asdiv-a?p=an-empirical-study-of-data-ability-boundary) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/an-empirical-study-of-data-ability-boundary/math-word-problem-solving-on-math)](https://paperswithcode.com/sota/math-word-problem-solving-on-math?p=an-empirical-study-of-data-ability-boundary)
</div>

<p align="center">
  <img src="./images/first_table.png" width="500" />
</p>

<p align="center">
  | <a href="https://arxiv.org/abs/2403.00799">ArXiv</a> | <a href="https://pan.quark.cn/s/2d16e640ed07">Models</a> | <a href="https://huggingface.co/datasets/cyzhh/MMOS">Data</a> | <a href="https://github.com/cyzhh/MMOS">Code</a> |
</p>


## 🔥 News
- [2024/2/28] 🔥🔥🔥Models **MMOS-DeepSeekMath 7B** show nice performence and released at [MMOS-DeepSeekMath 7B](https://pan.quark.cn/s/b939a0510658) !!
- [2024/2/27] 🔥🔥🔥Models **MMOS-LLEMMA 7B** show nice performence and released at [MMOS-LLEMMA 7B](https://pan.quark.cn/s/59024b402c1b) !!
- [2024/2/27] 🔥🔥 Models **MMOS-CODE 13B**  and **MMOS-CODE 34B** released at [MMOS-CODE 13B](https://pan.quark.cn/s/5d5ee083676f) and [MMOS-CODE 34B](https://pan.quark.cn/s/734ff44143da) !!
- [2024/2/27] 🔥 Models **MMOS-CODE 7B** released at [MMOS-CODE 7B](https://pan.quark.cn/s/62a6644c0e02) !!
- [2024/2/26] 🔥 Dataset **MMOS** released at [😊 HuggingFace](https://huggingface.co/datasets/cyzhh/MMOS) !!
- [2024/2/23] 🔥🔥🔥Arxiv released at [An Empirical Study of Data Ability Boundary in LLMs' Math Reasoning](https://arxiv.org/abs/2403.00799) ~ 

<!-- - [2024/1/12] Models ZZ-Math 7B released at [Google Drive](https://drive.google.com/drive/folders/13tpLR0bNLLg1oLkjUuwJT8STCB10uSSS?usp=sharing) or [Quark](https://pan.quark.cn/s/0b69ec84c793) 
- [2024/1/11] Dataset released at [😊 HuggingFace](https://huggingface.co/datasets/cyzhh/TAL-SCQ-CN_mix) -->

## 💡 Introductions & Performances

Mix of Minimal Optimal Sets (MMOS) of dataset has two advantages for two aspects, higher performance and lower construction costs on math reasoning.


| Model            | Size | GSM8K | SVAMP | ASDiv | MATH | Size | GSM8K | SVAMP | ASDiv | MATH |Size | GSM8K | SVAMP | ASDiv | MATH |
|------------------|------|-------|-------|-------|------|------|-------|-------|-------|------|------|-------|-------|-------|------|
| WizardMath       | 7B   | 54.9  | 57.3  | 59.1  | 10.7 | 13B  | 63.9  | 64.3  | 65.8  | 14.0 | 34B  | -     | -     | -     | -    |
| MAMMOTH          | 7B   | 53.6  | 67.7  | 31.5  | -    | 13B  | 62.0  | 72.4  | -     | 34.2 | 34B  | -     | -     | -     | -    |
| MetaMath         | 7B   | 66.5  | -     | -     | 19.8 | 13B  | 72.3  | -     | -     | 22.4 | 34B  | -     | -     | -     | -    |
| MathCoder-L      | 7B   | 64.2  | 71.5  | -     | 23.3 | 13B  | 72.6  | 76.9  | -     | 29.9 | 34B  | -     | -     | -     | -    |
| MathCoder-CL     | 7B   | 67.8  | 70.7  | -     | 30.2 | 13B  | 74.1  | 78.0  | -     | 35.9 | 34B  | -     | -     | -     | -    |
| TORA             | 7B   | 68.8  | 68.2  | 73.9  | 40.1 | 13B  | 72.7  | 72.9  | 77.2  | 43.0 | 34B  | -     | -     | -     | -    |
| TORA-CODE        | 7B   | 72.6  | 70.4  | 78.7  | 44.6 | 13B  | 75.8  | 75.7  | 81.4  | 48.1 | 34B  | **80.7**   | **80.5** | 84.2 | **50.8** |
| **MMOS**             | 7B   | 69.9  | 73.4  | 76.8  | 40.2 | 13B  | 74.8  | 77.0  | 80.0  | 43.2 | 34B  | -     | -     | -     | -    |
| [MMOS-CODE](https://pan.quark.cn/s/ca1319076367)        | 7B   | 73.9  | 76.4  | 78.6  | 44.3 | 13B  | **77.1**  | **77.5**  | **81.9**  | **48.1** | 34B   | 80.4  | 80.6  | **85.1**  | 49.5 |
| **MMOS-MinCODE**     | 7B   | 70.3  | 72.5  | 76.7  | 44.6 | 13B  | -     | -     | -     | -    | 34B  | -     | -     | -     | -    |
| [MMOS-LLEMMA](https://pan.quark.cn/s/59024b402c1b)      | 7B   | **76.5**  | **77.7**  | **81.4**  | **48.8** | 13B  | -     | -     | -     | -    | 34B  | -     | -    | -     | -    |
| [MMOS-DeepSeekMath](https://pan.quark.cn/s/b939a0510658)      | 7B   | **80.5**  | **79.3**  | **87.6**  | **55.0** | 13B  | -     | -     | -     | -    | 34B  | -     | -    | -     | -    |

<!-- | LLAMA-2          | 7B   | 13.3  | 38.0  | 50.7  | 4.1  | 13B  | 24.3  | 43.1  | 56.3  | 6.3  | 34B  | -     | -     | -     | -    |
| Code Llama      | 7B   | 10.5  | -     | -     | 4.5    | 13B  | -    | -     | -     | -    | 34B  | 29.6     | -     | -     | 12.2   |
| LLEMMA     | 7B   | 36.4  | -     | -     | 18    | 13B  | -    | -     | -     | -    | 34B  | 51.5     | -     | -     | 25   |
| LLAMA-2 SFT      | 7B   | 41.3  | 31.9  | 47.4  | 7.2  | 13B  | 51.1  | 46.3  | 58.6  | 9.2  | 34B  | -     | -     | -     | -    |
| LLAMA-2 RFT      | 7B   | 50.3  | -     | -     | -    | 13B  | 55.3     | -     | -     | -    | 34B  | 57.9     | -     | -     | -    |
| Code Llama(PAL)      | 7B   | 27.1  | -     | -     | 17.2    | 13B  | -    | -     | -     | -    | 34B  | 52.7     | -     | -     | 23.5   |
| LLEMMA(PAL)     | 7B   | 40.1  | -     | -     | 21.5    | 13B  | -    | -     | -     | -    | 34B  | 62.6     | -     | -     | 27.1   | -->


## 💾 Install

    git clone https://github.com/cyzhh/MMOS.git
    cd MMOS
    conda create -n MMOS python=3.10 
    conda activate MMOS
    pip install -r requirements.txt

## 📚 Dataset

To identify the minimal optimal set, we follow these steps: 
1) Sample a sufficient number of correct reasoning paths to form initial set. 
2) Implement a deduplication algorithm to obtain its deduplicated subset. 
3) Conduct a statistical analysis on the upper limit of reasoning paths per question k with the subset data amount N. 
4) Perform SFT on several subsets to analyze the impact of removing duplicates and keeping varied reasoning paths.

We use [ToRA](https://github.com/microsoft/ToRA?tab=readme-ov-file) series to generate QA-pairs from open source dataset GSM8K, MATH, TAL-SCQ. The QA-pairs are processed by our  deduplication algorithm, resulting in the dataset `MMOS`. The total number of QA-pairs is **135K**.


The DATA, which we publish at [😊 HuggingFace](https://huggingface.co/datasets/cyzhh/MMOS), need to be placed under the relative path, `./train_data/MMOS/`.

If you are interested in our work, we will publish details about the data processing aspects after the paper is published.

## ⚙️ Auto Problem Generator

You can generate a data set for testing the numerical robustness of model performance by executing the following script command：

    bash scripts/generate.sh
    bash scripts/attack.sh
    bash scripts/rerank.sh

## 🚀 Training
Due to resource constraints, we performed supervised fine-tuning on [CodeLLaMA 7B](https://huggingface.co/codellama/CodeLlama-7b-Python-hf), [CodeLLaMA 13B](https://huggingface.co/codellama/CodeLlama-13b-Python-hf) and [CodeLLaMA 34B](https://huggingface.co/codellama/CodeLlama-34b-Python-hf) using our dataset on A100 40G GPUs. To reproduce our work from CodeLLaMA 7B/13B, you can train according to the following instruction. You can also train the 34B model through DDP script instructions.

    bash scripts/train_single.sh codellama 7b
    bash scripts/train.sh codellama 34b

## 💻 Inference

    bash scripts/infer.sh

## 📜 Citations

If you find this repository helpful, please consider citing our paper:

    @misc{chen2024empirical,
          title={An Empirical Study of Data Ability Boundary in LLMs' Math Reasoning}, 
          author={Zui Chen and Yezeng Chen and Jiaqi Han and Zhijie Huang and Ji Qi and Yi Zhou},
          year={2024},
          eprint={2403.00799},
          archivePrefix={arXiv},
          primaryClass={cs.CL}
    }

## 😇 Acknowledgements

- [ToRA](https://github.com/microsoft/ToRA?tab=readme-ov-file)

## 🌟Star History

[![Star History Chart](https://api.star-history.com/svg?repos=cyzhh/MMOS&type=Date)](https://star-history.com/#cyzhh/MMOS&Date)
