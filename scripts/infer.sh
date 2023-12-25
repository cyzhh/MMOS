set -ex

#MODEL_NAME_OR_PATH="/data/pretrain_model/tora-7b-code"
#MODEL_NAME_OR_PATH="/data/pretrain_model/tora-13b-code"
#MODEL_NAME_OR_PATH="/data/pretrain_model/tora-code-34b-v1.0"
MODEL_NAME_OR_PATH="/data/pretrain_model/tora-70b"

DATA="gsm8k"

SPLIT="test"
PROMPT_TYPE="tora"
NUM_TEST_SAMPLE=-1


CUDA_VISIBLE_DEVICES=4,5,6,7  TOKENIZERS_PARALLELISM=false \
python -m infer.inference \
--model_name_or_path ${MODEL_NAME_OR_PATH} \
--data ${DATA} \
--split ${SPLIT} \
--prompt_type ${PROMPT_TYPE} \
--use_train_prompt_format \
--num_test_sample ${NUM_TEST_SAMPLE} \
--seed 0 \
--temperature 0.9 \
--n_sampling 100 \
--top_p 0.95 \
--start 0 \
--end -1 \
--test_id 0 \
