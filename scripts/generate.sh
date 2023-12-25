DATA=gsm_test

# source_file 一直往后加就完事了
python -m src.generate \
    --data ${DATA} \
    --source_files /data/ToRA/src/outputs/pretrain_model/tora-70b/gsm8k/cz993_test_tora_-1_seed0_t0.9_s0_e1319_11-18_10-40.jsonl 


python -m src.nodup \
    --input_file ./data/${DATA}/extract.jsonl \
    --save \
    --topk 3