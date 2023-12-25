
OUTPUT=gsm_distribution
DATA=gsm_test

python -m src.attack \
    --method distribution \
    --direction ./data/${OUTPUT} \
    --data ${OUTPUT} \
    --original_path ./data/${DATA}/extract_main.jsonl \
    --remain_path ./data/${DATA}/extract_remain.jsonl \
    --save
