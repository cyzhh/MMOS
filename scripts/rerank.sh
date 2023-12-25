python -m src.rerank \
    --task train \
    --data mix30 \
    --source_files /data/cyz/create/gsm_distribution_5000/fix/extract_remain_changed_true_dataset.jsonl /data/ToRA/src/train_data/mix9/train.jsonl \
    --rate 1 1 \
    # --k 9 \
    # --seed 42 \
    # --rs