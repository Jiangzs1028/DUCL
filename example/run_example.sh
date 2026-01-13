CUDA_VISIBLE_DEVICES=7 python main.py \
    --data ./data/train_dataset.jsonl \
    --source_val ./data/source_dataset.jsonl \
    --target_val ./data/target_val.jsonl \
    --cache_path ./temp \
    --result_path ./result/result.jsonl \
    --encoder_path sentence-transformers/all-MiniLM-L12-v2 \
    --chunk_num 8 \
    --batch_size 100 \
