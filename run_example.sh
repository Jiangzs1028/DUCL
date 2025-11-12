CUDA_VISIBLE_DEVICES=7 python main.py \
    --data ./data/pile-math-500.jsonl \
    --initial_data ./data/fineweb-edu-500.jsonl \
    --source_val ./data/fineweb-edu-500.jsonl \
    --cache_path ./temp \
    --result_path ./result/result.jsonl