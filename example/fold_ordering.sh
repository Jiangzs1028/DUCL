python ./scripts/data_order.py \
    --input ./result/result-test.jsonl \
    --output ./result/ordering/fold.jsonl \
    --batch_size 4 \
    --strategy fold \
    --k 3 \
    --seed 42 \
