python ./scripts/data_order.py \
    --input ./result/result-test.jsonl \
    --output ./result/ordering/window_linear.jsonl \
    --batch_size 4 \
    --strategy window_linear \
    --alpha 0.5 \
    --seed 42 \
