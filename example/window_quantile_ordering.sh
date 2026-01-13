python ./scripts/data_order.py \
    --input ./result/result-test.jsonl \
    --output ./result/ordering/window_quantile.jsonl \
    --batch_size 4 \
    --strategy window_quantile \
    --alpha 0.5 \
    --seed 42 \
