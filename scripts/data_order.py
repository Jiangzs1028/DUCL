#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
import random
from typing import List, Dict, Any, Tuple


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at line {i+1}: {e}") from e
            if "DUE" not in obj:
                raise KeyError(f'Missing key "DUE" at line {i+1}')
            data.append(obj)
    return data


def write_jsonl(path: str, data: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for obj in data:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def percentile(sorted_vals: List[float], q: float) -> float:
    """
    q in [0, 1]. Uses linear interpolation between neighboring ranks.
    """
    if not sorted_vals:
        raise ValueError("Empty values for percentile")
    if q <= 0:
        return sorted_vals[0]
    if q >= 1:
        return sorted_vals[-1]
    n = len(sorted_vals)
    pos = (n - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_vals[lo]
    w = pos - lo
    return sorted_vals[lo] * (1 - w) + sorted_vals[hi] * w


def strategy_sort_asc(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Sort samples by DUE in ascending order
    return sorted(data, key=lambda x: float(x["DUE"]))


def strategy_fold(data_sorted: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
    """
    Fold-based reordering.

    data_sorted: data already sorted by DUE in ascending order.
    Example (k=2):
        indices: 0,2,4,6,... followed by 1,3,5,7,...
    """
    if k <= 0:
        raise ValueError("k must be >= 1")
    out = []
    n = len(data_sorted)
    for offset in range(k):
        for i in range(offset, n, k):
            out.append(data_sorted[i])
    return out


def window_linear(
    data: List[Dict[str, Any]],
    batch_size: int,
    alpha: float,
    seed: int,
) -> List[Dict[str, Any]]:
    """
    2.1 Linear window curriculum strategy:
    """
    rng = random.Random(seed)
    n = len(data)
    if n == 0:
        return []
    steps = math.ceil(n / batch_size)

    dues = [float(x["DUE"]) for x in data]
    dues_sorted = sorted(dues)
    p10 = percentile(dues_sorted, 0.10)
    p95 = percentile(dues_sorted, 0.95)

    # Number of steps required to reach the upper threshold
    # (at least 1 to avoid division by zero)
    reach = max(1, int(math.ceil(alpha * steps)))

    used = [False] * n
    remaining = n
    out = []

    def threshold_at(step_idx: int) -> float:
        # Linear interpolation within [p10, p95] for step_idx in [0, reach-1];
        # stays at p95 afterwards
        if step_idx <= 0:
            return p10
        if step_idx >= reach - 1:
            return p95
        t = step_idx / (reach - 1)
        return p10 + t * (p95 - p10)

    step = 0
    while remaining > 0:
        # The effective step index may advance early
        # if the current threshold does not yield enough samples
        step_local = step

        while True:
            thr = threshold_at(step_local)
            candidates = [
                i for i in range(n)
                if (not used[i]) and float(data[i]["DUE"]) <= thr
            ]
            if len(candidates) >= min(batch_size, remaining):
                break
            # Not enough samples: advance threshold to the next step
            if step_local < steps - 1:
                step_local += 1
                continue
            # Last step and still insufficient: take all remaining samples
            candidates = [i for i in range(n) if not used[i]]
            break

        take = min(batch_size, remaining, len(candidates))
        chosen = rng.sample(candidates, take) if take < len(candidates) else candidates
        for idx in chosen:
            used[idx] = True
            out.append(data[idx])
        remaining -= take

        # Move to the next nominal training step
        step += 1

    return out


def window_quantile(
    data: List[Dict[str, Any]],
    batch_size: int,
    alpha: float,
    seed: int,
) -> List[Dict[str, Any]]:
    """
    2.2 Quantile-based window curriculum strategy:
    """
    rng = random.Random(seed)
    n = len(data)
    if n == 0:
        return []
    steps = math.ceil(n / batch_size)
    reach = max(1, int(math.ceil(alpha * steps)))

    data_sorted = sorted(data, key=lambda x: float(x["DUE"]))

    used = [False] * n  # usage flags over sorted indices
    remaining = n
    out = []

    def pool_frac(step_idx: int) -> float:
        # Pool fraction increases linearly from 10% to 100%
        # for step_idx in [0, reach-1], and stays at 100% afterwards
        if step_idx <= 0:
            return 0.10
        if step_idx >= reach - 1:
            return 1.00
        t = step_idx / (reach - 1)
        return 0.10 + t * (1.00 - 0.10)

    step = 0
    while remaining > 0:
        step_local = step
        while True:
            frac = pool_frac(step_local)
            pool_size = max(1, int(math.floor(frac * n)))
            pool_size = min(pool_size, n)
            candidates = [i for i in range(pool_size) if not used[i]]
            if len(candidates) >= min(batch_size, remaining):
                break
            if step_local < steps - 1:
                step_local += 1
                continue
            candidates = [i for i in range(n) if not used[i]]
            break

        take = min(batch_size, remaining, len(candidates))
        chosen = rng.sample(candidates, take) if take < len(candidates) else candidates
        for si in chosen:
            used[si] = True
            out.append(data_sorted[si])
        remaining -= take
        step += 1

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, required=True, help="Input jsonl file")
    ap.add_argument("--output", type=str, required=True, help="Output sorted jsonl file")
    ap.add_argument("--batch_size", type=int, required=True, help="Training batch size")
    ap.add_argument(
        "--strategy",
        type=str,
        required=True,
        choices=["asc", "window_linear", "window_quantile", "fold"],
        help="Sorting strategy: asc | window_linear | window_quantile | fold",
    )
    ap.add_argument("--alpha", type=float, default=0.5, help="Alpha for window strategies (default: 0.5)")
    ap.add_argument("--k", type=int, default=3, help="Fold count k for fold strategy (default: 2)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for window sampling (default: 42)")
    args = ap.parse_args()

    if args.batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if args.alpha <= 0:
        raise ValueError("alpha must be > 0")

    data = read_jsonl(args.input)

    if args.strategy == "asc":
        out = strategy_sort_asc(data)
    elif args.strategy == "fold":
        out = strategy_fold(strategy_sort_asc(data), args.k)
    elif args.strategy == "window_linear":
        out = window_linear(data, args.batch_size, args.alpha, args.seed)
    elif args.strategy == "window_quantile":
        out = window_quantile(data, args.batch_size, args.alpha, args.seed)
    else:
        raise ValueError(f"Unknown strategy: {args.strategy}")

    write_jsonl(args.output, out)


if __name__ == "__main__":
    main()
