#!/usr/bin/env python3
"""
Generate a parameter sweep CSV for peregrine.py.
Computes total combinations (product of all param value list sizes), samples N
random indices in [0, total), and decodes each index to a parameter combo via
mixed-radix (product order: last key varies fastest). One row per combination.
Supports arbitrarily large total (e.g. > 10^20): no range(total) or product iteration;
sampling uses randrange and index decoding uses O(num_params) integer arithmetic.
"""

import argparse
import random

import pandas as pd

# Predefined values per parameter. Modify as needed.
PARAM_VALUES = {
    "int_reg_issue_width": list(range(1, 8 + 1)),
    "int_mult_div_issue_width": list(range(1, 8 + 1)),
    "fp_reg_issue_width": list(range(1, 8 + 1)),
    "fp_mult_div_issue_width": list(range(1, 8 + 1)),
    "read_port_issue_width": list(range(1, 8 + 1)),
    "rdwr_port_issue_width": list(range(1, 8 + 1)),
    "simd_unit_issue_width": [1],
    "fetch_width": list(range(1, 12 + 1)),
    "decode_width": list(range(1, 12 + 1)),
    "rename_width": list(range(1, 12 + 1)),
    "wb_width": list(range(1, 12 + 1)),
    "commit_width": list(range(1, 12 + 1)),
    "rob_size": list(range(1, 1024 + 1)),
    "lq_entries": list(range(1, 256 + 1)),
    "sq_entries": list(range(1, 256 + 1)),
    "branch_predictor": ["local", "tage"],
    "l1d_size": ["16KiB", "32KiB", "64KiB", "128KiB", "256KiB"],
    "l1i_size": ["16KiB", "32KiB", "64KiB", "128KiB", "256KiB"],
    "l2_size": ["512KiB", "1MiB", "2MiB", "4MiB"],
    "max_icache_fills": list(range(1, 32 + 1)),
    "stride_prefetcher_degree": [0, 4],
}

# Fixed order: same as itertools.product(*value_lists), last varies fastest
PARAM_KEYS = sorted(PARAM_VALUES.keys())


def total_combinations(param_values):
    """Total number of combinations (product of all value list sizes)."""
    n = 1
    for k in PARAM_KEYS:
        n *= len(param_values[k])
    return n


def _compute_strides(param_values):
    """Strides for mixed-radix decode: strides[i] = product of sizes[i+1:]."""
    sizes = [len(param_values[k]) for k in PARAM_KEYS]
    strides = [1] * len(sizes)
    for i in range(len(sizes) - 2, -1, -1):
        strides[i] = strides[i + 1] * sizes[i + 1]
    return sizes, strides


def index_to_combination(index, param_values, sizes=None, strides=None):
    """
    Decode a linear index into a parameter dict using mixed-radix.
    Order matches itertools.product(*value_lists) with PARAM_KEYS: last varies fastest.
    Pass precomputed sizes, strides (from _compute_strides) when decoding many indices
    to avoid recomputation and to keep working with huge totals.
    """
    value_lists = [param_values[k] for k in PARAM_KEYS]
    if sizes is None or strides is None:
        sizes, strides = _compute_strides(param_values)
    digits = [(index // strides[i]) % sizes[i] for i in range(len(sizes))]
    return dict(
        zip(PARAM_KEYS, [value_lists[i][d] for i, d in enumerate(digits)])
    )


# When n >= total we would return all indices; only materialize if total is small
_MAX_INDICES_MATERIALIZE = 10**9


def sample_random_indices(total, n):
    """
    Return n distinct random indices in [0, total) without materializing range(total).
    Safe for arbitrarily large total (e.g. > 10^20); uses randrange and a set.
    When n >= total, returns all indices only if total <= _MAX_INDICES_MATERIALIZE.
    """
    if n >= total:
        if total <= _MAX_INDICES_MATERIALIZE:
            return list(range(total))
        raise ValueError(
            f"total combinations ({total}) is very large; "
            "num_combinations must be less than total (use -n to sample a subset)."
        )
    chosen = []
    seen = set()
    while len(chosen) < n:
        idx = random.randrange(total)
        if idx not in seen:
            seen.add(idx)
            chosen.append(idx)
    return chosen


def main():
    parser = argparse.ArgumentParser(
        description="Generate parameter sweep CSV for peregrine.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-n",
        "--num-combinations",
        type=int,
        default=2**14,
        help="Number of combination indices to sample. Capped at total combinations.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="param_sweep.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=262,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    total = total_combinations(PARAM_VALUES)
    n = min(args.num_combinations, total)

    print(f"Total combinations: {total}. Sampling {n} random indices.")

    # N distinct random indices in [0, total); safe for huge total (no range(total))
    random_indices = sample_random_indices(total, n)
    # Precompute strides once for decoding (avoids repeated big-int work for huge total)
    sizes, strides = _compute_strides(PARAM_VALUES)
    combinations = [
        index_to_combination(k, PARAM_VALUES, sizes=sizes, strides=strides)
        for k in random_indices
    ]

    # One row per combination
    df = pd.DataFrame(combinations)
    df = df[PARAM_KEYS]
    df.to_csv(args.output, index=False)
    print(f"Wrote {len(df)} rows to {args.output}")


if __name__ == "__main__":
    main()
