#!/usr/bin/env python3
"""
Randomly sample program regions of a fixed size from a total instruction stream.

Each sampled region is defined by its start instruction index (inclusive). Can be
configured to disallow overlapping regions.

Usage:
    python sample_regions.py

Examples:
    python sample_regions.py --total-insts 1000000
    python sample_regions.py --total-insts 5000000 --num-samples 200 --region-size 1000000
    python sample_regions.py --total-insts 5000000 --output my_regions.csv --seed 42
"""

import argparse
import random
import sys

import pandas as pd


def sample_regions(
    total_instructions: int,
    num_samples: int,
    region_size: int,
    seed: int | None = None,
) -> list[int]:
    if region_size > total_instructions:
        raise ValueError(
            f"Region size ({region_size}) exceeds "
            f"total instructions ({total_instructions})."
        )

    rng = random.Random(seed)
    max_start = total_instructions - region_size
    regions = []
    seen = set()

    while len(regions) < num_samples:
        start = rng.randint(0, max_start)
        if start not in seen:
            regions.append(start)
            seen.add(start)

    return regions


def sample_regions_wo_overlap(
    total_instructions: int,
    num_samples: int,
    region_size: int,
    seed: int | None = None,
) -> list[int]:
    """
    Samples non-overlapping program regions by dividing the program
    into consecutive region-sized chunks and randomly sampling from
    those. So not truly uniform but close enough at scale.
    """
    if region_size > total_instructions:
        raise ValueError(
            f"Region size ({region_size}) exceeds "
            f"total instructions ({total_instructions})."
        )

    chunks = total_instructions // region_size

    if chunks < num_samples:
        raise ValueError(
            f"Number of consecutive region-size ({region_size}) "
            f"chunks in program of {total_instructions} "
            f"instructions is less than the requested number of "
            f"samples ({num_samples})."
        )

    rng = random.Random(seed)
    regions = []
    seen = set()

    while len(regions) < num_samples:
        start = rng.randint(0, chunks) * region_size
        if start not in seen:
            regions.append(start)
            seen.add(start)

    return regions


def main():
    parser = argparse.ArgumentParser(
        description="Randomly sample program regions from an instruction stream.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-ti",
        "--total-instructions",
        type=int,
        help="Total number of instructions in the program.",
    )
    parser.add_argument(
        "-ns",
        "--num-samples",
        type=int,
        default=100,
        help="Number of regions to sample.",
    )
    parser.add_argument(
        "-rs",
        "--region-size",
        type=int,
        default=100_000,
        help="Number of instructions per region.",
    )
    parser.add_argument(
        "-no",
        "--no-overlap",
        action="store_true",
        default=False,
        help="Guarantee non-overlapping regions.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="regions.csv",
        help="Output CSV file path.",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility.",
    )

    args = parser.parse_args()

    if args.total_instructions <= 0:
        print(
            "Error: total_instructions must be a positive integer.",
            file=sys.stderr,
        )
        sys.exit(1)
    if args.num_samples <= 0:
        print(
            "Error: --num-samples must be a positive integer.", file=sys.stderr
        )
        sys.exit(1)
    if args.region_size <= 0:
        print(
            "Error: --region-size must be a positive integer.", file=sys.stderr
        )
        sys.exit(1)

    try:
        if args.no_overlap:
            regions = sample_regions_wo_overlap(
                total_instructions=args.total_instructions,
                num_samples=args.num_samples,
                region_size=args.region_size,
                seed=args.seed,
            )
        else:
            regions = sample_regions(
                total_instructions=args.total_instructions,
                num_samples=args.num_samples,
                region_size=args.region_size,
                seed=args.seed,
            )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print(
        f"Sampled {len(regions):,} region(s) of {args.region_size:,} instructions "
        f"from a stream of {args.total_instructions:,} total instructions."
    )

    df = pd.DataFrame(regions)
    df.to_csv(args.output, index=False)

    print(f"Output written to: {args.output}")


if __name__ == "__main__":
    main()
