#!/usr/bin/env python3
"""
Generate simulation sweep CSVs for peregrine.py.

Outputs:
1) Checkpoint generation CSV (one row per benchmark), where each row contains a
   comma-separated list of checkpoint instruction counts for one gem5 run.
2) Region simulation CSV (one row per sampled region), with random architecture
   parameter combinations and the nearest usable checkpoint + fast-forward distance.
"""

import argparse
import bisect
import random
from dataclasses import dataclass
from typing import (
    Dict,
    List,
    Sequence,
    Tuple,
)

import pandas as pd
from gen_param_sweep import (
    PARAM_KEYS,
    PARAM_VALUES,
    _compute_strides,
    index_to_combination,
    sample_random_indices,
    total_combinations,
)

benchmark_lengths: Dict[str, Tuple[int, int]] = {
    "505.mcf_r": (300_000_000, 1_570_000_000),
    "520.omnetpp_r": (100_000_000, 12_195_565_497),
    "523.xalancbmk_r": (100_000_000, 324_007_592),
    "541.leela_r": (100_000_000, 23_772_938_110),
    "548.exchange2_r": (7_505_000_000, 20_205_000_000),
    "531.deepsjeng_r": (100_000_000, 637_116_693),
    "557.xz_r": (100_000_000, 791_549_354),
    "525.x264_r": (10_000_000_000, 22_700_000_000),
    "502.gcc_r": (1_000_000, 15_000_000),
}


@dataclass(frozen=True)
class Region:
    benchmark: str
    start: int
    end: int


def allocate_region_counts(
    lengths: Dict[str, Tuple[int, int]], num_regions: int
) -> Dict[str, int]:
    """Allocate per-benchmark counts proportional to benchmark span."""
    spans = {b: stop - start for b, (start, stop) in lengths.items()}
    total_span = sum(spans.values())
    if total_span <= 0:
        raise ValueError("Total benchmark span must be positive.")

    raw = {b: (num_regions * spans[b] / total_span) for b in lengths}
    base = {b: int(raw[b]) for b in lengths}
    assigned = sum(base.values())
    remainder = num_regions - assigned

    if remainder > 0:
        order = sorted(
            lengths.keys(),
            key=lambda b: (raw[b] - base[b]),
            reverse=True,
        )
        for i in range(remainder):
            base[order[i % len(order)]] += 1
    elif remainder < 0:
        order = sorted(
            lengths.keys(),
            key=lambda b: (raw[b] - base[b]),
        )
        for i in range(-remainder):
            if base[order[i % len(order)]] == 0:
                continue
            base[order[i % len(order)]] -= 1

    return base


def sample_non_overlapping_starts(
    start: int,
    stop: int,
    region_length: int,
    n_regions: int,
    rng: random.Random,
) -> List[int]:
    """
    Sample n non-overlapping region starts in [start, stop], where each region is
    [s, s + region_length). Uses slot sampling for guaranteed non-overlap.
    """
    span = stop - start
    if span < region_length:
        if n_regions > 0:
            raise ValueError(
                f"Benchmark span [{start}, {stop}) is smaller than region_length={region_length}."
            )
        return []

    # Integer start positions in [0, U], U = span - region_length.
    # For non-overlap with fixed region_length L:
    #   s_{i+1} - s_i >= L
    # Let x_i be n distinct integers sampled from [0, K], sorted asc, where:
    #   K = U - (n - 1) * L
    # Then s_i = x_i + i * L are valid non-overlapping starts with 1-inst granularity.
    u = span - region_length
    max_regions = span // region_length
    if n_regions > max_regions:
        raise ValueError(
            f"Requested {n_regions} non-overlapping regions but max is {max_regions} "
            f"for span [{start}, {stop}) and region_length={region_length}."
        )
    if n_regions == 0:
        return []

    k = u - (n_regions - 1) * region_length
    if k < 0:
        raise ValueError(
            "No feasible placement for non-overlapping regions with current settings."
        )

    xs = sorted(rng.sample(range(k + 1), n_regions))
    starts = [start + x + i * region_length for i, x in enumerate(xs)]
    return starts


def sample_regions(
    lengths: Dict[str, Tuple[int, int]],
    num_regions: int,
    region_length: int,
    min_fast_forward: int,
    seed: int,
) -> Dict[str, List[Region]]:
    """Sample regions proportionally by benchmark span, with no overlap per benchmark."""
    rng = random.Random(seed)
    per_bench_counts = allocate_region_counts(lengths, num_regions)

    regions_by_benchmark: Dict[str, List[Region]] = {}
    for bench, (bench_start, bench_stop) in lengths.items():
        n_bench = per_bench_counts[bench]
        # Guarantee at least min_fast_forward instructions are available from a
        # checkpoint at or after benchmark start.
        sample_start = bench_start + min_fast_forward
        if sample_start >= bench_stop:
            if n_bench > 0:
                raise ValueError(
                    f"Benchmark {bench} has no room after min_fast_forward={min_fast_forward}."
                )
            regions_by_benchmark[bench] = []
            continue
        starts = sample_non_overlapping_starts(
            sample_start, bench_stop, region_length, n_bench, rng
        )
        regions = [
            Region(benchmark=bench, start=s, end=s + region_length)
            for s in starts
        ]
        regions_by_benchmark[bench] = regions
    return regions_by_benchmark


def checkpoints_for_max_distance(
    starts: Sequence[int], max_distance: int
) -> Tuple[List[int], List[int]]:
    """
    Build minimal checkpoint list with greedy coverage under:
    checkpoint <= region_start and region_start - checkpoint <= max_distance.
    Returns (checkpoints, distances_per_start).
    """
    if not starts:
        return [], []

    starts_sorted = sorted(starts)
    checkpoints: List[int] = []
    distances: List[int] = []

    i = 0
    while i < len(starts_sorted):
        cpt = starts_sorted[i]
        checkpoints.append(cpt)
        j = i
        while (
            j < len(starts_sorted) and (starts_sorted[j] - cpt) <= max_distance
        ):
            distances.append(starts_sorted[j] - cpt)
            j += 1
        i = j

    return checkpoints, distances


def choose_checkpoints(
    starts: Sequence[int],
    min_fast_forward: int,
    target_distance: int,
    max_distance: int,
    distance_step: int,
    distance_weight: float,
    count_weight: float,
) -> List[int]:
    """
    Search max-distance budget and pick checkpoint plan minimizing weighted score:
      score = distance_weight * mean_distance + count_weight * num_checkpoints
    """
    if not starts:
        return []
    if distance_step <= 0:
        raise ValueError("--checkpoint-distance-step must be > 0")
    if min_fast_forward < 1:
        raise ValueError("--min-fast-forward must be >= 1")

    best_score = None
    best_checkpoints: List[int] = []
    effective_starts = [s - min_fast_forward for s in starts]
    d = target_distance
    while d <= max_distance:
        cpts, distances = checkpoints_for_max_distance(effective_starts, d)
        # Shift back so score reflects actual fast-forward to region starts.
        shifted_distances = [dist + min_fast_forward for dist in distances]
        mean_distance = sum(shifted_distances) / len(shifted_distances)
        score = distance_weight * mean_distance + count_weight * len(cpts)
        if best_score is None or score < best_score:
            best_score = score
            best_checkpoints = cpts
        d += distance_step

    return best_checkpoints


def nearest_checkpoint_and_ff(
    checkpoints: Sequence[int], start_inst: int, min_fast_forward: int
) -> Tuple[int, int]:
    """Find nearest usable checkpoint (<= start_inst - min_fast_forward)."""
    if not checkpoints:
        raise ValueError("Checkpoint list is empty.")
    if min_fast_forward < 1:
        raise ValueError("--min-fast-forward must be >= 1")

    max_cpt = start_inst - min_fast_forward
    idx = bisect.bisect_right(checkpoints, max_cpt) - 1
    if idx < 0:
        raise ValueError(
            f"No checkpoint exists <= start_inst - min_fast_forward "
            f"({start_inst} - {min_fast_forward})."
        )
    cpt = checkpoints[idx]
    ff = start_inst - cpt
    if ff < min_fast_forward:
        raise ValueError(
            f"Computed fast-forward {ff} < min_fast_forward {min_fast_forward}."
        )
    return cpt, ff


def sample_param_rows(
    n_rows: int, seed: int, with_replacement: bool
) -> List[Dict[str, object]]:
    """Sample random parameter combinations using gen_param_sweep helpers."""
    if n_rows <= 0:
        return []

    rng = random.Random(seed)
    random.seed(seed)
    total = total_combinations(PARAM_VALUES)
    sizes, strides = _compute_strides(PARAM_VALUES)

    if with_replacement:
        indices = [rng.randrange(total) for _ in range(n_rows)]
    else:
        n_unique = min(n_rows, total)
        indices = sample_random_indices(total, n_unique)
        if n_unique < n_rows:
            extra = [rng.randrange(total) for _ in range(n_rows - n_unique)]
            indices.extend(extra)

    rows = [
        index_to_combination(idx, PARAM_VALUES, sizes=sizes, strides=strides)
        for idx in indices
    ]
    return rows


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate region and checkpoint sweep CSVs for peregrine.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-n",
        "--num-regions",
        type=int,
        default=100_000,
        help="Total number of regions to sample across all benchmarks.",
    )
    parser.add_argument(
        "--region-length",
        type=int,
        default=100_000,
        help="Length of each sampled simulation region (instructions).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=262,
        help="Seed for region sampling.",
    )
    parser.add_argument(
        "--param-seed",
        type=int,
        default=263,
        help="Seed for random parameter combinations.",
    )
    parser.add_argument(
        "--param-with-replacement",
        action="store_true",
        default=False,
        help="Allow repeated random parameter combinations when sampling rows.",
    )
    parser.add_argument(
        "--checkpoint-target-distance",
        type=int,
        default=10_000_000,
        help="Target max distance from checkpoint to region start (instructions).",
    )
    parser.add_argument(
        "--checkpoint-max-distance",
        type=int,
        default=100_000_000,
        help="Largest max-distance candidate searched by optimizer.",
    )
    parser.add_argument(
        "--checkpoint-distance-step",
        type=int,
        default=1_000_000,
        help="Step size while searching checkpoint max-distance candidates.",
    )
    parser.add_argument(
        "--checkpoint-distance-weight",
        type=float,
        default=1.0,
        help="Weight for mean checkpoint-to-start distance in optimization score.",
    )
    parser.add_argument(
        "--checkpoint-count-weight",
        type=float,
        default=10_000.0,
        help="Weight for number of checkpoints in optimization score.",
    )
    parser.add_argument(
        "--min-fast-forward",
        type=int,
        default=100_000,
        help=(
            "Minimum fast-forward instructions from checkpoint to region start. "
            "Must be >= 1."
        ),
    )
    parser.add_argument(
        "--output-checkpoint-csv",
        type=str,
        default="sim_checkpoint_sweep.csv",
        help="Output CSV for checkpoint generation runs (one row per benchmark).",
    )
    parser.add_argument(
        "--output-checkpoint-points-csv",
        type=str,
        default="sim_checkpoint_points.csv",
        help="Output CSV with one row per benchmark/checkpoint pair.",
    )
    parser.add_argument(
        "--output-sim-csv",
        type=str,
        default="sim_region_param_sweep.csv",
        help="Output CSV for full region simulation runs.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.num_regions <= 0:
        raise ValueError("--num-regions must be > 0")
    if args.region_length <= 0:
        raise ValueError("--region-length must be > 0")
    if args.min_fast_forward < 1:
        raise ValueError("--min-fast-forward must be >= 1")
    if args.checkpoint_target_distance <= 0:
        raise ValueError("--checkpoint-target-distance must be > 0")
    if args.checkpoint_max_distance < args.checkpoint_target_distance:
        raise ValueError(
            "--checkpoint-max-distance must be >= --checkpoint-target-distance"
        )

    regions_by_benchmark = sample_regions(
        lengths=benchmark_lengths,
        num_regions=args.num_regions,
        region_length=args.region_length,
        min_fast_forward=args.min_fast_forward,
        seed=args.seed,
    )

    checkpoint_rows = []
    checkpoint_point_rows = []
    benchmark_checkpoints: Dict[str, List[int]] = {}

    for bench, regions in regions_by_benchmark.items():
        starts = [r.start for r in regions]
        cpts = choose_checkpoints(
            starts=starts,
            min_fast_forward=args.min_fast_forward,
            target_distance=args.checkpoint_target_distance,
            max_distance=args.checkpoint_max_distance,
            distance_step=args.checkpoint_distance_step,
            distance_weight=args.checkpoint_distance_weight,
            count_weight=args.checkpoint_count_weight,
        )
        benchmark_checkpoints[bench] = cpts

        checkpoint_rows.append(
            {
                "benchmark": bench,
                "num_regions": len(regions),
                "num_checkpoints": len(cpts),
                "checkpoints": ",".join(str(x) for x in cpts),
            }
        )
        for cpt in cpts:
            checkpoint_point_rows.append(
                {
                    "benchmark": bench,
                    "checkpoint": cpt,
                }
            )

    all_regions: List[Region] = []
    for bench_regions in regions_by_benchmark.values():
        all_regions.extend(bench_regions)
    all_regions.sort(key=lambda r: (r.benchmark, r.start))

    param_rows = sample_param_rows(
        n_rows=len(all_regions),
        seed=args.param_seed,
        with_replacement=args.param_with_replacement,
    )

    sim_rows = []
    for i, region in enumerate(all_regions):
        cpt, ff = nearest_checkpoint_and_ff(
            benchmark_checkpoints[region.benchmark],
            region.start,
            args.min_fast_forward,
        )
        row = {
            "benchmark": region.benchmark,
            "region_start": region.start,
            "region_end": region.end,
            "max_insts": args.region_length,
            "checkpoint": cpt,
            "fast_forward": ff,
        }
        row.update(param_rows[i])
        sim_rows.append(row)

    checkpoint_df = pd.DataFrame(checkpoint_rows).sort_values("benchmark")
    checkpoint_df.to_csv(args.output_checkpoint_csv, index=False)

    checkpoint_points_df = pd.DataFrame(checkpoint_point_rows).sort_values(
        ["benchmark", "checkpoint"]
    )
    checkpoint_points_df.to_csv(args.output_checkpoint_points_csv, index=False)

    sim_df = pd.DataFrame(sim_rows)
    sim_df = sim_df[
        [
            "benchmark",
            "checkpoint",
            "fast_forward",
            *PARAM_KEYS,
        ]
    ]
    sim_df.to_csv(args.output_sim_csv, index=False)

    print(
        f"Wrote {len(checkpoint_df)} benchmark rows to {args.output_checkpoint_csv}"
    )
    print(
        "Wrote "
        f"{len(checkpoint_points_df)} checkpoint rows to {args.output_checkpoint_points_csv}"
    )
    print(f"Wrote {len(sim_df)} simulation rows to {args.output_sim_csv}")


if __name__ == "__main__":
    main()
