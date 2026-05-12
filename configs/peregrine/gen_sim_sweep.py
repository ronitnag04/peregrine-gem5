#!/usr/bin/env python3
"""
Generate simulation sweep CSVs for peregrine.py.

Outputs:
1) Checkpoint generation CSV (one row per benchmark), where each row contains a
   comma-separated list of checkpoint instruction counts for one gem5 run.
2) Region simulation CSV (one row per sampled region), with architecture
   parameter combinations and the nearest usable checkpoint + fast-forward
   distance.

Architecture parameter rows are drawn by ``gen_param_sweep.sample_combinations``;
the sampling strategy is selected via ``--param-sampling-mode``. See the
module docstring in gen_param_sweep.py for the full set of modes (``grid``,
``valid-uniform``, ``valid-lhs``, ``valid-cost-stratified``).
"""

import argparse
import bisect
import os
import random
from dataclasses import dataclass
from typing import (
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
)

import pandas as pd
from gen_param_sweep import (
    PARAM_KEYS,
    PARAM_VALUES,
    SAMPLING_MODES,
    sample_combinations,
    total_combinations,
)

spec_benchmark_lengths: Dict[str, Tuple[int, int]] = {
    "505.mcf_r": (300_000_000, 1_570_000_000),
    "520.omnetpp_r": (100_000_000, 12_195_565_497),
    "523.xalancbmk_r": (100_000_000, 324_007_592),
    "541.leela_r": (100_000_000, 23_772_938_110),
    "548.exchange2_r": (7_505_000_000, 20_205_000_000),
    "531.deepsjeng_r": (100_000_000, 637_116_693),
    "557.xz_r": (100_000_000, 791_549_354),
    "525.x264_r": (10_000_000_000, 22_700_000_000),
    # "502.gcc_r": (1_000_000, 15_000_000),
}

adversarial_benchmark_lengths: Dict[str, Tuple[int, int]] = {
    "adversarial_branches": (100_000_000, 9_000_000_000),
    "icache_blast": (100_000_000, 9_600_000_000),
    "many_pages_streaming": (100_000_000, 6_800_000_000),
    "pow2_stride_benign": (100_000_000, 9_200_000_000),
    "pow2_stride_thrash": (100_000_000, 9_300_000_000),
    "ptrchase_rand": (100_000_000, 1_900_000_000),
    "serial_mul_chain": (100_000_000, 9_600_000_000),
    "stlf_misalign": (100_000_000, 9_600_000_000),
}

BENCHMARK_SETS: Dict[str, Dict[str, Tuple[int, int]]] = {
    "spec": spec_benchmark_lengths,
    "adversarial": adversarial_benchmark_lengths,
}


@dataclass(frozen=True)
class Region:
    benchmark: str
    start: int
    end: int


def allocate_region_counts(
    lengths: Dict[str, Tuple[int, int]],
    num_regions: int,
    even: bool = False,
) -> Dict[str, int]:
    """
    Allocate per-benchmark region counts.

    Default (`even=False`): proportional to each benchmark's instruction
    span. Suited to SPEC where benchmark lengths differ by orders of
    magnitude and proportional coverage yields a more representative
    training distribution.

    `even=True`: evenly divide `num_regions` across benchmarks. Any
    remainder is spread across the first benchmarks in sorted order so
    the allocation is deterministic. Use this for the adversarial set,
    where each benchmark is a targeted probe of a distinct feature
    blind spot and deserves equal representation in training.
    """
    benches = sorted(lengths.keys())
    if even:
        base = {b: num_regions // len(benches) for b in benches}
        remainder = num_regions - sum(base.values())
        for i in range(remainder):
            base[benches[i]] += 1
        return base

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
    even: bool = False,
) -> Dict[str, List[Region]]:
    """Sample regions with either proportional or even per-benchmark allocation."""
    rng = random.Random(seed)
    per_bench_counts = allocate_region_counts(lengths, num_regions, even=even)

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


def load_existing_checkpoints(
    checkpoint_dir: str,
    benchmarks: Sequence[str],
) -> Dict[str, List[int]]:
    """Load checkpoint instruction counts from ``<checkpoint_dir>/<benchmark>/cpt.<n>/``.

    Each benchmark's subdirectory is expected to contain entries named
    ``cpt.<instruction_count>``; the numeric suffix is extracted and returned
    sorted ascending. Benchmarks with no subdirectory return an empty list.
    """
    if not os.path.isdir(checkpoint_dir):
        raise ValueError(
            f"--checkpoint-dir {checkpoint_dir!r} is not a directory."
        )

    result: Dict[str, List[int]] = {}
    for bench in benchmarks:
        bench_dir = os.path.join(checkpoint_dir, bench)
        if not os.path.isdir(bench_dir):
            result[bench] = []
            continue
        cpts: List[int] = []
        for entry in os.listdir(bench_dir):
            if not entry.startswith("cpt."):
                continue
            suffix = entry[len("cpt.") :]
            if not suffix.isdigit():
                continue
            cpts.append(int(suffix))
        cpts.sort()
        result[bench] = cpts
    return result


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
    n_rows: int,
    seed: int,
    mode: str = "grid",
    with_replacement: bool = False,
    cost_stratified_oversample: int = 8,
    cost_stratified_bins: int = None,
) -> List[Dict[str, object]]:
    """Sample ``n_rows`` parameter combinations via gen_param_sweep's dispatcher.

    ``mode`` selects the sampling strategy; see
    ``gen_param_sweep.SAMPLING_MODES``. ``with_replacement`` applies only to
    ``grid`` mode (feasibility-aware modes are always with-replacement against
    the feasible region). Cost-stratified mode accepts its oversample/bin knobs.
    Prints a one-line summary of the draw so sweep CSVs are traceable.
    """
    if n_rows <= 0:
        return []

    rng = random.Random(seed)
    kwargs = {}
    if mode == "grid":
        kwargs["with_replacement"] = with_replacement
    elif mode == "valid-cost-stratified":
        kwargs["oversample"] = cost_stratified_oversample
        kwargs["n_bins"] = cost_stratified_bins

    rows, stats = sample_combinations(
        PARAM_VALUES, n_rows, mode=mode, rng=rng, **kwargs
    )
    total = total_combinations(PARAM_VALUES)
    if mode == "grid":
        print(
            f"Param sampling [{mode}]: {stats['n']} configs from grid of "
            f"{total:.2e} (feasibility filter off)."
        )
    elif mode == "valid-cost-stratified":
        print(
            f"Param sampling [{mode}]: pool={stats['pool_n']} "
            f"bins={stats['n_bins']} → {stats['n']} configs "
            f"(valid subspace: {stats['total_valid']:.2e})."
        )
    else:
        print(
            f"Param sampling [{mode}]: {stats['n']} configs from valid "
            f"subspace of {stats['total_valid']:.2e} "
            f"({stats['total_valid'] / total:.2%} of raw space)."
        )
    return rows


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate region and checkpoint sweep CSVs for peregrine.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--benchmark-set",
        type=str,
        default="spec",
        choices=sorted(BENCHMARK_SETS.keys()),
        help=(
            "Which benchmark-length table to sweep: 'spec' (SPEC CPU 2017) "
            "or 'adversarial' (adversarial/probe benchmarks in "
            "/home/ubuntu/peregrine/benchmarks)."
        ),
    )
    parser.add_argument(
        "--even-distribution",
        action="store_true",
        default=False,
        help=(
            "Distribute region samples evenly across benchmarks instead of "
            "weighting by instruction span. Recommended for the adversarial "
            "set where each benchmark probes a distinct feature blind spot."
        ),
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
        "--param-sampling-mode",
        type=str,
        default="grid",
        choices=list(SAMPLING_MODES),
        help=(
            "Architecture parameter sampling strategy. 'grid' is uniform iid "
            "over the raw grid; 'valid-uniform' / 'valid-lhs' / "
            "'valid-cost-stratified' restrict to the feasible region defined "
            "by optimize_hw_config.is_valid. See gen_param_sweep.py for full "
            "documentation of each mode."
        ),
    )
    parser.add_argument(
        "--param-with-replacement",
        action="store_true",
        default=False,
        help=(
            "For --param-sampling-mode=grid only: allow repeated random "
            "parameter combinations. Ignored by feasibility-aware modes."
        ),
    )
    parser.add_argument(
        "--param-cost-stratified-oversample",
        type=int,
        default=8,
        help=(
            "Oversample factor for valid-cost-stratified mode: draws "
            "factor*N candidates, bins by hardware_cost, subsamples to N."
        ),
    )
    parser.add_argument(
        "--param-cost-stratified-bins",
        type=int,
        default=None,
        help=(
            "Number of cost-quantile bins for valid-cost-stratified mode. "
            "Defaults to min(N, 20)."
        ),
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help=(
            "Directory of pre-existing checkpoints laid out as "
            "<dir>/<benchmark>/cpt.<instruction_count>/. When set, the "
            "sampler uses these checkpoints directly instead of choosing "
            "new ones, and the --checkpoint-target-distance / "
            "--checkpoint-max-distance / --checkpoint-distance-step / "
            "--checkpoint-*-weight flags are ignored. Benchmarks with no "
            "subdirectory in this tree are dropped from the sweep."
        ),
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
        "-o",
        "--output-dir",
        type=str,
        default="sim_sweep",
        help=(
            "Directory to write sweep CSVs into. Creates "
            "sim_checkpoint_sweep.csv, sim_checkpoint_points.csv, and "
            "sim_region_param_sweep.csv inside this directory."
        ),
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

    lengths = BENCHMARK_SETS[args.benchmark_set]

    preloaded_checkpoints: Optional[Dict[str, List[int]]] = None
    if args.checkpoint_dir is not None:
        preloaded_checkpoints = load_existing_checkpoints(
            args.checkpoint_dir, sorted(lengths.keys())
        )
        missing = [b for b, cpts in preloaded_checkpoints.items() if not cpts]
        if missing:
            print(
                f"--checkpoint-dir {args.checkpoint_dir}: no checkpoints "
                f"found for {sorted(missing)}; dropping from sweep."
            )
        lengths = {b: lengths[b] for b in lengths if preloaded_checkpoints[b]}
        preloaded_checkpoints = {b: preloaded_checkpoints[b] for b in lengths}

    regions_by_benchmark = sample_regions(
        lengths=lengths,
        num_regions=args.num_regions,
        region_length=args.region_length,
        min_fast_forward=args.min_fast_forward,
        seed=args.seed,
        even=args.even_distribution,
    )

    checkpoint_rows = []
    checkpoint_point_rows = []
    benchmark_checkpoints: Dict[str, List[int]] = {}

    for bench, regions in regions_by_benchmark.items():
        if preloaded_checkpoints is not None:
            cpts = preloaded_checkpoints[bench]
        else:
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
        mode=args.param_sampling_mode,
        with_replacement=args.param_with_replacement,
        cost_stratified_oversample=args.param_cost_stratified_oversample,
        cost_stratified_bins=args.param_cost_stratified_bins,
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

    os.makedirs(args.output_dir, exist_ok=True)
    output_checkpoint_csv = os.path.join(
        args.output_dir, "sim_checkpoint_sweep.csv"
    )
    output_checkpoint_points_csv = os.path.join(
        args.output_dir, "sim_checkpoint_points.csv"
    )
    output_sim_csv = os.path.join(
        args.output_dir, "sim_region_param_sweep.csv"
    )

    checkpoint_df = pd.DataFrame(checkpoint_rows).sort_values("benchmark")
    checkpoint_df.to_csv(output_checkpoint_csv, index=False)

    checkpoint_points_df = pd.DataFrame(checkpoint_point_rows).sort_values(
        ["benchmark", "checkpoint"]
    )
    checkpoint_points_df.to_csv(output_checkpoint_points_csv, index=False)

    sim_df = pd.DataFrame(sim_rows)
    sim_df = sim_df[
        [
            "benchmark",
            "checkpoint",
            "fast_forward",
            *PARAM_KEYS,
        ]
    ]
    sim_df.to_csv(output_sim_csv, index=False)

    print(
        f"Wrote {len(checkpoint_df)} benchmark rows to {output_checkpoint_csv}"
    )
    print(
        "Wrote "
        f"{len(checkpoint_points_df)} checkpoint rows to {output_checkpoint_points_csv}"
    )
    print(f"Wrote {len(sim_df)} simulation rows to {output_sim_csv}")


if __name__ == "__main__":
    main()
