#!/usr/bin/env bash
# Generate instruction traces for all Peregrine microbenchmarks.
#
# For each benchmark and checkpoint, this restores the checkpoint produced by
# spec_checkpoint_benchmarks.sh at multiple fast-forward points, then runs:
#   build/X86/gem5.opt configs/peregrine/peregrine.py --trace --restore-checkpoint ...
#
# The resulting trace file produced by gem5 is `trace.csv` in the run --outdir.
# This script copies it to:
#   <trace_dir>/<bench>_<ff_instructions>/trace.csv
#
# Defaults:
# - GEM5_ROOT: /home/ubuntu/peregrine-gem5 (override via env)
# - TRACE_DIR: <GEM5_ROOT>/configs/peregrine/traces (override via env)
# - CHECKPOINT_DIR: <GEM5_ROOT>/configs/peregrine/checkpoints (override via env;
#   must contain per-benchmark+instruction dirs from spec_checkpoint_benchmarks.sh)
#
# Run from anywhere; script cd's to GEM5_ROOT.

set -euo pipefail

if ! command -v parallel &>/dev/null; then
  echo "GNU parallel is required. Install with: apt install parallel" >&2
  exit 1
fi

GEM5_ROOT="${GEM5_ROOT:-/home/ubuntu/peregrine-gem5}"
GEM5_BIN="${GEM5_BIN:-$GEM5_ROOT/build/X86/gem5.opt}"
TRACE_DIR="${TRACE_DIR:-$GEM5_ROOT/configs/peregrine/traces}"
TRACE_OUT_BASE="${TRACE_OUT_BASE:-$GEM5_ROOT/configs/peregrine/trace_outputs}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$GEM5_ROOT/configs/peregrine/checkpoints}"

BENCHMARKS=("505.mcf_r" "520.omnetpp_r" "523.xalancbmk_r" "541.leela_r" "548.exchange2_r" "531.deepsjeng_r" "557.xz_r" "525.x264_r" "502.gcc_r") # "500.perlbench_r"

# Map each benchmark to multiple fast-forward instruction counts.
# Each benchmark maps to a space-separated list of instruction counts.
# Format: benchmark_name -> "instructions1 instructions2 instructions3 ..."
# Note: seq <start> <step> <stop> to define the sequence of instruction counts. <stop> is inclusive.
# This must match the checkpoints created by spec_checkpoint_benchmarks.sh
declare -A benchmark_checkpoints=(
  ["505.mcf_r"]="$(seq 300000000 10000000 1570000000)"
  ["520.omnetpp_r"]="$(seq 100000000 94500000 12195565497)"
  ["523.xalancbmk_r"]="$(seq 100000000 1760000 324007592)"
  ["541.leela_r"]="$(seq 100000000 185000000 23772938110)"
  ["548.exchange2_r"]="$(seq 7505000000 100000000 20205000000)"
  ["531.deepsjeng_r"]="$(seq 100000000 4200000 637116693)"
  ["557.xz_r"]="$(seq 100000000 5430000 791549354)"
  ["525.x264_r"]="$(seq 10000000000 100000000 22700000000)"
  ["502.gcc_r"]="$(seq 1000000 110000 15000000)"
)
BENCHMARK_CHECKPOINTS_DEF="$(declare -p benchmark_checkpoints)"
export GEM5_ROOT GEM5_BIN TRACE_DIR TRACE_OUT_BASE CHECKPOINT_DIR BENCHMARKS BENCHMARK_CHECKPOINTS_DEF

if [[ ! -d "$GEM5_ROOT" ]]; then
  echo "GEM5_ROOT not found: $GEM5_ROOT" >&2
  exit 1
fi

cd "$GEM5_ROOT"

if [[ ! -x "$GEM5_BIN" ]]; then
  echo "gem5 binary not found or not executable: $GEM5_BIN" >&2
  echo "Build it first, e.g.: scons build/X86/gem5.opt -j \$(nproc)" >&2
  exit 1
fi

mkdir -p "$TRACE_DIR"

run_bench() {
  local bench="$1"
  local ff_instructions="$2"

  local checkpoint_dir="$CHECKPOINT_DIR/${bench}_${ff_instructions}"
  local cpt_file="$checkpoint_dir/m5.cpt"
  local pmem_file="$checkpoint_dir/board.physmem.store0.pmem"

  if [[ ! -d "$checkpoint_dir" ]]; then
    echo "Checkpoint directory not found: $checkpoint_dir" >&2
    echo "Run configs/peregrine/spec_checkpoint_benchmarks.sh first." >&2
    return 1
  fi
  if [[ ! -f "$cpt_file" ]]; then
    echo "Checkpoint file not found: $cpt_file" >&2
    return 1
  fi
  if [[ ! -f "$pmem_file" ]]; then
    echo "Physical memory file not found: $pmem_file" >&2
    return 1
  fi

  local outdir="$TRACE_OUT_BASE/m5out_${bench}_${ff_instructions}"
  rm -rf "$outdir"
  mkdir -p "$outdir"
  local stdout_file="$outdir/gem5_stdout.log"

  echo "Tracing benchmark: $bench at ${ff_instructions} instructions (restore from $checkpoint_dir)"
  set +e
  (
    cd "$GEM5_ROOT" || exit 1
    "$GEM5_BIN" --redirect-stdout --stdout-file="$stdout_file" configs/peregrine/peregrine.py \
      --trace \
      --benchmark "$bench" \
      --outdir "$outdir" \
      --restore-checkpoint \
      --checkpoint-dir "$checkpoint_dir" \
      --max-insts 100000
  )
  local gem_status=$?
  set -e

  if [[ $gem_status -eq 0 ]]; then
    rm -f "$stdout_file"
  else
    echo "gem5.opt failed (status $gem_status) for benchmark=$bench at ${ff_instructions} instructions; stdout log: $stdout_file" >&2
    return "$gem_status"
  fi

  local src_trace="$outdir/trace.csv"
  if [[ ! -f "$src_trace" ]]; then
    echo "Expected trace not found: $src_trace" >&2
    return 1
  fi

  local dest_dir="$TRACE_DIR/${bench}_${ff_instructions}"
  mkdir -p "$dest_dir"
  cp -f "$src_trace" "$dest_dir/trace.csv"

  echo "  Wrote: $dest_dir/trace.csv"

  rm -rf "$outdir"
}

export -f run_bench

# Function to generate all benchmark+checkpoint combinations
generate_combinations() {
  eval "$BENCHMARK_CHECKPOINTS_DEF"

  for bench in "${BENCHMARKS[@]}"; do
    if [[ -n "${benchmark_checkpoints[$bench]:-}" ]]; then
      instructions_list="${benchmark_checkpoints[$bench]}"
      # Parse space-separated instruction counts
      for instructions in $instructions_list; do
        echo "$bench $instructions"
      done
    else
      echo "Warning: No checkpoints defined for benchmark $bench" >&2
    fi
  done
}

# Generate combinations and run in parallel
generate_combinations | parallel -j "$(nproc)" --colsep ' ' \
  --env GEM5_ROOT --env GEM5_BIN --env TRACE_DIR --env TRACE_OUT_BASE --env CHECKPOINT_DIR --env run_bench \
  run_bench {1} {2}

rm -rf "$TRACE_OUT_BASE"

echo "Done. Traces in: $TRACE_DIR/<benchmark>_<ff_instructions>/trace.csv"
