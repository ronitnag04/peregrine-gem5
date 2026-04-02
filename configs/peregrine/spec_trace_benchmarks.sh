#!/usr/bin/env bash
# Generate instruction traces for all Peregrine microbenchmarks.
#
# For each benchmark, this restores the checkpoint produced by
# spec_checkpoint_benchmarks.sh (same fast-forward point), then runs:
#   build/X86/gem5.opt configs/peregrine/peregrine.py --trace --restore-checkpoint ...
#
# The resulting trace file produced by gem5 is `trace.csv` in the run --outdir.
# This script copies it to:
#   <trace_dir>/<bench>/trace.csv
#
# Defaults:
# - GEM5_ROOT: /home/ubuntu/peregrine-gem5 (override via env)
# - TRACE_DIR: <GEM5_ROOT>/configs/peregrine/traces (override via env)
# - CHECKPOINT_DIR: <GEM5_ROOT>/configs/peregrine/checkpoints (override via env;
#   must contain per-benchmark dirs from spec_checkpoint_benchmarks.sh)
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
export GEM5_ROOT GEM5_BIN TRACE_DIR TRACE_OUT_BASE CHECKPOINT_DIR

BENCHMARKS=("505.mcf_r" "520.omnetpp_r" "523.xalancbmk_r" "541.leela_r" "548.exchange2_r" "531.deepsjeng_r" "557.xz_r" "525.x264_r" "502.gcc_r") # "500.perlbench_r"

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

  local checkpoint_dir="$CHECKPOINT_DIR/${bench}"
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

  local outdir="$TRACE_OUT_BASE/m5out_${bench}"
  rm -rf "$outdir"
  mkdir -p "$outdir"
  local stdout_file="$outdir/gem5_stdout.log"

  echo "Tracing benchmark: $bench (restore from $checkpoint_dir)"
  set +e
  (
    cd "$GEM5_ROOT" || exit 1
    "$GEM5_BIN" --redirect-stdout --stdout-file="$stdout_file" configs/peregrine/peregrine.py \
      --trace \
      --benchmark "$bench" \
      --outdir "$outdir" \
      --restore-checkpoint \
      --checkpoint-dir "$checkpoint_dir" \
      --max-insts 1000000
  )
  local gem_status=$?
  set -e

  if [[ $gem_status -eq 0 ]]; then
    rm -f "$stdout_file"
  else
    echo "gem5.opt failed (status $gem_status) for benchmark=$bench; stdout log: $stdout_file" >&2
    return "$gem_status"
  fi

  local src_trace="$outdir/trace.csv"
  if [[ ! -f "$src_trace" ]]; then
    echo "Expected trace not found: $src_trace" >&2
    return 1
  fi

  local dest_dir="$TRACE_DIR/$bench"
  mkdir -p "$dest_dir"
  cp -f "$src_trace" "$dest_dir/trace.csv"

  echo "  Wrote: $dest_dir/trace.csv"

  rm -rf "$outdir"
}

export -f run_bench

printf "%s\n" "${BENCHMARKS[@]}" | parallel -j "$(nproc)" \
  --env GEM5_ROOT --env GEM5_BIN --env TRACE_DIR --env TRACE_OUT_BASE --env CHECKPOINT_DIR --env run_bench \
  run_bench

rm -rf "$TRACE_OUT_BASE"

echo "Done. Traces in: $TRACE_DIR/<benchmark>/trace.csv"
