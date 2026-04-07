#!/usr/bin/env bash
# Create gem5 checkpoints for Peregrine SPEC CPU 2017 rate benchmarks.
#
# For each benchmark and checkpoint point, runs fast-forward simulation to the specified instruction count, then
# writes a checkpoint. This is the companion to spec_trace_benchmarks.sh: traces
# restore these checkpoints at the same fast-forward point.
#
# Invocation (per benchmark+checkpoint combination):
#   build/X86/gem5.opt configs/peregrine/peregrine.py \
#     --benchmark <bench> --fast-forward <instructions> --take-checkpoint \
#     --checkpoint-dir <CHECKPOINT_DIR>/<bench>_<instructions> --outdir <CHECKPOINT_OUT_BASE>/m5out_<bench>_<instructions>
#
# Each successful run leaves:
#   <CHECKPOINT_DIR>/<bench>_<instructions>/m5.cpt
#   <CHECKPOINT_DIR>/<bench>_<instructions>/board.physmem.store0.pmem
#
# Defaults (override via environment):
# - GEM5_ROOT: /home/ubuntu/peregrine-gem5
# - GEM5_BIN:  $GEM5_ROOT/build/X86/gem5.opt
# - CHECKPOINT_DIR: $GEM5_ROOT/configs/peregrine/checkpoints
# - CHECKPOINT_OUT_BASE: $GEM5_ROOT/configs/peregrine/checkpoints_outputs (temporary m5out; removed after each run and at the end)
#
# Requires GNU parallel. Benchmark+checkpoint combinations run in parallel (-j $(nproc)).
# Run from anywhere; the script cd's to GEM5_ROOT.
#

set -euo pipefail

if ! command -v parallel &>/dev/null; then
  echo "GNU parallel is required. Install with: apt install parallel" >&2
  exit 1
fi

GEM5_ROOT="${GEM5_ROOT:-/home/ubuntu/peregrine-gem5}"
GEM5_BIN="${GEM5_BIN:-$GEM5_ROOT/build/X86/gem5.opt}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$GEM5_ROOT/configs/peregrine/checkpoints}"
CHECKPOINT_OUT_BASE="${CHECKPOINT_OUT_BASE:-$GEM5_ROOT/configs/peregrine/checkpoints_outputs}"
export GEM5_ROOT GEM5_BIN CHECKPOINT_DIR CHECKPOINT_OUT_BASE

BENCHMARKS=("505.mcf_r" "520.omnetpp_r" "523.xalancbmk_r" "541.leela_r" "548.exchange2_r" "531.deepsjeng_r" "557.xz_r" "525.x264_r" "502.gcc_r") # "500.perlbench_r"

# Full benchmark instruction count
# "520.omnetpp_r" : 12_195_565_497
# "523.xalancbmk_r" : 324_007_592
# "541.leela_r" : 23_772_938_110
# "531.deepsjeng_r" : 637_116_693   # Generates output, but is incorrect?
# "557.xz_r" : 791_549_354          # Terminates on its own
# "502.gcc_r" : 15_090_619

# Work start benchmark instruction count
# "505.mcf_r" : 368_545_706         # starting primal_net_simplex
# "548.exchange2_r" : 7_504_323_138 # Outputs first moves
# "525.x264_r" : 10_196_809_375     # Finished frame 0
# "531.deepsjeng_r" : 340_078_068   # Starting to make moves
# "557.xz_r" : 497_090_979          # After the input data is loaded
# "541.leela_r" : 20_419_485_412    # Printed out a lot of info

# Map each benchmark to multiple fast-forward instruction counts.
# Each benchmark maps to a space-separated list of instruction counts.
# Format: benchmark_name -> "instructions1 instructions2 instructions3 ..."
# Note: seq <start> <step> <stop> to define the sequence of instruction counts. <stop> is inclusive.
declare -A benchmark_checkpoints=(
  ["505.mcf_r"]="$(seq 300000000 10000000 1580000000)"
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
export BENCHMARK_CHECKPOINTS_DEF

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

mkdir -p "$CHECKPOINT_DIR"

run_bench() {
  local bench="$1"
  local ff_instructions="$2"

  local outdir="$CHECKPOINT_OUT_BASE/m5out_${bench}_${ff_instructions}"
  rm -rf "$outdir"
  mkdir -p "$outdir"
  local stdout_file="$outdir/gem5_stdout.log"

  local checkpoint_dir="$CHECKPOINT_DIR/${bench}_${ff_instructions}"

  echo "Checkpointing benchmark: $bench at ${ff_instructions} instructions"
  set +e
  (
    cd "$GEM5_ROOT" || exit 1
    "$GEM5_BIN" --redirect-stdout --stdout-file="$stdout_file" configs/peregrine/peregrine.py \
      --benchmark "$bench" \
      --outdir "$outdir" \
      --fast-forward "$ff_instructions" \
      --take-checkpoint \
      --checkpoint-dir "$checkpoint_dir"
  )
  local gem_status=$?
  set -e

  if [[ $gem_status -eq 0 ]]; then
    rm -f "$stdout_file"
  else
    echo "gem5.opt failed (status $gem_status) for benchmark=$bench at ${ff_instructions} instructions; stdout log: $stdout_file" >&2
    return "$gem_status"
  fi

  # Verify checkpoint was created successfully
  if [[ ! -d "$checkpoint_dir" ]]; then
    echo "Checkpoint directory not created: $checkpoint_dir" >&2
    return 1
  fi

  local cpt_file="$checkpoint_dir/m5.cpt"
  local pmem_file="$checkpoint_dir/board.physmem.store0.pmem"

  if [[ ! -f "$cpt_file" ]]; then
    echo "Checkpoint file not found: $cpt_file" >&2
    return 1
  fi

  if [[ ! -f "$pmem_file" ]]; then
    echo "Physical memory file not found: $pmem_file" >&2
    return 1
  fi

  echo "${bench}_${ff_instructions} Checkpoint created successfully: $checkpoint_dir"

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
  --env GEM5_ROOT --env GEM5_BIN --env CHECKPOINT_DIR --env CHECKPOINT_OUT_BASE --env run_bench \
  run_bench {1} {2}


rm -rf "$CHECKPOINT_OUT_BASE"

echo "Done. Checkpoints in $CHECKPOINT_DIR"
