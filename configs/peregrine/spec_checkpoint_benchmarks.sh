#!/usr/bin/env bash
# Create gem5 checkpoints for Peregrine SPEC CPU 2017 rate benchmarks.
#
# For each benchmark, runs fast-forward simulation to a fixed cycle count, then
# writes a checkpoint. This is the companion to spec_trace_benchmarks.sh: traces
# restore these checkpoints at the same fast-forward point.
#
# Invocation (per benchmark):
#   build/X86/gem5.opt configs/peregrine/peregrine.py \
#     --benchmark <bench> --fast-forward <cycles> --take-checkpoint \
#     --checkpoint-dir <CHECKPOINT_DIR>/<bench> --outdir <CHECKPOINT_OUT_BASE>/m5out_<bench>
#
# Each successful run leaves:
#   <CHECKPOINT_DIR>/<bench>/m5.cpt
#   <CHECKPOINT_DIR>/<bench>/board.physmem.store0.pmem
#
# Defaults (override via environment):
# - GEM5_ROOT: /home/ubuntu/peregrine-gem5
# - GEM5_BIN:  $GEM5_ROOT/build/X86/gem5.opt
# - CHECKPOINT_DIR: $GEM5_ROOT/configs/peregrine/checkpoints
# - CHECKPOINT_OUT_BASE: $GEM5_ROOT/configs/peregrine/checkpoints_outputs (temporary m5out; removed after each run and at the end)
#
# Requires GNU parallel. Benchmarks run in parallel (-j $(nproc)).
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

# Map each benchmark to a fast-forward cycle count.
declare -A fast_forward_cycles=(
  ["505.mcf_r"]=370000000           # After start of the primal_net_simplex function
  ["520.omnetpp_r"]=1000000000      # Still setting up omnetpp
  ["523.xalancbmk_r"]=100000000     # Good, in the middle of printing out html
  ["541.leela_r"]=1000000000        # Probably doing work by this point
  ["548.exchange2_r"]=7505000000    # After output of first move
  ["531.deepsjeng_r"]=350000000     # After making first moves
  ["557.xz_r"]=500000000            # Before termination, probably near the input hashing
  ["525.x264_r"]=10000000000        # Before frame 0 finished, ideally while its working
  ["502.gcc_r"]=10000000            # Good, no output .s file written, but full program only takes 15_090_619 insts
)
FAST_FORWARD_CYCLES_DEF="$(declare -p fast_forward_cycles)"
export FAST_FORWARD_CYCLES_DEF

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

  eval "$FAST_FORWARD_CYCLES_DEF"

  local ff_cycles
  ff_cycles="${fast_forward_cycles[$bench]}"

  local outdir="$CHECKPOINT_OUT_BASE/m5out_${bench}"
  rm -rf "$outdir"
  mkdir -p "$outdir"
  local stdout_file="$outdir/gem5_stdout.log"

  local checkpoint_dir="$CHECKPOINT_DIR/${bench}"

  echo "Checkpointing benchmark: $bench"
  set +e
  (
    cd "$GEM5_ROOT" || exit 1
    "$GEM5_BIN" --redirect-stdout --stdout-file="$stdout_file" configs/peregrine/peregrine.py \
      --benchmark "$bench" \
      --outdir "$outdir" \
      --fast-forward "$ff_cycles" \
      --take-checkpoint \
      --checkpoint-dir "$checkpoint_dir"
  )
  local gem_status=$?
  set -e

  if [[ $gem_status -eq 0 ]]; then
    rm -f "$stdout_file"
  else
    echo "gem5.opt failed (status $gem_status) for benchmark=$bench; stdout log: $stdout_file" >&2
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

  echo "${bench} Checkpoint created successfully: $checkpoint_dir"

  rm -rf "$outdir"
}

export -f run_bench

printf "%s\n" "${BENCHMARKS[@]}" | parallel -j "$(nproc)" \
  --env GEM5_ROOT --env GEM5_BIN --env CHECKPOINT_DIR --env CHECKPOINT_OUT_BASE --env run_bench --env FAST_FORWARD_CYCLES_DEF \
  run_bench


rm -rf "$CHECKPOINT_OUT_BASE"

echo "Done. Checkpoints in $CHECKPOINT_DIR"
