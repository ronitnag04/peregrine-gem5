#!/usr/bin/env bash

set -e

BENCHMARKS=("505.mcf_r" "520.omnetpp_r" "523.xalancbmk_r" "541.leela_r" "548.exchange2_r" "531.deepsjeng_r" "557.xz_r" "525.x264_r" "502.gcc_r") # "500.perlbench_r"

declare -A benchmark_checkpoints
benchmark_checkpoints["505.mcf_r"]="$(seq 300000000 20000000 1570000000)"
benchmark_checkpoints["520.omnetpp_r"]="$(seq 100000000 189000000 12195565497)"
benchmark_checkpoints["523.xalancbmk_r"]="$(seq 100000000 7040000 324007592)"
benchmark_checkpoints["541.leela_r"]="$(seq 100000000 370000000 23772938110)"
benchmark_checkpoints["548.exchange2_r"]="$(seq 7505000000 200000000 20205000000)"
benchmark_checkpoints["531.deepsjeng_r"]="$(seq 100000000 16800000 637116693)"
benchmark_checkpoints["557.xz_r"]="$(seq 100000000 21720000 791549354)"
benchmark_checkpoints["525.x264_r"]="$(seq 10000000000 200000000 22700000000)"
benchmark_checkpoints["502.gcc_r"]="$(seq 1000000 880000 15000000)"

TRACE_DIR="/home/ubuntu/peregrine-gem5/configs/peregrine/traces_04_07_2026"
DEST_DIR="/home/ubuntu/peregrine/ronamol/traces_04_07_2026"

mkdir -p "$DEST_DIR"

count=0
for benchmark in "${BENCHMARKS[@]}"; do
  for checkpoint in ${benchmark_checkpoints[$benchmark]}; do
    src_path="$TRACE_DIR/${benchmark}_${checkpoint}"
    dest_path="$DEST_DIR/${benchmark}_${checkpoint}"

    if [[ -d "$src_path" ]]; then
      echo "Copying $src_path to $dest_path"
      cp -r "$src_path" "$dest_path"
      count=$((count + 1))
    else
      echo "Warning: Source directory not found: $src_path"
    fi
  done
done

echo "Copied $count traces"
