#!/usr/bin/env bash

set -euo pipefail

CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-/home/ubuntu/peregrine-gem5/configs/peregrine/spec_checkpoints}"
S3_PREFIX="${S3_PREFIX:-s3://ronitnag04-peregrine/spec/spec-v3/spec_checkpoints}"
JOBS="${JOBS:-$(nproc)}"

if [[ ! -d "$CHECKPOINTS_DIR" ]]; then
  echo "Error: checkpoints directory not found: $CHECKPOINTS_DIR" >&2
  exit 1
fi

if ! command -v parallel >/dev/null 2>&1; then
  echo "Error: GNU parallel is required" >&2
  exit 1
fi

if ! command -v aws >/dev/null 2>&1; then
  echo "Error: aws CLI is required" >&2
  exit 1
fi

process_checkpoint() {
  local benchmark="$1"
  local cpt_name="$2"
  local checkpoints_dir="$3"
  local s3_prefix="$4"

  local bench_dir="$checkpoints_dir/$benchmark"
  local zip_path="$bench_dir/${cpt_name}.zip"
  local s3_dest="$s3_prefix/$benchmark/${cpt_name}.zip"

  echo "[$benchmark/$cpt_name] zipping..."
  (cd "$bench_dir" && zip -qr "${cpt_name}.zip" "$cpt_name")

  echo "[$benchmark/$cpt_name] uploading to $s3_dest"
  aws s3 cp --only-show-errors "$zip_path" "$s3_dest"

  echo "[$benchmark/$cpt_name] removing local zip"
  rm -f "$zip_path"

  echo "[$benchmark/$cpt_name] done"
}

export -f process_checkpoint

jobs_list=$(mktemp)
trap 'rm -f "$jobs_list"' EXIT

for bench_dir in "$CHECKPOINTS_DIR"/*/; do
  benchmark=$(basename "$bench_dir")
  for cpt_dir in "$bench_dir"cpt.*/; do
    [[ -d "$cpt_dir" ]] || continue
    cpt_name=$(basename "$cpt_dir")
    printf '%s\t%s\n' "$benchmark" "$cpt_name" >> "$jobs_list"
  done
done

total=$(wc -l < "$jobs_list")
echo "Found $total checkpoints. Running with $JOBS parallel jobs."

parallel --colsep '\t' --jobs "$JOBS" --halt soon,fail=1 \
  process_checkpoint {1} {2} "$CHECKPOINTS_DIR" "$S3_PREFIX" \
  :::: "$jobs_list"

echo "All $total checkpoints compressed and uploaded."
