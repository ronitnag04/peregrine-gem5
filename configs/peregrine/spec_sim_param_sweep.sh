#!/usr/bin/env bash
# Run gem5 peregrine param sweep: one job per (CSV row, benchmark).
# Restores the checkpoint from spec_checkpoint_benchmarks.sh (same fast-forward point as
# spec_trace_benchmarks.sh), then runs detailed O3 with the sweep parameters.
# Uses GNU parallel. Results append to a single CSV with locking.
#
# Memory: each gem5 restore can use a lot of RSS. Running -j $(nproc) without limits
# often triggers the OOM killer (exit 137). This script uses GNU parallel's
# --memsuspend so new jobs wait and running jobs are suspended when RAM is tight,
# then resumed when space is available (see SWEEP_MEMSUSPEND).
#
# Requires: GNU parallel (apt install parallel); checkpoints under CHECKPOINT_DIR
# Run from repo root or any dir; script cd's to peregrine-gem5.

set -e
if ! command -v parallel &>/dev/null; then
  echo "GNU parallel is required. Install with: apt install parallel"
  exit 1
fi
GEM5_ROOT="${GEM5_ROOT:-/home/ubuntu/peregrine-gem5}"
GEM5_BIN="${GEM5_BIN:-$GEM5_ROOT/build/X86/gem5.opt}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$GEM5_ROOT/configs/peregrine/checkpoints}"
cd "$GEM5_ROOT"

if [[ ! -x "$GEM5_BIN" ]]; then
  echo "gem5 binary not found or not executable: $GEM5_BIN" >&2
  echo "Build it first, e.g.: scons build/X86/gem5.opt -j \$(nproc)" >&2
  exit 1
fi
SWEEP_CSV="${SWEEP_CSV:-configs/peregrine/param_sweep.csv}"
OUT_BASE="$GEM5_ROOT/configs/peregrine/sweep_outputs"
ERR_LOG_DIR="$GEM5_ROOT/configs/peregrine/sweep_errors"
mkdir -p "$ERR_LOG_DIR"
RESULTS_CSV="${RESULTS_CSV:-$GEM5_ROOT/configs/peregrine/sweep_results.csv}"
FAILED_CSV="$ERR_LOG_DIR/failed_param_sweep.csv"
LOCK_FILE="$GEM5_ROOT/configs/peregrine/sweep_results.lock"
mkdir -p "$OUT_BASE"

BENCHMARKS=("505.mcf_r" "520.omnetpp_r" "523.xalancbmk_r" "541.leela_r" "548.exchange2_r" "531.deepsjeng_r" "557.xz_r" "525.x264_r" "502.gcc_r") # "500.perlbench_r"
for bench in "${BENCHMARKS[@]}"; do
  _cpt_dir="$CHECKPOINT_DIR/${bench}"
  _cpt_file="$_cpt_dir/m5.cpt"
  _pmem_file="$_cpt_dir/board.physmem.store0.pmem"
  if [[ ! -d "$_cpt_dir" ]]; then
    echo "Checkpoint directory not found: $_cpt_dir" >&2
    echo "Run configs/peregrine/spec_checkpoint_benchmarks.sh first." >&2
    exit 1
  fi
  if [[ ! -f "$_cpt_file" ]]; then
    echo "Checkpoint file not found: $_cpt_file" >&2
    exit 1
  fi
  if [[ ! -f "$_pmem_file" ]]; then
    echo "Physical memory file not found: $_pmem_file" >&2
    exit 1
  fi
done
unset _cpt_dir _cpt_file _pmem_file

export GEM5_ROOT GEM5_BIN CHECKPOINT_DIR SWEEP_CSV OUT_BASE ERR_LOG_DIR RESULTS_CSV FAILED_CSV LOCK_FILE BENCHMARKS

log_failed_sweep_row() {
  (
    flock -x 9
    echo "$1" >> "$FAILED_CSV"
  ) 9>>"$LOCK_FILE"
}
export -f log_failed_sweep_row

run_one() {
  local line="$1"
  IFS=',' read -ra F <<< "$line"
  local row="${F[0]}"
  local bench="${F[1]}"
  local bp="${F[2]}"
  local commit_width="${F[3]}"
  local decode_width="${F[4]}"
  local fetch_width="${F[5]}"
  local fp_mult_div_issue_width="${F[6]}"
  local fp_reg_issue_width="${F[7]}"
  local int_mult_div_issue_width="${F[8]}"
  local int_reg_issue_width="${F[9]}"
  local l1d_size="${F[10]}"
  local l1i_size="${F[11]}"
  local l2_size="${F[12]}"
  local lq_entries="${F[13]}"
  local max_icache_fills="${F[14]}"
  local rdwr_port_issue_width="${F[15]}"
  local read_port_issue_width="${F[16]}"
  local rename_width="${F[17]}"
  local rob_size="${F[18]}"
  local simd_unit_issue_width="${F[19]}"
  local sq_entries="${F[20]}"
  local stride_prefetcher_degree="${F[21]}"

  local checkpoint_dir="$CHECKPOINT_DIR/${bench}"

  local outdir="$OUT_BASE/m5out_${PARALLEL_SEQ:-$$}"
  mkdir -p "$outdir"
  local log_file="$outdir/$bench-$bp-$commit_width-$decode_width-$fetch_width-$fp_mult_div_issue_width-$fp_reg_issue_width-$int_mult_div_issue_width-$int_reg_issue_width-$l1d_size-$l1i_size-$l2_size-$lq_entries-$max_icache_fills-$rdwr_port_issue_width-$read_port_issue_width-$rename_width-$rob_size-$simd_unit_issue_width-$sq_entries-$stride_prefetcher_degree.log"
  # Subshell stderr (gem5 stderr, bash "Killed", etc.) — gem5 stdout goes to log_file via --stdout-file
  local job_stderr="$outdir/job.stderr"

  # Run gem5, capturing exit status so we can decide whether to keep or delete the log.
  set +e
  (
    cd "$GEM5_ROOT" || exit 1
    "$GEM5_BIN" --redirect-stdout --stdout-file="$log_file" configs/peregrine/peregrine.py \
      --max-insts 1000000 \
      --restore-checkpoint \
      --checkpoint-dir "$checkpoint_dir" \
      --branch-predictor "$bp" \
      --commit-width "$commit_width" \
      --decode-width "$decode_width" \
      --fetch-width "$fetch_width" \
      --fp-mult-div-issue-width "$fp_mult_div_issue_width" \
      --fp-reg-issue-width "$fp_reg_issue_width" \
      --int-mult-div-issue-width "$int_mult_div_issue_width" \
      --int-reg-issue-width "$int_reg_issue_width" \
      --l1d-size "$l1d_size" \
      --l1i-size "$l1i_size" \
      --l2-size "$l2_size" \
      --lq-entries "$lq_entries" \
      --max-icache-fills "$max_icache_fills" \
      --rdwr-port-issue-width "$rdwr_port_issue_width" \
      --read-port-issue-width "$read_port_issue_width" \
      --rename-width "$rename_width" \
      --rob-size "$rob_size" \
      --simd-unit-issue-width "$simd_unit_issue_width" \
      --sq-entries "$sq_entries" \
      --stride-prefetcher-degree "$stride_prefetcher_degree" \
      --benchmark "$bench" \
      --outdir "$outdir"
  ) 2>"$job_stderr"
  local gem_status=$?
  set -e

  # Logfile handling:
  # - If gem5 succeeded, delete the per-run log.
  # - If gem5 failed, keep the log so the failure can be inspected.
  if [[ $gem_status -eq 0 ]]; then
    rm -f "$log_file" "$job_stderr"
  else
    local err_log_file="$ERR_LOG_DIR/$(basename "$log_file")"
    mv "$log_file" "$err_log_file"
    local err_tmp="${err_log_file}.tmp.$$"
    {
      printf '%s\n' \
        "=== gem5 sweep failure ===" \
        "exit_code=${gem_status}" \
        "benchmark=${bench}" \
        "param_sweep_row_index=${row}" \
        "checkpoint_dir=${checkpoint_dir}" \
        "saved_at=$(date '+%Y-%m-%d %H:%M:%S %Z')"
      if [[ $gem_status -eq 137 ]]; then
        printf '%s\n' "note: exit 137 often indicates OOM killer (SIGKILL)."
      fi
      if [[ -s "$job_stderr" ]]; then
        printf '\n%s\n' "=== job stderr (gem5 + shell) ==="
        cat "$job_stderr"
      fi
      printf '\n%s\n' "=== gem5 stdout ==="
      cat "$err_log_file"
    } >"$err_tmp" && mv "$err_tmp" "$err_log_file"
    rm -f "$job_stderr"
    # Same columns as param_sweep.csv (branch_predictor … stride_prefetcher_degree)
    local param_sweep_row="${bp},${commit_width},${decode_width},${fetch_width},${fp_mult_div_issue_width},${fp_reg_issue_width},${int_mult_div_issue_width},${int_reg_issue_width},${l1d_size},${l1i_size},${l2_size},${lq_entries},${max_icache_fills},${rdwr_port_issue_width},${read_port_issue_width},${rename_width},${rob_size},${simd_unit_issue_width},${sq_entries},${stride_prefetcher_degree}"
    log_failed_sweep_row "$param_sweep_row"
    echo "gem5.opt failed (status $gem_status) for benchmark=$bench, row=$row; log saved at: $err_log_file" >&2
  fi

  local cpi=""
  # Only attempt to read and record CPI if the simulation completed successfully.
  if [[ $gem_status -eq 0 && -f "$outdir/stats.txt" ]]; then
    # Get the CPI from the stats.txt file, using seconds set of stats dump corresponding to m5_work region of interest
    cpi=$(awk '/board.processor.detailed.core.cpi/ {print $2}' "$outdir/stats.txt")
    local csv_row="${cpi},${bench},${bp},${commit_width},${decode_width},${fetch_width},${fp_mult_div_issue_width},${fp_reg_issue_width},${int_mult_div_issue_width},${int_reg_issue_width},${l1d_size},${l1i_size},${l2_size},${lq_entries},${max_icache_fills},${rdwr_port_issue_width},${read_port_issue_width},${rename_width},${rob_size},${simd_unit_issue_width},${sq_entries},${stride_prefetcher_degree}"
    (
      flock -x 9
      echo "$csv_row" >> "$RESULTS_CSV"
    ) 9>>"$LOCK_FILE"
  fi
  if [[ -d "$outdir" && "$outdir" == *"/m5out_"* ]]; then
    rm -rf "$outdir"
  fi
}
export -f run_one

# Write CSV header (cpi first, then benchmark, param columns; matches param_sweep.csv)
echo "cpi,benchmark,branch_predictor,commit_width,decode_width,fetch_width,fp_mult_div_issue_width,fp_reg_issue_width,int_mult_div_issue_width,int_reg_issue_width,l1d_size,l1i_size,l2_size,lq_entries,max_icache_fills,rdwr_port_issue_width,read_port_issue_width,rename_width,rob_size,simd_unit_issue_width,sq_entries,stride_prefetcher_degree" > "$RESULTS_CSV"
# Failed gem5 runs only: header matches param_sweep.csv (parameter columns only)
head -n 1 "$SWEEP_CSV" > "$FAILED_CSV"
touch "$LOCK_FILE"

echo "Sweep started at $(date '+%Y-%m-%d %H:%M:%S %Z')"
SWEEP_START_EPOCH=$(date +%s)

# Build job lines: row,benchmark,branch_predictor,commit_width,...,stride_prefetcher_degree (22 fields)
# Explicitly read and discard exactly one header line, then stream all parameter lines.
# Number rows from 1.
{
  # Read and discard header
  IFS= read -r _header

  job_count=0
  while IFS= read -r csv_line; do
    ((job_count++)) || true
    for bench in "${BENCHMARKS[@]}"; do
      echo "${job_count},${bench},${csv_line}"
    done
  done
} < "$SWEEP_CSV" | parallel -j 179 --env run_one --env log_failed_sweep_row run_one

SWEEP_END_EPOCH=$(date +%s)
SWEEP_ELAPSED=$((SWEEP_END_EPOCH - SWEEP_START_EPOCH))
SWEEP_ELAPSED_H=$((SWEEP_ELAPSED / 3600))
SWEEP_ELAPSED_M=$(((SWEEP_ELAPSED % 3600) / 60))
SWEEP_ELAPSED_S=$((SWEEP_ELAPSED % 60))
echo "Sweep finished at $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "Sweep wall time: ${SWEEP_ELAPSED}s (${SWEEP_ELAPSED_H}h ${SWEEP_ELAPSED_M}m ${SWEEP_ELAPSED_S}s)"

rm -rf "$OUT_BASE"

_had_gem5_failures=0
if [[ -f "$FAILED_CSV" ]]; then
  if [[ $(wc -l < "$FAILED_CSV" | tr -d ' ') -gt 1 ]]; then
    _had_gem5_failures=1
  else
    rm -f "$FAILED_CSV"
  fi
fi

if [[ -d "$ERR_LOG_DIR" && -z "$(ls -A "$ERR_LOG_DIR")" ]]; then
  rmdir "$ERR_LOG_DIR"
fi

rm "$LOCK_FILE"

echo "Results in $RESULTS_CSV"
if [[ "$_had_gem5_failures" -eq 1 ]]; then
  echo "Failed param rows (gem5 errors) recorded in $FAILED_CSV"
fi
unset _had_gem5_failures
