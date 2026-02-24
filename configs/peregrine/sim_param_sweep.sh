#!/usr/bin/env bash
# Run gem5 peregrine param sweep: one job per (CSV row, benchmark).
# Uses GNU parallel with 31 workers (leaves 1 CPU free). Results append to a single CSV with locking.
#
# Requires: GNU parallel (apt install parallel)
# Run from repo root or any dir; script cd's to peregrine-gem5.

set -e
if ! command -v parallel &>/dev/null; then
  echo "GNU parallel is required. Install with: apt install parallel"
  exit 1
fi
GEM5_ROOT="/home/ubuntu/peregrine-gem5"
cd "$GEM5_ROOT"
SWEEP_CSV="configs/peregrine/param_sweep.csv"
OUT_BASE="$GEM5_ROOT/configs/peregrine/sweep_outputs"
RESULTS_CSV="$GEM5_ROOT/configs/peregrine/sweep_results.csv"
LOCK_FILE="$GEM5_ROOT/configs/peregrine/sweep_results.lock"
mkdir -p "$OUT_BASE"

BENCHMARKS=(branch_storm collatz dhrystone linpack sieve sparse towers whetstone)
export GEM5_ROOT SWEEP_CSV OUT_BASE RESULTS_CSV LOCK_FILE BENCHMARKS

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
  local wb_width="${F[22]}"

  local outdir="$OUT_BASE/m5out_${PARALLEL_SEQ:-$$}"
  mkdir -p "$outdir"

  (
    cd "$GEM5_ROOT"
    ./build/X86/gem5.opt configs/peregrine/peregrine.py \
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
      --wb-width "$wb_width" \
      --benchmark "$bench" \
      --outdir "$outdir"
  )

  local cpi=""
  if [[ -f "$outdir/stats.txt" ]]; then
    cpi=$(awk '/board.processor.cores.core.cpi/ {count++; if (count==2) print $2}' "$outdir/stats.txt")
  fi
  if [[ -n "$cpi" ]]; then
    local csv_row="${cpi},${bench},${bp},${commit_width},${decode_width},${fetch_width},${fp_mult_div_issue_width},${fp_reg_issue_width},${int_mult_div_issue_width},${int_reg_issue_width},${l1d_size},${l1i_size},${l2_size},${lq_entries},${max_icache_fills},${rdwr_port_issue_width},${read_port_issue_width},${rename_width},${rob_size},${simd_unit_issue_width},${sq_entries},${stride_prefetcher_degree},${wb_width}"
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

# Write CSV header (cpi first, then benchmark, param columns)
echo "cpi,benchmark,branch_predictor,commit_width,decode_width,fetch_width,fp_mult_div_issue_width,fp_reg_issue_width,int_mult_div_issue_width,int_reg_issue_width,l1d_size,l1i_size,l2_size,lq_entries,max_icache_fills,rdwr_port_issue_width,read_port_issue_width,rename_width,rob_size,simd_unit_issue_width,sq_entries,stride_prefetcher_degree,wb_width" > "$RESULTS_CSV"
touch "$LOCK_FILE"

# Build job lines: row,benchmark,branch_predictor,commit_width,...,wb_width (23 fields)
# Skip CSV header; number rows from 1. Run with 31 parallel jobs (leave 1 CPU for SSH/monitor).
job_count=0
while IFS= read -r csv_line; do
  ((job_count++)) || true
  for bench in "${BENCHMARKS[@]}"; do
    echo "${job_count},${bench},${csv_line}"
  done
done < <(tail -n +2 "$SWEEP_CSV") | parallel -j 31 --env run_one run_one

rm -rf "$OUT_BASE"
rm "$LOCK_FILE"

echo "Sweep finished. Results in $RESULTS_CSV"
