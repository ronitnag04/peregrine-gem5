#!/usr/bin/env bash
# Full SPEC region simulation + trace + ronamol training row generation.
# - Input: sim_region_param_sweep.csv (benchmark,checkpoint,fast_forward + params)
# - Output #1: raw sim CSV with added CPI column.
# - Output #2: training CSV with CPI + sweep params + ronamol features.
# - Per-run trace is gzipped, uploaded to S3, and deleted locally.

set -euo pipefail

if ! command -v parallel >/dev/null 2>&1; then
  echo "GNU parallel is required. Install with: apt install parallel" >&2
  exit 1
fi
if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required." >&2
  exit 1
fi
if ! command -v aws >/dev/null 2>&1; then
  echo "aws CLI is required for trace upload." >&2
  exit 1
fi

GEM5_ROOT="${GEM5_ROOT:-/home/ubuntu/peregrine-gem5}"
PEREGRINE_ROOT="${PEREGRINE_ROOT:-/home/ubuntu/peregrine}"
GEM5_BIN="${GEM5_BIN:-$GEM5_ROOT/build/X86/gem5.opt}"
SWEEP_CSV="${SWEEP_CSV:-$GEM5_ROOT/configs/peregrine/sim_region_param_sweep.csv}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$GEM5_ROOT/configs/peregrine/checkpoints}"
TRACE_ROOT="${TRACE_ROOT:-$PEREGRINE_ROOT/ronamol/traces_04_25_2026}"
S3_PREFIX="${S3_PREFIX:-s3://ronitnag04-peregrine/spec/spec-v3/traces_04_25_2026}"

RESULTS_DIR="${RESULTS_DIR:-$GEM5_ROOT/configs/peregrine/sweep_outputs_v3}"
RUN_OUT_BASE="${RUN_OUT_BASE:-$RESULTS_DIR/m5out}"
ERR_LOG_DIR="${ERR_LOG_DIR:-$RESULTS_DIR/errors}"
RESULTS_CSV="${RESULTS_CSV:-$RESULTS_DIR/sweep_results.csv}"
TRAINING_CSV="${TRAINING_CSV:-$RESULTS_DIR/ronamol_spec_training_data_v3.csv}"
FAILED_CSV="${FAILED_CSV:-$ERR_LOG_DIR/failed_runs_v3.csv}"
CSV_LOCK_FILE="${CSV_LOCK_FILE:-$RESULTS_DIR/results.lock}"
FAILED_LOCK_FILE="${FAILED_LOCK_FILE:-$RESULTS_DIR/failed.lock}"
TRAINING_LOCK_FILE="${TRAINING_LOCK_FILE:-$RESULTS_DIR/training.lock}"

MAX_INSTS="${MAX_INSTS:-100000}"
JOBS="${JOBS:-$(nproc)}"
ROW_LIMIT="${ROW_LIMIT:-0}"     # 0 means all rows
ROW_OFFSET="${ROW_OFFSET:-0}"   # rows skipped from data section (after header)

export GEM5_ROOT PEREGRINE_ROOT GEM5_BIN SWEEP_CSV CHECKPOINT_DIR TRACE_ROOT S3_PREFIX
export RESULTS_DIR RUN_OUT_BASE ERR_LOG_DIR RESULTS_CSV TRAINING_CSV FAILED_CSV
export CSV_LOCK_FILE FAILED_LOCK_FILE TRAINING_LOCK_FILE MAX_INSTS

if [[ ! -d "$GEM5_ROOT" ]]; then
  echo "GEM5_ROOT not found: $GEM5_ROOT" >&2
  exit 1
fi
if [[ ! -d "$PEREGRINE_ROOT" ]]; then
  echo "PEREGRINE_ROOT not found: $PEREGRINE_ROOT" >&2
  exit 1
fi
if [[ ! -x "$GEM5_BIN" ]]; then
  echo "gem5 binary not found or not executable: $GEM5_BIN" >&2
  exit 1
fi
if [[ ! -f "$SWEEP_CSV" ]]; then
  echo "Sweep CSV not found: $SWEEP_CSV" >&2
  exit 1
fi

mkdir -p "$RESULTS_DIR" "$RUN_OUT_BASE" "$ERR_LOG_DIR" "$TRACE_ROOT"
touch "$CSV_LOCK_FILE" "$FAILED_LOCK_FILE" "$TRAINING_LOCK_FILE"

parse_size_to_kb() {
  local size="$1"
  case "$size" in
    *KiB) echo "${size%KiB}" ;;
    *MiB) echo "$(( ${size%MiB} * 1024 ))" ;;
    *)
      echo "Unsupported cache size format: $size" >&2
      return 1
      ;;
  esac
}

append_failed_row() {
  local row_id="$1"
  local stage="$2"
  local reason="$3"
  local csv_row="$4"
  (
    flock -x 9
    printf '%s,%s,"%s","%s"\n' "$row_id" "$stage" "$reason" "$csv_row" >> "$FAILED_CSV"
  ) 9>>"$FAILED_LOCK_FILE"
}
export -f parse_size_to_kb append_failed_row

append_training_row() {
  local raw_csv="$1"
  local cpi="$2"
  local trace_file="$3"
  local branch_predictor="$4"
  local l1i_kb="$5"
  local l1d_kb="$6"
  local l2_kb="$7"
  local training_csv="$8"

  local trace_dir rona_dir program_path cache_path bp_path
  trace_dir="$(dirname "$trace_file")"
  rona_dir="$trace_dir/ronamol"
  program_path="$rona_dir/program_features.csv"
  cache_path="$rona_dir/cache_latency_summary.csv"
  bp_path="$rona_dir/bp_rates_summary.csv"

  local cache_header cache_row cache_extra bp_header bp_row bp_extra
  cache_header="$(awk 'NR==1{print; exit}' "$cache_path")"
  cache_row="$(awk 'NR==2{print; exit}' "$cache_path")"
  cache_extra="$(awk 'NR==3{print; exit}' "$cache_path")"
  [[ -n "$cache_header" && -n "$cache_row" ]] || { echo "Missing cache summary rows in $cache_path" >&2; return 1; }
  [[ -z "$cache_extra" ]] || { echo "Expected a single cache summary row in $cache_path" >&2; return 1; }

  bp_header="$(awk 'NR==1{print; exit}' "$bp_path")"
  bp_row="$(awk 'NR==2{print; exit}' "$bp_path")"
  bp_extra="$(awk 'NR==3{print; exit}' "$bp_path")"
  [[ -n "$bp_header" && -n "$bp_row" ]] || { echo "Missing bp summary rows in $bp_path" >&2; return 1; }
  [[ -z "$bp_extra" ]] || { echo "Expected a single bp summary row in $bp_path" >&2; return 1; }

  local -a out_headers out_values raw_fields cache_headers cache_values bp_headers bp_values
  IFS=',' read -r -a raw_fields <<< "$raw_csv"
  ((${#raw_fields[@]} == 23)) || { echo "Unexpected raw row width: ${#raw_fields[@]}" >&2; return 1; }
  IFS=',' read -r -a cache_headers <<< "$cache_header"
  IFS=',' read -r -a cache_values <<< "$cache_row"
  IFS=',' read -r -a bp_headers <<< "$bp_header"
  IFS=',' read -r -a bp_values <<< "$bp_row"
  for i in "${!cache_headers[@]}"; do
    cache_headers[$i]="${cache_headers[$i]%$'\r'}"
    cache_values[$i]="${cache_values[$i]%$'\r'}"
  done
  for i in "${!bp_headers[@]}"; do
    bp_headers[$i]="${bp_headers[$i]%$'\r'}"
    bp_values[$i]="${bp_values[$i]%$'\r'}"
  done

  out_headers=(cpi benchmark checkpoint fast_forward branch_predictor commit_width decode_width fetch_width fp_mult_div_issue_width fp_reg_issue_width int_mult_div_issue_width int_reg_issue_width l1d_size l1i_size l2_size lq_entries max_icache_fills rdwr_port_issue_width read_port_issue_width rename_width rob_size simd_unit_issue_width sq_entries stride_prefetcher_degree)
  out_values=("$cpi" "${raw_fields[@]}")

  local prog_header prog_row
  prog_header="$(awk 'NR==1{print; exit}' "$program_path")"
  prog_row="$(awk 'NR==2{print; exit}' "$program_path")"
  [[ -n "$prog_header" && -n "$prog_row" ]] || { echo "Missing rows in program features CSV: $program_path" >&2; return 1; }
  local -a prog_headers prog_values
  IFS=',' read -r -a prog_headers <<< "$prog_header"
  IFS=',' read -r -a prog_values <<< "$prog_row"
  for i in "${!prog_headers[@]}"; do
    prog_headers[$i]="${prog_headers[$i]%$'\r'}"
    prog_values[$i]="${prog_values[$i]%$'\r'}"
  done
  for ((i=0; i<${#prog_headers[@]}; i++)); do
    out_headers+=("prog_${prog_headers[$i]}")
    out_values+=("${prog_values[$i]}")
  done

  local i bp_rate_idx=-1
  for ((i=0; i<${#cache_headers[@]}; i++)); do
    case "${cache_headers[$i]}" in
      l1i_kb|l1d_kb|l2_kb) ;;
      *)
        out_headers+=("cache_${cache_headers[$i]}")
        out_values+=("${cache_values[$i]}")
        ;;
    esac
  done
  for ((i=0; i<${#bp_headers[@]}; i++)); do
    if [[ "${bp_headers[$i]}" == "misprediction_rate" ]]; then
      bp_rate_idx=$i
      break
    fi
  done
  ((bp_rate_idx >= 0)) || { echo "Could not find misprediction_rate column in bp summary" >&2; return 1; }
  out_headers+=(bp_misprediction_rate)
  out_values+=("${bp_values[$bp_rate_idx]}")

  local out_header_line out_value_line
  out_header_line="$(IFS=','; echo "${out_headers[*]}")"
  out_value_line="$(IFS=','; echo "${out_values[*]}")"

  (
    flock -x 9
    if [[ ! -s "$training_csv" ]]; then
      printf '%s\n%s\n' "$out_header_line" "$out_value_line" >> "$training_csv"
    else
      local existing_header
      IFS= read -r existing_header < "$training_csv"
      if [[ "$existing_header" != "$out_header_line" ]]; then
        echo "Training CSV header mismatch; refusing to append inconsistent schema." >&2
        exit 1
      fi
      printf '%s\n' "$out_value_line" >> "$training_csv"
    fi
  ) 9>>"$TRAINING_LOCK_FILE"
}
export -f append_training_row

cleanup_outputs() {
  rm -f "$CSV_LOCK_FILE" "$FAILED_LOCK_FILE" "$TRAINING_LOCK_FILE"
}
trap cleanup_outputs EXIT

run_one() {
  local row_id="$1"
  local line="$2"
  IFS=',' read -r benchmark checkpoint fast_forward branch_predictor commit_width decode_width fetch_width fp_mult_div_issue_width fp_reg_issue_width int_mult_div_issue_width int_reg_issue_width l1d_size l1i_size l2_size lq_entries max_icache_fills rdwr_port_issue_width read_port_issue_width rename_width rob_size simd_unit_issue_width sq_entries stride_prefetcher_degree <<< "$line"

  local checkpoint_path="$CHECKPOINT_DIR/$benchmark/cpt.$checkpoint"
  local cpt_file="$checkpoint_path/m5.cpt"
  local pmem_file="$checkpoint_path/board.physmem.store0.pmem"
  if [[ ! -f "$cpt_file" || ! -f "$pmem_file" ]]; then
    append_failed_row "$row_id" "checkpoint" "missing checkpoint files at $checkpoint_path" "$line"
    return 0
  fi

  local outdir="$RUN_OUT_BASE/m5out_row_${row_id}"
  mkdir -p "$outdir"
  local log_file="$outdir/gem5_stdout.log"
  local stderr_file="$outdir/job.stderr"

  set +e
  (
    cd "$GEM5_ROOT" || exit 1
    "$GEM5_BIN" --redirect-stdout --stdout-file="$log_file" configs/peregrine/peregrine.py \
      --benchmark "$benchmark" \
      --restore-checkpoint \
      --checkpoint-dir "$checkpoint_path" \
      --fast-forward "$fast_forward" \
      --max-insts "$MAX_INSTS" \
      --trace \
      --branch-predictor "$branch_predictor" \
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
      --outdir "$outdir"
  ) 2>"$stderr_file"
  local gem_status=$?
  set -e
  if [[ "$gem_status" -ne 0 ]]; then
    local err_log="$ERR_LOG_DIR/row_${row_id}.log"
    {
      echo "stage=gem5"
      echo "row_id=$row_id"
      echo "benchmark=$benchmark checkpoint=$checkpoint fast_forward=$fast_forward"
      echo "exit_code=$gem_status"
      echo "saved_at=$(date '+%Y-%m-%d %H:%M:%S %Z')"
      echo
      echo "=== job.stderr ==="
      [[ -f "$stderr_file" ]] && cat "$stderr_file"
      echo
      echo "=== gem5 stdout ==="
      [[ -f "$log_file" ]] && cat "$log_file"
    } > "$err_log"
    append_failed_row "$row_id" "gem5" "gem5 failed with status $gem_status; log=$err_log" "$line"
    rm -rf "$outdir"
    return 0
  fi

  local stats_file="$outdir/stats.txt"
  local trace_file="$outdir/trace.csv"
  if [[ ! -f "$stats_file" || ! -f "$trace_file" ]]; then
    append_failed_row "$row_id" "gem5_output" "missing stats.txt or trace.csv in $outdir" "$line"
    rm -rf "$outdir"
    return 0
  fi

  local cpi
  cpi="$(awk '/board.processor.detailed.core.cpi/ {print $2}' "$stats_file" | tail -n 1)"
  if [[ -z "$cpi" ]]; then
    append_failed_row "$row_id" "cpi" "could not extract CPI from $stats_file" "$line"
    rm -rf "$outdir"
    return 0
  fi

  (
    flock -x 9
    printf '%s,%s\n' "$cpi" "$line" >> "$RESULTS_CSV"
  ) 9>>"$CSV_LOCK_FILE"

  local trace_subdir="$TRACE_ROOT/row_${row_id}"
  mkdir -p "$trace_subdir"
  local working_trace="$trace_subdir/trace.csv"
  mv "$trace_file" "$working_trace"

  local l1i_kb l1d_kb l2_kb
  l1i_kb="$(parse_size_to_kb "$l1i_size")" || { append_failed_row "$row_id" "size_parse" "invalid l1i_size=$l1i_size" "$line"; rm -rf "$outdir"; return 0; }
  l1d_kb="$(parse_size_to_kb "$l1d_size")" || { append_failed_row "$row_id" "size_parse" "invalid l1d_size=$l1d_size" "$line"; rm -rf "$outdir"; return 0; }
  l2_kb="$(parse_size_to_kb "$l2_size")" || { append_failed_row "$row_id" "size_parse" "invalid l2_size=$l2_size" "$line"; rm -rf "$outdir"; return 0; }

  if [[ "$branch_predictor" != "local" && "$branch_predictor" != "tage" ]]; then
    append_failed_row "$row_id" "bp_map" "unsupported branch predictor for ronamol: $branch_predictor" "$line"
    rm -rf "$outdir" "$trace_subdir"
    return 0
  fi

  set +e
  (
    cd "$PEREGRINE_ROOT" || exit 1
    python3 gen_cache_latency.py --trace "$working_trace" --l1i-size "$l1i_kb" --l1d-size "$l1d_kb" --l2-size "$l2_kb" >/dev/null
  )
  local cache_status=$?
  set -e
  if [[ "$cache_status" -ne 0 ]]; then
    append_failed_row "$row_id" "gen_cache_latency" "gen_cache_latency.py failed" "$line"
    rm -rf "$outdir" "$trace_subdir"
    return 0
  fi

  set +e
  (
    cd "$PEREGRINE_ROOT" || exit 1
    python3 gen_bp_rate.py --trace "$working_trace" --branch-predictor "$branch_predictor" >/dev/null
  )
  local bp_status=$?
  set -e
  if [[ "$bp_status" -ne 0 ]]; then
    append_failed_row "$row_id" "gen_bp_rate" "gen_bp_rate.py failed" "$line"
    rm -rf "$outdir" "$trace_subdir"
    return 0
  fi

  set +e
  (
    cd "$PEREGRINE_ROOT/ronamol" || exit 1
    python3 python/gen_features.py "$working_trace" --program-features-format csv >/dev/null
  )
  local feat_status=$?
  set -e
  if [[ "$feat_status" -ne 0 ]]; then
    append_failed_row "$row_id" "gen_features" "gen_features.py failed" "$line"
    rm -rf "$outdir" "$trace_subdir"
    return 0
  fi

  set +e
  append_training_row "$line" "$cpi" "$working_trace" "$branch_predictor" "$l1i_kb" "$l1d_kb" "$l2_kb" "$TRAINING_CSV"
  local tr_status=$?
  set -e
  if [[ "$tr_status" -ne 0 ]]; then
    append_failed_row "$row_id" "training_merge" "failed to append training row" "$line"
    rm -rf "$outdir" "$trace_subdir"
    return 0
  fi

  rm -rf "$trace_subdir/bp_rates" "$trace_subdir/cache_latencies" "$trace_subdir/ronamol"

  set +e
  gzip -f "$working_trace"
  local gzip_status=$?
  set -e
  if [[ "$gzip_status" -ne 0 ]]; then
    append_failed_row "$row_id" "gzip" "failed to gzip trace" "$line"
    rm -rf "$outdir" "$trace_subdir"
    return 0
  fi

  local gz_trace="$working_trace.gz"
  local s3_key="$S3_PREFIX/row_${row_id}.trace.csv.gz"
  set +e
  aws s3 cp "$gz_trace" "$s3_key" >/dev/null
  local s3_status=$?
  set -e
  if [[ "$s3_status" -ne 0 ]]; then
    append_failed_row "$row_id" "s3_upload" "aws s3 cp failed for $s3_key" "$line"
    rm -rf "$outdir"
    return 0
  fi

  rm -f "$gz_trace"
  rmdir "$trace_subdir" 2>/dev/null || true
  rm -rf "$outdir"
}
export -f run_one

if [[ ! -s "$RESULTS_CSV" ]]; then
  IFS= read -r _sweep_header < "$SWEEP_CSV"
  printf 'cpi,%s\n' "$_sweep_header" > "$RESULTS_CSV"
fi

if [[ ! -s "$FAILED_CSV" ]]; then
  echo "row_id,stage,reason,csv_row" > "$FAILED_CSV"
fi

awk -v offset="$ROW_OFFSET" -v limit="$ROW_LIMIT" '
NR == 1 { next }
{
  idx = NR - 1
  if (idx <= offset) {
    next
  }
  print idx "\t" $0
  emitted++
  if (limit > 0 && emitted >= limit) {
    exit
  }
}
' "$SWEEP_CSV" | parallel -j "$JOBS" --colsep '\t' run_one {1} {2}

if [[ -d "$RUN_OUT_BASE" && -z "$(ls -A "$RUN_OUT_BASE")" ]]; then
  rmdir "$RUN_OUT_BASE"
fi
_failed_count=0
if [[ -f "$FAILED_CSV" ]]; then
  _failed_count=$(wc -l < "$FAILED_CSV" | tr -d ' ')
  if [[ "$_failed_count" -le 1 ]]; then
    rm -f "$FAILED_CSV"
    _failed_count=0
  fi
fi

if [[ -d "$ERR_LOG_DIR" && -z "$(ls -A "$ERR_LOG_DIR")" ]]; then
  rmdir "$ERR_LOG_DIR"
fi

if [[ "$_failed_count" -gt 0 ]]; then
  echo "Completed with failures. Failed runs CSV: $FAILED_CSV" >&2
else
  echo "All runs passed." >&2
fi
