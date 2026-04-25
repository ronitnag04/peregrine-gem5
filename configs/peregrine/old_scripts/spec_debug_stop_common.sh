#!/usr/bin/env bash
# Shared helpers for SPEC debug runs that stop gem5 when a benchmark milestone
# is reached. Sourced by run_*_debug_stop.sh in configs/peregrine/debug_scripts/.

spec_debug_stop_terminate_gem5() {
  local pid="$1"
  if kill -0 "$pid" 2>/dev/null; then
    kill -TERM "$pid" 2>/dev/null || true
    local w=0
    while kill -0 "$pid" 2>/dev/null && [[ $w -lt 60 ]]; do
      sleep 1
      w=$((w + 1))
    done
    kill -KILL "$pid" 2>/dev/null || true
  fi
  wait "$pid" 2>/dev/null || true
}

# Print the last gem5 CPU progress line (Atomic/O3 progress_interval output).
spec_debug_stop_print_last_progress() {
  local gem5_log="$1"
  echo ""
  echo "Last CPU progress line from gem5_stdout.log (total committed instructions):"
  if [[ -f "$gem5_log" ]]; then
    local lines
    lines="$(grep -F 'progress event, total committed' "$gem5_log" 2>/dev/null | tail -n 1 || true)"
    if [[ -n "$lines" ]]; then
      printf '%s\n' "$lines"
    else
      echo "(no lines matched 'progress event, total committed'; see full log: $gem5_log)"
    fi
  else
    echo "(missing: $gem5_log)"
  fi
}

# Args: benchmark name, SPEC run directory, outdir, gem5 log path, poll seconds, check command string for messages
# The caller must define spec_debug_stop_check() { return 0 when stop }; it may use RUN_DIR, OUTDIR, GEM5_LOG.
spec_debug_stop_main() {
  local bench="$1"
  local run_dir="$2"
  local outdir="$3"
  local gem5_log="$4"
  local poll_sec="${5:-2}"

  if [[ ! -d "$GEM5_ROOT" ]]; then
    echo "GEM5_ROOT not found: $GEM5_ROOT" >&2
    return 1
  fi
  if [[ ! -x "$GEM5_BIN" ]]; then
    echo "gem5 binary not found or not executable: $GEM5_BIN" >&2
    return 1
  fi
  if [[ ! -d "$run_dir" ]]; then
    echo "SPEC run directory not found: $run_dir" >&2
    return 1
  fi

  rm -rf "$outdir"
  mkdir -p "$outdir"

  echo "Benchmark: $bench"
  echo "gem5 --outdir: $outdir"
  echo "gem5 stdout log: $gem5_log"
  echo "SPEC rundir: $run_dir"
  echo ""

  (
    cd "$GEM5_ROOT" || exit 1
    exec "$GEM5_BIN" --redirect-stdout --stdout-file="$gem5_log" \
      configs/peregrine/peregrine.py \
      --benchmark "$bench" \
      --outdir "$outdir" \
      --fast-only \
      --progress-interval 1000
  ) &
  local gem5_pid=$!

  local triggered=0
  local gem5_done=0

  while true; do
    if ! kill -0 "$gem5_pid" 2>/dev/null; then
      gem5_done=1
      break
    fi
    if spec_debug_stop_check; then
      triggered=1
      break
    fi
    sleep "$poll_sec"
  done

  local wait_status=0
  if [[ "$triggered" -eq 1 ]]; then
    echo "Stop condition met; terminating gem5 (pid $gem5_pid)..."
    spec_debug_stop_terminate_gem5 "$gem5_pid"
    wait_status=0
  elif [[ "$gem5_done" -eq 1 ]]; then
    wait "$gem5_pid" || wait_status=$?
    echo "gem5 exited on its own (status $wait_status)."
  fi

  spec_debug_stop_print_last_progress "$gem5_log"
  return "$wait_status"
}
