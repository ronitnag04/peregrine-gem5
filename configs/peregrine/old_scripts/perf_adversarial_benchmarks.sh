#!/usr/bin/env bash
# Run native Peregrine adversarial microbenchmarks under perf stat and
# print a summary of instructions and cycles for each.
#
# Requires: perf, and the prebuilt benchmark binaries in
# /home/ubuntu/peregrine/benchmarks/build/bin (see benchmarks/Makefile).
# Must use AWS ".metal" instances for perf hardware counters.
#
# Usage:
#   [BENCHDIR=/path/to/bin] ./perf_adversarial_benchmarks.sh
#
# None of these benchmarks take input arguments.

set -euo pipefail

BENCHDIR="${BENCHDIR:-/home/ubuntu/peregrine/benchmarks/build/bin}"

if [[ ! -d "$BENCHDIR" ]]; then
  echo "error: benchmark directory missing: $BENCHDIR" >&2
  exit 1
fi

if ! command -v perf &>/dev/null; then
  echo "error: perf not found (install linux-tools or linux-cloud-tools package)." >&2
  exit 1
fi

BENCHMARKS=(
  "adversarial_branches"
  "icache_blast"
  "many_pages_streaming"
  "pow2_stride_benign"
  "pow2_stride_thrash"
  "ptrchase_rand"
  "serial_mul_chain"
  "stlf_misalign"
)

# First count field on a perf stat line (strip commas). $2 is the event name (may include :u).
perf_first_field() {
  local pattern="$1"
  local file="$2"
  awk -v pat="$pattern" 'NF >= 2 && $2 ~ ("^(" pat ")") { gsub(/,/,"",$1); print $1; exit }' "$file"
}

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

declare -a SUM_BENCH SUM_INST SUM_CYC SUM_RC
any_fail=0

echo "BENCHDIR=${BENCHDIR}"

for bench in "${BENCHMARKS[@]}"; do
  out="${TMP}/${bench}.stdout"
  err="${TMP}/${bench}.stderr"
  bin_path="${BENCHDIR}/${bench}"

  echo ""
  echo "=== ${bench} ==="

  if [[ ! -f "$bin_path" ]]; then
    echo "error: benchmark binary missing: $bin_path" >&2
    SUM_BENCH+=("$bench")
    SUM_INST+=("")
    SUM_CYC+=("")
    SUM_RC+=(1)
    any_fail=1
    continue
  fi

  echo "+ perf stat -e instructions,cycles ${bin_path}"

  set +e
  perf stat -e instructions,cycles "$bin_path" >"$out" 2>"$err"
  rc=$?
  set -e

  [[ -s "$out" ]] && cat "$out"
  [[ -s "$err" ]] && cat "$err" >&2

  inst=$(perf_first_field 'instructions' "$err" || true)
  cyc=$(perf_first_field 'cycles|cpu_cycles|cpu-cycles' "$err" || true)
  [[ -n "$inst" ]] || inst=""
  [[ -n "$cyc" ]] || cyc=""

  SUM_BENCH+=("$bench")
  SUM_INST+=("$inst")
  SUM_CYC+=("$cyc")
  SUM_RC+=("$rc")
  [[ "$rc" -eq 0 && -n "$inst" && -n "$cyc" ]] || any_fail=1
done

commafy() {
  local n="$1"
  if [[ -z "$n" ]]; then
    printf '%s' "n/a"
    return
  fi
  printf '%s\n' "$n" | awk '{
    x = $0 + 0
    s = sprintf("%d", x)
    out = ""
    while (length(s) > 3) {
      out = "," substr(s, length(s) - 2, 3) out
      s = substr(s, 1, length(s) - 3)
    }
    print s out
  }'
}

echo ""
echo "========================================================================"
echo "SUMMARY (perf stat -e instructions,cycles)"
echo "========================================================================"
hdr=$(printf '%-22s %18s %18s %6s' "benchmark" "instructions" "cycles" "exit")
echo "$hdr"
echo "--------------------------------------------------------------------------------"

for i in "${!SUM_BENCH[@]}"; do
  b="${SUM_BENCH[$i]}"
  ins="${SUM_INST[$i]}"
  cy="${SUM_CYC[$i]}"
  r="${SUM_RC[$i]}"
  ins_p=$(commafy "$ins")
  cy_p=$(commafy "$cy")
  printf '%-22s %18s %18s %6s\n' "$b" "$ins_p" "$cy_p" "$r"
done

[[ "$any_fail" -eq 0 ]]
