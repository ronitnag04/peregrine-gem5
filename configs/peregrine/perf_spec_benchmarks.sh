#!/usr/bin/env bash
# Run native SPEC CPU2017 rate (Peregrine) benchmarks under perf stat and
# print a summary of instructions and cycles for each.
#
# Requires: perf, and SPEC run directories from runcpu with the Peregrine
# config (see configs/peregrine/README.md).
#
# Usage:
#   SPECDIR=/path/to/cpu2017-root [SIZE=test|train] ./perf_spec_benchmarks.sh
#
# SPECDIR is the same root passed to gem5 as --specdir.
# SIZE selects run_base_<SIZE>_peregrine-m64.0000 and the argv list below.

set -euo pipefail

SPECDIR="${SPECDIR:-/home/ubuntu/SPEC2017-1-1-9}"
SIZE="${SIZE:-test}"
SIZE_LC="${SIZE,,}"
case "$SIZE_LC" in
test | train) ;;
*)
  echo "error: SIZE must be test or train (got: $SIZE)" >&2
  exit 1
  ;;
esac

if [[ -z "$SPECDIR" ]]; then
  echo "error: set SPECDIR to your SPEC CPU2017 installation root (gem5 --specdir)." >&2
  exit 1
fi

if ! command -v perf &>/dev/null; then
  echo "error: perf not found (install linux-tools or linux-cloud-tools package)." >&2
  exit 1
fi

# Same order as spec_benchmarks in configs/peregrine/utils.py
BENCHMARKS=(
  "505.mcf_r"
  "520.omnetpp_r"
  "523.xalancbmk_r"
  "541.leela_r"
  "548.exchange2_r"
  "531.deepsjeng_r"
  "557.xz_r"
  "525.x264_r"
  "502.gcc_r"
)

# --- argv per SIZE (suffix _test / _train; bench id uses _ instead of .) ---
# Derived from each benchmark's run_base_<SIZE>_peregrine-m64.0000/speccmds.cmd
# (single representative invocation where SPEC lists several).

ARGS_505_mcf_r_test=(inp.in)
ARGS_505_mcf_r_train=(inp.in)

ARGS_520_omnetpp_r_test=(-c General -r 0)
ARGS_520_omnetpp_r_train=(-c General -r 0)

ARGS_523_xalancbmk_r_test=(-v test.xml xalanc.xsl)
ARGS_523_xalancbmk_r_train=(-v allbooks.xml xalanc.xsl)

ARGS_541_leela_r_test=(test.sgf)
ARGS_541_leela_r_train=(train.sgf)

ARGS_548_exchange2_r_test=(0)
ARGS_548_exchange2_r_train=(1)

ARGS_531_deepsjeng_r_test=(test.txt)
ARGS_531_deepsjeng_r_train=(train.txt)

ARGS_557_xz_r_test=(
  cpu2006docs.tar.xz
  4
  055ce243071129412e9dd0b3b69a21654033a9b723d874b2015c774fac1553d9713be561ca86f74e4f16f22e664fc17a79f30caa5ad2c04fbc447549c2810fae
  1548636
  1555348
  0
)
ARGS_557_xz_r_train=(
  input.combined.xz
  40
  a841f68f38572a49d86226b7ff5baeb31bd19dc637a922a972b2e6d1257a890f6a544ecab967c313e370478c74f760eb229d4eef8a8d2836d233d3e9dd1430bf
  6356684
  -1
  8
)

ARGS_525_x264_r_test=(--dumpyuv 50 --frames 156 -o BuckBunny_New.264 BuckBunny.yuv 1280x720)
ARGS_525_x264_r_train=(--dumpyuv 50 --frames 142 -o BuckBunny_New.264 BuckBunny.yuv 1280x720)

ARGS_502_gcc_r_test=(t1.c -O3 -finline-limit=50000 -o t1.opts-O3_-finline-limit_50000.s)
ARGS_502_gcc_r_train=(train01.c -O3 -finline-limit=50000 -o train01.opts-O3_-finline-limit_50000.s)

# First count field on a perf stat line (strip commas). $2 is the event name (may include :u).
perf_first_field() {
  local pattern="$1"
  local file="$2"
  awk -v pat="$pattern" 'NF >= 2 && $2 ~ ("^(" pat ")") { gsub(/,/,"",$1); print $1; exit }' "$file"
}

# Sets globals: binary=, args=()  from BENCHMARK name and SIZE_LC (test|train).
spec_benchmark_cmd() {
  local bench="$1"
  local size="$2"
  local suf="${bench//./_}"
  local argvar="ARGS_${suf}_${size}"

  binary=
  args=()

  if ! declare -p "$argvar" &>/dev/null; then
    echo "error: no argv array ${argvar}[] for benchmark ${bench} SIZE=${size}" >&2
    return 1
  fi
  local -n _argv="$argvar"
  args=("${_argv[@]}")

  case "$bench" in
  505.mcf_r) binary="mcf_r_base.peregrine-m64" ;;
  520.omnetpp_r) binary="omnetpp_r_base.peregrine-m64" ;;
  523.xalancbmk_r) binary="cpuxalan_r_base.peregrine-m64" ;;
  541.leela_r) binary="leela_r_base.peregrine-m64" ;;
  548.exchange2_r) binary="exchange2_r_base.peregrine-m64" ;;
  531.deepsjeng_r) binary="deepsjeng_r_base.peregrine-m64" ;;
  557.xz_r) binary="xz_r_base.peregrine-m64" ;;
  525.x264_r) binary="x264_r_base.peregrine-m64" ;;
  502.gcc_r) binary="cpugcc_r_base.peregrine-m64" ;;
  *)
    echo "error: no binary mapping for benchmark: $bench" >&2
    return 1
    ;;
  esac
}

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

declare -a SUM_BENCH SUM_INST SUM_CYC SUM_RC
any_fail=0

echo "SPECDIR=${SPECDIR}  SIZE=${SIZE_LC}  (run_base_${SIZE_LC}_peregrine-m64.0000)"

for bench in "${BENCHMARKS[@]}"; do
  bench_dir="${SPECDIR}/benchspec/CPU/${bench}/run/run_base_${SIZE_LC}_peregrine-m64.0000"
  out="${TMP}/${bench//./_}.stdout"
  err="${TMP}/${bench//./_}.stderr"

  echo ""
  echo "=== ${bench} ==="

  if [[ ! -d "$bench_dir" ]]; then
    echo "error: run directory missing: $bench_dir" >&2
    SUM_BENCH+=("$bench")
    SUM_INST+=("")
    SUM_CYC+=("")
    SUM_RC+=(1)
    any_fail=1
    continue
  fi

  spec_benchmark_cmd "$bench" "$SIZE_LC" || {
    any_fail=1
    SUM_BENCH+=("$bench")
    SUM_INST+=("")
    SUM_CYC+=("")
    SUM_RC+=(1)
    continue
  }

  bin_path="${bench_dir}/${binary}"
  if [[ ! -f "$bin_path" ]]; then
    echo "error: benchmark binary missing: $bin_path" >&2
    SUM_BENCH+=("$bench")
    SUM_INST+=("")
    SUM_CYC+=("")
    SUM_RC+=(1)
    any_fail=1
    continue
  fi

  echo "cwd: ${bench_dir}"
  echo "+ perf stat -e instructions,cycles ./${binary} ${args[*]}"

  set +e
  (
    cd "$bench_dir"
    perf stat -e instructions,cycles "./${binary}" "${args[@]}"
  ) >"$out" 2>"$err"
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
echo "SUMMARY (perf stat -e instructions,cycles)  SIZE=${SIZE_LC}"
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
