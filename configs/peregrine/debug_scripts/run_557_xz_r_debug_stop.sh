#!/usr/bin/env bash
# Run 557.xz_r under gem5 (peregrine, --fast-only) and stop when guest stdout
# (stdout.txt under --outdir) contains the spec_mem_load decompress summary line
# (after decompress, before first SHA-512 of the buffer).
# Then print the last committed-instruction progress line from gem5_stdout.log.
#
# Layout: configs/peregrine/debug_outputs/m5out_557.xz_r/{gem5_stdout.log,stdout.txt,...}
#
# Env: GEM5_ROOT, GEM5_BIN, SPEC_DIR, POLL_SEC (default 2)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=spec_debug_stop_common.sh
source "$SCRIPT_DIR/spec_debug_stop_common.sh"

GEM5_ROOT="${GEM5_ROOT:-/home/ubuntu/peregrine-gem5}"
GEM5_BIN="${GEM5_BIN:-$GEM5_ROOT/build/X86/gem5.opt}"
SPEC_DIR="${SPEC_DIR:-/home/ubuntu/SPEC2017-1-1-9}"
DEBUG_OUT_BASE="${DEBUG_OUT_BASE:-$GEM5_ROOT/configs/peregrine/debug_outputs}"
POLL_SEC="${POLL_SEC:-2}"

BENCH="557.xz_r"
RUN_DIR="$SPEC_DIR/benchspec/CPU/$BENCH/run/run_base_test_peregrine-m64.0000"
OUTDIR="$DEBUG_OUT_BASE/m5out_${BENCH}"
GEM5_LOG="$OUTDIR/gem5_stdout.log"
GUEST_STDOUT="$OUTDIR/stdout.txt"

STOP_LINE='Compressed size: 1287176; Uncompressed size: 9041920'

spec_debug_stop_check() {
  [[ -f "$GUEST_STDOUT" ]] && grep -Fq "$STOP_LINE" "$GUEST_STDOUT"
}

echo "Guest stdout (stop when line appears): $GUEST_STDOUT"
echo "Trigger substring: $STOP_LINE"
echo ""

spec_debug_stop_main "$BENCH" "$RUN_DIR" "$OUTDIR" "$GEM5_LOG" "$POLL_SEC"
