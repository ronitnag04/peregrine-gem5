#!/usr/bin/env bash
# Run 525.x264_r under gem5 (peregrine, --fast-only) and stop when frame_1.yuv
# appears in the SPEC run directory. Then print the last committed-instruction
# progress line from gem5_stdout.log under debug_outputs.
#
# Layout: configs/peregrine/debug_outputs/m5out_525.x264_r/{gem5_stdout.log,stdout.txt,...}
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

BENCH="525.x264_r"
RUN_DIR="$SPEC_DIR/benchspec/CPU/$BENCH/run/run_base_test_peregrine-m64.0000"
OUTDIR="$DEBUG_OUT_BASE/m5out_${BENCH}"
GEM5_LOG="$OUTDIR/gem5_stdout.log"

spec_debug_stop_check() {
  [[ -f "$RUN_DIR/frame_0.yuv" ]]
}

echo "Stop when created: $RUN_DIR/frame_0.yuv"
echo ""

spec_debug_stop_main "$BENCH" "$RUN_DIR" "$OUTDIR" "$GEM5_LOG" "$POLL_SEC"
