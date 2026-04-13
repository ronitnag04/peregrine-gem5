"""
gem5 SE-mode script:
  1. Boots a binary with TimingSimpleCPU.
  2. Exits after `--inst-count` committed instructions (default 1000).
  3. Writes a checkpoint to <outdir>/checkpoint/.

Usage:
  gem5 make_cpt.py --binary ./hello
  gem5 make_cpt.py --binary ./spec_binary --args "-i input.txt" --inst-count 10000000
"""

import argparse
import sys

import m5
from m5.objects import (
    AddrRange,
    DDR3_1600_8x8,
    MemCtrl,
    Process,
    Root,
    SEWorkload,
    SrcClockDomain,
    System,
    SystemXBar,
    VoltageDomain,
    X86AtomicSimpleCPU,
)

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--binary", required=True)
parser.add_argument(
    "--args", default="", help="Binary arguments as one quoted string"
)
parser.add_argument(
    "--inst-count",
    type=int,
    default=1000,
    help="Committed instructions before checkpoint (default: 1000)",
)
parser.add_argument("--outdir", default="m5out")
args = parser.parse_args()

# ── System ────────────────────────────────────────────────────────────────────
system = System()
system.clk_domain = SrcClockDomain(
    clock="3GHz", voltage_domain=VoltageDomain()
)
system.mem_mode = "atomic"
system.mem_ranges = [AddrRange("512MB")]

# ── CPU ───────────────────────────────────────────────────────────────────────
# AtomicSimpleCPU is ideal for reaching a checkpoint quickly.
# Swap for X86O3CPU if you want to capture out-of-order state.
system.cpu = X86AtomicSimpleCPU()

# Stop and request a checkpoint after N committed instructions.
system.cpu.max_insts_any_thread = args.inst_count

# ── Memory bus + DRAM ─────────────────────────────────────────────────────────
system.membus = SystemXBar()

# Wire CPU ports directly to the bus (no caches for simplicity;
# add L1I/L1DCache objects here if you want a more realistic hierarchy).
system.cpu.icache_port = system.membus.cpu_side_ports
system.cpu.dcache_port = system.membus.cpu_side_ports
system.system_port = system.membus.cpu_side_ports

system.mem_ctrl = MemCtrl()
system.mem_ctrl.dram = DDR3_1600_8x8()
system.mem_ctrl.dram.range = system.mem_ranges[0]
system.mem_ctrl.port = system.membus.mem_side_ports

# ── x86 interrupt controller (required in SE mode for x86) ───────────────────
system.cpu.createInterruptController()
system.cpu.interrupts[0].pio = system.membus.mem_side_ports
system.cpu.interrupts[0].int_requestor = system.membus.cpu_side_ports
system.cpu.interrupts[0].int_responder = system.membus.mem_side_ports

# ── Workload / Process ────────────────────────────────────────────────────────
binary_args = args.args.split() if args.args else []
process = Process()
process.cmd = [args.binary] + binary_args
system.cpu.workload = process
system.cpu.createThreads()
system.workload = SEWorkload.init_compatible(args.binary)

# ── Instantiate & run ─────────────────────────────────────────────────────────
root = Root(full_system=False, system=system)
m5.options.outdir = args.outdir
m5.core.setOutputDir(args.outdir)
m5.instantiate()

print(f"[cpt_script] Running until {args.inst_count} committed instructions …")

exit_event = m5.simulate()
cause = exit_event.getCause()
print(f"[cpt_script] Stopped: {cause!r}  tick={m5.curTick()}")

# max_insts fires with cause "a thread reached the max instruction count"
# Write the checkpoint regardless of cause so you also get one on early exit.
print(f"[cpt_script] Writing checkpoint → {args.outdir}/checkpoint/")
m5.checkpoint(args.outdir + "/checkpoint")

print("[cpt_script] Done.")
sys.exit(0)
