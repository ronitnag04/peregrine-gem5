# restore_and_run.py
"""
gem5 SE-mode script: restore from a checkpoint and simulate forward.

Usage:
    gem5 restore_and_run.py --binary ./collatz --checkpoint-dir m5out/checkpoint/

Optional:
    --inst-count N   stop after N additional committed instructions (default: run to completion)
    --outdir DIR     output directory (default: m5out)
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
    X86TimingSimpleCPU,
)

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument(
    "--binary",
    required=True,
    help="Path to the same ELF binary used when checkpointing",
)
parser.add_argument(
    "--args", default="", help="Binary arguments as one quoted string"
)
parser.add_argument(
    "--checkpoint-dir",
    required=True,
    help="Path to the checkpoint directory (e.g. m5out/cpt.1234567/)",
)
parser.add_argument(
    "--inst-count",
    type=int,
    default=None,
    help="Stop after this many additional committed instructions "
    "(default: run to completion)",
)
parser.add_argument("--outdir", default="m5out")
args = parser.parse_args()

# ── System  (must be identical to the one used at checkpoint time) ────────────
system = System()
system.clk_domain = SrcClockDomain(
    clock="3GHz", voltage_domain=VoltageDomain()
)
system.mem_mode = "timing"
system.mem_ranges = [AddrRange("512MB")]

# ── CPU ───────────────────────────────────────────────────────────────────────
system.cpu = X86TimingSimpleCPU()

if args.inst_count is not None:
    system.cpu.max_insts_any_thread = args.inst_count

# ── Memory bus + DRAM ─────────────────────────────────────────────────────────
system.membus = SystemXBar()

system.cpu.icache_port = system.membus.cpu_side_ports
system.cpu.dcache_port = system.membus.cpu_side_ports
system.system_port = system.membus.cpu_side_ports

system.mem_ctrl = MemCtrl()
system.mem_ctrl.dram = DDR3_1600_8x8()
system.mem_ctrl.dram.range = system.mem_ranges[0]
system.mem_ctrl.port = system.membus.mem_side_ports

# ── x86 interrupt controller ──────────────────────────────────────────────────
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

# ── Instantiate, then restore ─────────────────────────────────────────────────
root = Root(full_system=False, system=system)
m5.options.outdir = args.outdir
m5.core.setOutputDir(args.outdir)

# Pass the checkpoint directory to instantiate() — this is what triggers
# the restore.  gem5 replays the serialised state on top of the freshly
# constructed SimObject tree, so the tree must match the one that was
# checkpointed.
m5.instantiate(args.checkpoint_dir)

print(f"[restore] Restored from: {args.checkpoint_dir}")
if args.inst_count is not None:
    print(
        f"[restore] Will stop after {args.inst_count} additional instructions."
    )
else:
    print("[restore] Running to completion.")

# ── Simulate ──────────────────────────────────────────────────────────────────
exit_event = m5.simulate()
cause = exit_event.getCause()
print(f"[restore] Stopped: {cause!r}  tick={m5.curTick()}")

sys.exit(0)
