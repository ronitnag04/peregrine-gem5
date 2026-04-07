import argparse
from pathlib import Path
from typing import (
    Callable,
    Optional,
    Type,
)

from utils import (
    peregrine_benchmarks,
    spec_benchmark_args,
    spec_benchmarks,
)

import m5
from m5.objects import (
    TAGE,
    X86O3CPU,
    AtomicSimpleCPU,
    BasePrefetcher,
    BaseXBar,
    BranchPredictor,
    FUPool,
    InstructionTracer,
    IQUnit,
    L2XBar,
    LocalBP,
    StridePrefetcher,
    TournamentBP,
)
from m5.objects.FuncUnit import *
from m5.objects.FuncUnitConfig import *

from gem5.components.boards.abstract_board import AbstractBoard
from gem5.components.boards.mem_mode import MemMode
from gem5.components.boards.simple_board import SimpleBoard
from gem5.components.cachehierarchies.abstract_cache_hierarchy import (
    AbstractCacheHierarchy,
)
from gem5.components.cachehierarchies.classic.caches.l1dcache import L1DCache
from gem5.components.cachehierarchies.classic.caches.l1icache import L1ICache
from gem5.components.cachehierarchies.classic.caches.l2cache import L2Cache
from gem5.components.cachehierarchies.classic.private_l1_shared_l2_cache_hierarchy import (
    PrivateL1SharedL2CacheHierarchy,
)
from gem5.components.memory.single_channel import SingleChannelDDR4_2400
from gem5.components.processors.base_cpu_core import BaseCPUCore
from gem5.components.processors.base_cpu_processor import BaseCPUProcessor
from gem5.components.processors.switchable_processor import SwitchableProcessor
from gem5.isas import ISA
from gem5.resources.resource import BinaryResource
from gem5.simulate.exit_event import ExitEvent
from gem5.simulate.simulator import Simulator
from gem5.utils.override import overrides


def parse_args():
    parser = argparse.ArgumentParser(
        description="Peregrine gem5 O3 configuration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Issue widths (functional units)
    parser.add_argument("--int-reg-issue-width", type=int, default=2)
    parser.add_argument("--int-mult-div-issue-width", type=int, default=2)
    parser.add_argument("--fp-reg-issue-width", type=int, default=2)
    parser.add_argument("--fp-mult-div-issue-width", type=int, default=2)
    parser.add_argument("--read-port-issue-width", type=int, default=2)
    parser.add_argument("--rdwr-port-issue-width", type=int, default=2)
    parser.add_argument("--simd-unit-issue-width", type=int, default=1)
    # O3 pipeline widths
    parser.add_argument("--fetch-width", type=int, default=8)
    parser.add_argument("--decode-width", type=int, default=8)
    parser.add_argument("--rename-width", type=int, default=8)
    parser.add_argument("--wb-width", type=int, default=8)
    parser.add_argument("--commit-width", type=int, default=8)
    # O3 queue sizes
    parser.add_argument("--rob-size", type=int, default=192)
    parser.add_argument("--lq-entries", type=int, default=32)
    parser.add_argument("--sq-entries", type=int, default=32)
    # Branch predictor
    parser.add_argument(
        "--branch-predictor",
        type=str,
        default="local",
        choices=["local", "tage"],
    )
    # Cache sizes (e.g. "32KiB", "256KiB")
    parser.add_argument("--l1d-size", type=str, default="32KiB")
    parser.add_argument("--l1i-size", type=str, default="32KiB")
    parser.add_argument("--l2-size", type=str, default="256KiB")
    parser.add_argument("--max-icache-fills", type=int, default=4)
    parser.add_argument("--stride-prefetcher-degree", type=int, default=4)
    # Benchmark
    parser.add_argument(
        "--specdir",
        type=str,
        default="/home/ubuntu/SPEC2017-1-1-9",
        help="Directory containing built SPEC2017 benchmarks using the peregrine.cfg",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        choices=peregrine_benchmarks + spec_benchmarks,
    )
    # Execution behavior
    parser.add_argument("--fast-only", action="store_true", default=False)
    parser.add_argument("--trace", action="store_true", default=False)
    parser.add_argument("--max-insts", type=int)
    parser.add_argument("--fast-forward", type=int)
    parser.add_argument(
        "--progress-interval",
        type=str,
        default=None,
        help=(
            "Enable periodic CPU progress messages (gem5 Param.Frequency, e.g. 1000). "
            "If omitted, progress_interval is not set (default CPU behavior, typically off)."
        ),
    )
    parser.add_argument(
        "--take-checkpoint", action="store_true", default=False
    )
    parser.add_argument(
        "--restore-checkpoint", action="store_true", default=False
    )
    # Output directory
    parser.add_argument("--outdir", type=str, default="m5out")
    parser.add_argument("--checkpoint-dir", type=str)

    args = parser.parse_args()

    # Check arguments for valid options
    if args.fast_only and args.fast_forward is not None:
        parser.error("--fast-only cannot be used with --fast-forward")

    if args.take_checkpoint and args.restore_checkpoint:
        parser.error(
            "--take-checkpoint and --restore-checkpoint are mutually exclusive"
        )

    if (
        args.take_checkpoint or args.restore_checkpoint
    ) and args.checkpoint_dir is None:
        parser.error(
            "--checkpoint-dir is required when taking or restoring a checkpoint"
        )

    if args.take_checkpoint and args.fast_forward is None:
        parser.error(
            "--take-checkpoint requires fast-forward (e.g. --fast-forward 100000000)"
        )

    if args.take_checkpoint and args.max_insts is not None:
        parser.error("--take-checkpoint ignores --max-insts")

    if args.restore_checkpoint and args.fast_forward:
        parser.error(
            "--restore-checkpoint cannot be used with --fast-forward (checkpoint already encodes that position)"
        )

    return args


_args = parse_args()

_issue_width = (
    _args.int_reg_issue_width
    + _args.int_mult_div_issue_width
    + _args.fp_reg_issue_width
    + _args.fp_mult_div_issue_width
    + _args.read_port_issue_width
    + _args.rdwr_port_issue_width
    + _args.simd_unit_issue_width
)


# Custom FU pool with modified counts
class MyFUPool(FUPool):
    FUList = [
        IntALU(count=_args.int_reg_issue_width),
        IntMultDiv(count=_args.int_mult_div_issue_width),
        FP_ALU(count=_args.fp_reg_issue_width),
        FP_MultDiv(count=_args.fp_mult_div_issue_width),
        ReadPort(count=_args.read_port_issue_width),
        RdWrPort(count=_args.rdwr_port_issue_width),
        SIMD_Unit(count=_args.simd_unit_issue_width),
    ]


class MyOutOfOrderCore(BaseCPUCore):
    def __init__(
        self,
        fetch_width,
        decode_width,
        rename_width,
        wb_width,
        commit_width,
        issue_width,
        rob_size,
        lq_entries,
        sq_entries,
        branch_predictor,
        core_id: int = 0,
    ):
        super().__init__(X86O3CPU(cpu_id=core_id), ISA.X86)
        # TODO: Convert all parameter settings to use Param notation
        self.core.numROBEntries = rob_size
        self.core.fetchWidth = fetch_width
        self.core.decodeWidth = decode_width
        self.core.renameWidth = rename_width
        self.core.wbWidth = wb_width
        self.core.commitWidth = commit_width

        self.core.issueWidth = issue_width
        self.core.instQueues = IQUnit(fuPool=MyFUPool())

        self.core.LQEntries = lq_entries
        self.core.SQEntries = sq_entries

        self.core.branchPred = branch_predictor


class MyOutOfOrderProcessor(BaseCPUProcessor):
    def __init__(self, core: MyOutOfOrderCore):
        super().__init__([core])


class AtomicCore(BaseCPUCore):
    def __init__(self, core_id: int = 0):
        super().__init__(AtomicSimpleCPU(cpu_id=core_id), ISA.X86)


class AtomicProcessor(BaseCPUProcessor):
    def __init__(self, core_id: int = 0):
        super().__init__([AtomicCore(core_id=core_id)])


class MySwitchableProcessor(SwitchableProcessor):
    def __init__(
        self,
        detailed_core: MyOutOfOrderCore,
        core_id: int = 0,
        start_detailed: bool = False,
    ):
        """
        :param start_detailed: If True, build with O3 active and Atomic switched out.
            Use for checkpoint restore when the checkpoint was taken after switch()
            (O3 running). Otherwise BaseCPU::unserialize expects _pid on the Atomic
            core but switched-out CPUs omit it from the checkpoint.
        """
        self._start_key = "fast_forward"
        self._switch_key = "detailed"
        self._start_detailed = start_detailed
        self._is_fast_forward = not start_detailed
        switchable_cores = {
            self._start_key: [AtomicCore(core_id=core_id)],
            self._switch_key: [detailed_core],
        }
        super().__init__(
            switchable_cores=switchable_cores,
            starting_cores=(
                self._switch_key if start_detailed else self._start_key
            ),
        )

    @overrides(SwitchableProcessor)
    def incorporate_processor(self, board: AbstractBoard) -> None:
        super().incorporate_processor(board=board)
        board.set_mem_mode(
            MemMode.TIMING if self._start_detailed else MemMode.ATOMIC
        )

    def switch(self):
        if self._is_fast_forward:
            self._board.set_mem_mode(MemMode.TIMING)
            self.switch_to_processor(self._switch_key)
            self._is_fast_forward = False

    def _post_instantiate(self) -> None:
        super()._post_instantiate()
        # After checkpoint restore, SimObject switched_out matches the saved CPU
        # but Python _current_cores still reflects the initial starting_cores.
        detailed_sim = self.detailed[0].get_simobject()
        if not detailed_sim.switched_out:
            self._current_cores = self.detailed
            self._is_fast_forward = False
        else:
            self._current_cores = self.fast_forward
            self._is_fast_forward = True


class MyCacheHierarchy(PrivateL1SharedL2CacheHierarchy):
    """
    Private L1 (I/D) + shared L2 with optional L1 D-cache prefetcher.
    """

    def __init__(
        self,
        l1d_size: str,
        l1i_size: str,
        l2_size: str,
        l1d_assoc: int = 8,
        l1i_assoc: int = 8,
        l1i_mshrs: int = 4,
        l2_assoc: int = 16,
        membus: Optional[BaseXBar] = None,
        PrefetcherCls: Optional[Type[BasePrefetcher]] = None,
    ):
        super().__init__(
            l1d_size, l1i_size, l2_size, l1d_assoc, l1i_assoc, l2_assoc, membus
        )
        self._l1d_prefetcher_cls = PrefetcherCls
        self._l1i_mshrs = l1i_mshrs

    @overrides(AbstractCacheHierarchy)
    def incorporate_cache(self, board: AbstractBoard) -> None:
        board.connect_system_port(self.membus.cpu_side_ports)
        for _, port in board.get_mem_ports():
            self.membus.mem_side_ports = port

        self.l1icaches = [
            L1ICache(
                size=self._l1i_size,
                assoc=self._l1i_assoc,
                writeback_clean=False,
                mshrs=self._l1i_mshrs,
            )
            for _ in range(board.get_processor().get_num_cores())
        ]
        self.l1dcaches = [
            L1DCache(
                size=self._l1d_size,
                assoc=self._l1d_assoc,
                PrefetcherCls=self._l1d_prefetcher_cls,
            )
            for _ in range(board.get_processor().get_num_cores())
        ]
        self.l2bus = L2XBar()
        self.l2cache = L2Cache(size=self._l2_size, assoc=self._l2_assoc)

        if board.has_coherent_io():
            self._setup_io_cache(board)

        for i, cpu in enumerate(board.get_processor().get_cores()):
            cpu.connect_icache(self.l1icaches[i].cpu_side)
            cpu.connect_dcache(self.l1dcaches[i].cpu_side)
            self.l1icaches[i].mem_side = self.l2bus.cpu_side_ports
            self.l1dcaches[i].mem_side = self.l2bus.cpu_side_ports
            self._connect_table_walker(i, cpu)
            if board.get_processor().get_isa() == ISA.X86:
                cpu.connect_interrupt(
                    self.membus.mem_side_ports, self.membus.cpu_side_ports
                )
            else:
                cpu.connect_interrupt()

        self.l2bus.mem_side_ports = self.l2cache.cpu_side
        self.membus.cpu_side_ports = self.l2cache.mem_side


class MyPrefetcher(StridePrefetcher):
    def __init__(self, degree: int = _args.stride_prefetcher_degree):
        super().__init__()
        self.degree = degree


if _args.branch_predictor == "local":
    branch_predictor = BranchPredictor(conditionalBranchPred=LocalBP())
elif _args.branch_predictor == "tage":
    branch_predictor = BranchPredictor(conditionalBranchPred=TAGE())
else:
    raise ValueError(f"Invalid branch predictor: {_args.branch_predictor}")

detailed_core = MyOutOfOrderCore(
    fetch_width=_args.fetch_width,
    decode_width=_args.decode_width,
    rename_width=_args.rename_width,
    wb_width=_issue_width,
    commit_width=_args.commit_width,
    issue_width=_issue_width,
    rob_size=_args.rob_size,
    lq_entries=_args.lq_entries,
    sq_entries=_args.sq_entries,
    branch_predictor=branch_predictor,
    core_id=0,
)

if _args.fast_only:
    # Fast only execution uses simple AtomicProcessor w/o O3 switching
    processor = AtomicProcessor(core_id=0)
    sim_cpu = processor.get_cores()[0].get_simobject()
elif _args.restore_checkpoint:
    # Same CPU topology as when the checkpoint was taken (after fast-forward switch).
    # Initial switched_out flags must match that state so unserialize matches the cpt.
    processor = MySwitchableProcessor(
        detailed_core=detailed_core, core_id=0, start_detailed=True
    )
    sim_cpu = detailed_core.get_simobject()
elif _args.take_checkpoint or _args.fast_forward:
    # Both checkpoint-taking and ordinary fast-forward use this processor
    processor = MySwitchableProcessor(detailed_core=detailed_core, core_id=0)
    ff_cpu = processor.fast_forward[0].get_simobject()
    ff_cpu.max_insts_any_thread = _args.fast_forward
    sim_cpu = detailed_core.get_simobject()
else:
    processor = MyOutOfOrderProcessor(core=detailed_core)
    sim_cpu = processor.get_cores()[0].get_simobject()

if _args.trace:
    tracer = InstructionTracer(
        manager=sim_cpu,
        output_file="trace.csv",
    )
    sim_cpu.instruction_tracer = tracer

if _args.max_insts:
    print(f"Setting max instructions of active CPU to {_args.max_insts}")
    sim_cpu.max_insts_any_thread = _args.max_insts

if _args.progress_interval is not None:
    sim_cpu.progress_interval = _args.progress_interval

main_memory = SingleChannelDDR4_2400(size="4GiB")
cache_hierarchy = MyCacheHierarchy(
    l1d_size=_args.l1d_size,
    l1i_size=_args.l1i_size,
    l2_size=_args.l2_size,
    l1d_assoc=8,
    l1i_assoc=8,
    l2_assoc=16,
    l1i_mshrs=_args.max_icache_fills,
    PrefetcherCls=MyPrefetcher,
)

board = SimpleBoard(
    processor=processor,
    memory=main_memory,
    cache_hierarchy=cache_hierarchy,
    clk_freq="3GHz",
)

run_cwd: Optional[str] = None

if _args.benchmark in peregrine_benchmarks:
    binary = BinaryResource(
        local_path=f"tests/peregrine-bmarks/{_args.benchmark}-gem5"
    )
    arguments = []
elif _args.benchmark in spec_benchmarks:
    rundir = f"{_args.specdir}/benchspec/CPU/{_args.benchmark}/run/run_base_test_peregrine-m64.0000"
    run_cwd = rundir
    binary = spec_benchmark_args[_args.benchmark]["binary"]
    arguments = spec_benchmark_args[_args.benchmark]["arguments"]
    binary = BinaryResource(local_path=f"{rundir}/{binary}")
else:
    raise ValueError(f"Invalid benchmark: {_args.benchmark}")

# Only set checkpoint if --restore-checkpoint
_restore_ckpt_path = (
    Path(_args.checkpoint_dir).expanduser().resolve()
    if _args.restore_checkpoint
    else None
)
board.set_se_binary_workload(
    binary=binary,
    arguments=arguments,
    stdout_file=Path("stdout.txt"),
    stderr_file=Path("stderr.txt"),
    checkpoint=_restore_ckpt_path,
)

if run_cwd is not None:
    sim_processor = board.get_processor()
    cores = (
        sim_processor._all_cores()
        if isinstance(sim_processor, SwitchableProcessor)
        else sim_processor.get_cores()
    )
    for core in cores:
        cpu_simobj = core.get_simobject()
        for process in cpu_simobj.workload:
            process.cwd = run_cwd

if _args.take_checkpoint:
    # Phase 1: fast-forward then snapshot
    def _checkpoint_and_exit():
        ckpt_path = Path(_args.checkpoint_dir).expanduser().resolve()
        ckpt_path.mkdir(parents=True, exist_ok=True)
        print(f"Fast-forward done. Taking checkpoint → {ckpt_path}")
        processor.switch()  # switch to O3 so the checkpoint captures O3 state
        m5.stats.reset()
        simulator.save_checkpoint(ckpt_path)
        print("Checkpoint written. Exiting.")
        yield True  # exit immediately after checkpoint

    print(f"Taking checkpoint after {_args.fast_forward} instructions.")
    simulator = Simulator(
        board=board,
        outdir=_args.outdir,
        on_exit_event={ExitEvent.MAX_INSTS: _checkpoint_and_exit()},
    )
elif _args.restore_checkpoint:
    # Phase 2: restore and run O3 directly
    print(f"Restoring checkpoint from {_restore_ckpt_path}")
    simulator = Simulator(board=board, outdir=_args.outdir)
elif _args.fast_forward:

    def _switch_after_fast_forward():
        print(
            f"Fast-forward complete at {_args.fast_forward} instructions; "
            "switching to detailed O3 CPU."
        )
        if _args.max_insts:
            print(f"Simulating O3 CPU with max insts: {_args.max_insts}")
        processor.switch()
        m5.stats.reset()
        # Continue simulation after the switch.
        yield False
        # For any later MAX_INSTS events (e.g., --max-insts on O3), exit.
        yield True

    print(
        f"Fast-forwarding to {_args.fast_forward} instructions using AtomicSimpleCPU"
    )
    simulator = Simulator(
        board=board,
        outdir=f"{_args.outdir}",
        on_exit_event={ExitEvent.MAX_INSTS: _switch_after_fast_forward()},
    )
else:
    simulator = Simulator(board=board, outdir=f"{_args.outdir}")
simulator.run()
