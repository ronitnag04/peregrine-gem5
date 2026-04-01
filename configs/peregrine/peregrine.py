import argparse
from pathlib import Path
from typing import (
    Callable,
    Optional,
    Type,
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
        default="branch_storm",
        choices=[
            "branch_storm",
            "collatz",
            "dhrystone",
            "linpack",
            "sieve",
            "sparse",
            "towers",
            "whetstone",
            "505.mcf_r",
            "520.omnetpp_r",
            "523.xalancbmk_r",
            "541.leela_r",
            "548.exchange2_r",
            "531.deepsjeng_r",
            "557.xz_r",
            # "500.perlbench_r",    # TODO: not compatible, may require FS mode to handle clock_nanosleep syscall
            "525.x264_r",
            "502.gcc_r",
        ],
    )
    # Execution behavior
    parser.add_argument(
        "--fast-only",
        action="store_true",
        default=False,
        help="Use only AtomicSimpleCPU for execution, skip O3 core",
    )
    parser.add_argument("--trace", action="store_true", default=False)
    parser.add_argument("--max-insts", type=int)
    parser.add_argument("--fast-forward", type=int)
    # Output directory
    parser.add_argument("--outdir", type=str, default="m5out")
    args = parser.parse_args()

    if args.fast_only and args.fast_forward is not None:
        parser.error("--fast-only cannot be used with --fast-forward")

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


class AtomicCore(BaseCPUCore):
    def __init__(self, core_id: int = 0):
        super().__init__(AtomicSimpleCPU(cpu_id=core_id), ISA.X86)


class MyOutOfOrderProcessor(BaseCPUProcessor):
    def __init__(self, core: MyOutOfOrderCore):
        super().__init__([core])


class AtomicProcessor(BaseCPUProcessor):
    def __init__(self, core_id: int = 0):
        super().__init__([AtomicCore(core_id=core_id)])


class FastForwardToO3Processor(SwitchableProcessor):
    def __init__(self, detailed_core: MyOutOfOrderCore, core_id: int = 0):
        self._start_key = "fast_forward"
        self._switch_key = "detailed"
        self._is_fast_forward = True
        switchable_cores = {
            self._start_key: [AtomicCore(core_id=core_id)],
            self._switch_key: [detailed_core],
        }
        super().__init__(
            switchable_cores=switchable_cores,
            starting_cores=self._start_key,
        )

    @overrides(SwitchableProcessor)
    def incorporate_processor(self, board: AbstractBoard) -> None:
        super().incorporate_processor(board=board)
        board.set_mem_mode(MemMode.ATOMIC)

    def switch(self):
        if self._is_fast_forward:
            self._board.set_mem_mode(MemMode.TIMING)
            self.switch_to_processor(self._switch_key)
            self._is_fast_forward = False


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
    processor = AtomicProcessor(core_id=0)
    sim_cpu = processor.get_cores()[0].get_simobject()
elif _args.fast_forward:
    processor = FastForwardToO3Processor(
        detailed_core=detailed_core, core_id=0
    )
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

peregrine_benchmarks = [
    "branch_storm",
    "collatz",
    "dhrystone",
    "linpack",
    "sieve",
    "sparse",
    "towers",
    "whetstone",
]


spec_benchmarks = {
    "505.mcf_r": {
        "binary": "mcf_r_base.peregrine-m64",
        "arguments": ["inp.in"],
    },
    "520.omnetpp_r": {
        "binary": "omnetpp_r_base.peregrine-m64",
        "arguments": ["-f", "omnetpp.ini", "-c", "General", "-r", "0"],
    },
    "523.xalancbmk_r": {
        "binary": "cpuxalan_r_base.peregrine-m64",
        "arguments": ["-v", "test.xml", "xalanc.xsl"],
    },
    "541.leela_r": {
        "binary": "leela_r_base.peregrine-m64",
        "arguments": ["test.sgf"],
    },
    "548.exchange2_r": {
        "binary": "exchange2_r_base.peregrine-m64",
        "arguments": ["0"],
    },
    "531.deepsjeng_r": {
        "binary": "deepsjeng_r_base.peregrine-m64",
        "arguments": ["test.txt"],
    },
    "557.xz_r": {
        "binary": "xz_r_base.peregrine-m64",
        "arguments": [
            "cpu2006docs.tar.xz",
            "4",
            "055ce243071129412e9dd0b3b69a21654033a9b723d874b2015c774fac1553d9"
            "713be561ca86f74e4f16f22e664fc17a79f30caa5ad2c04fbc447549c2810fae",
            "1548636",
            "1555348" "0",
        ],
    },
    "500.perlbench_r": {
        "binary": "perlbench_r_base.peregrine-m64",
        "arguments": ["test.pl"],
    },
    "525.x264_r": {
        "binary": "x264_r_base.peregrine-m64",
        "arguments": [
            "--dumpyuv",
            "50",
            "--frames",
            "156",
            "-o",
            "BuckBunny_New.264",
            "BuckBunny.yuv",
            "1280x720",
        ],
    },
    "502.gcc_r": {
        "binary": "cpugcc_r_base.peregrine-m64",
        "arguments": [
            "t1.c",
            "-O3",
            "-finline-limit=50000",
            "-o",
            "t1.opts-O3_-finline-limit_50000.s",
        ],
    },
}

spec_cwd: Optional[str] = None

if _args.benchmark in peregrine_benchmarks:
    binary = BinaryResource(
        local_path=f"tests/peregrine-bmarks/{_args.benchmark}-gem5"
    )
    arguments = []
elif _args.benchmark in spec_benchmarks:
    rundir = f"{_args.specdir}/benchspec/CPU/{_args.benchmark}/run/run_base_test_peregrine-m64.0000"
    spec_cwd = rundir
    binary = spec_benchmarks[_args.benchmark]["binary"]
    arguments = spec_benchmarks[_args.benchmark]["arguments"]
    binary = BinaryResource(local_path=f"{rundir}/{binary}")
else:
    raise ValueError(f"Invalid benchmark: {_args.benchmark}")

board.set_se_binary_workload(
    binary=binary,
    arguments=arguments,
    stdout_file=Path("stdout.txt"),
    stderr_file=Path("stderr.txt"),
)

if spec_cwd is not None:
    sim_processor = board.get_processor()
    cores = (
        sim_processor._all_cores()
        if isinstance(sim_processor, SwitchableProcessor)
        else sim_processor.get_cores()
    )
    for core in cores:
        cpu_simobj = core.get_simobject()
        for process in cpu_simobj.workload:
            process.cwd = spec_cwd

if _args.fast_forward and not _args.fast_only:

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
