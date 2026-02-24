import argparse
from typing import (
    Callable,
    Optional,
    Type,
)

from m5.objects import (
    TAGE,
    X86O3CPU,
    BasePrefetcher,
    BaseXBar,
    BranchPredictor,
    FUPool,
    IQUnit,
    L2XBar,
    LocalBP,
    StridePrefetcher,
    TournamentBP,
)
from m5.objects.FuncUnit import *
from m5.objects.FuncUnitConfig import *

from gem5.components.boards.abstract_board import AbstractBoard
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
from gem5.isas import ISA
from gem5.resources.resource import BinaryResource
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
        ],
    )
    return parser.parse_args()


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
    ):
        super().__init__(X86O3CPU(), ISA.X86)
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
    ):
        cores = [
            MyOutOfOrderCore(
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
            )
        ]
        super().__init__(cores)


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

processor = MyOutOfOrderProcessor(
    fetch_width=_args.fetch_width,
    decode_width=_args.decode_width,
    rename_width=_args.rename_width,
    wb_width=_args.wb_width,
    commit_width=_args.commit_width,
    issue_width=_issue_width,
    rob_size=_args.rob_size,
    lq_entries=_args.lq_entries,
    sq_entries=_args.sq_entries,
    branch_predictor=branch_predictor,
)

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

binary = BinaryResource(
    local_path=f"/home/ubuntu/peregrine-gem5/tests/peregrine-bmarks/{_args.benchmark}-gem5"
)
board.set_se_binary_workload(binary)

simulator = Simulator(board=board)
simulator.run()
