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


# Custom FU pool with modified counts
class MyFUPool(FUPool):
    FUList = [
        IntALU(count=2),
        IntMultDiv(count=2),
        FP_ALU(count=4),
        FP_MultDiv(count=2),
        RdWrPort(count=4),
        SIMD_Unit(count=4),
    ]


class MyOutOfOrderCore(BaseCPUCore):
    def __init__(
        self,
        fetch_width,
        decode_width,
        rename_width,
        issue_width,
        wb_width,
        commit_width,
        rob_size,
        num_int_regs,
        num_fp_regs,
        lq_entries,
        sq_entries,
        branch_predictor,
    ):
        super().__init__(X86O3CPU(), ISA.X86)
        # TODO: Convert all parameter settings to use Param notation
        self.core.fetchWidth = fetch_width
        self.core.decodeWidth = decode_width
        self.core.renameWidth = rename_width
        self.core.issueWidth = issue_width  # not in concorde parameterization (split between alu, fp, l/s)
        self.core.instQueues = IQUnit(fuPool=MyFUPool())
        self.core.wbWidth = wb_width  # not in concorde parameterization
        self.core.commitWidth = commit_width

        self.core.numROBEntries = rob_size

        self.core.numPhysIntRegs = (
            num_int_regs  # not in concorde parameterization
        )
        self.core.numPhysFloatRegs = (
            num_fp_regs  # not in concorde parameterization
        )

        self.core.LQEntries = lq_entries
        self.core.SQEntries = sq_entries

        self.core.branchPred = branch_predictor


class MyOutOfOrderProcessor(BaseCPUProcessor):
    def __init__(
        self,
        fetch_width,
        decode_width,
        rename_width,
        issue_width,
        wb_width,
        commit_width,
        rob_size,
        num_int_regs,
        num_fp_regs,
        lq_entries,
        sq_entries,
        branch_predictor,
    ):
        cores = [
            MyOutOfOrderCore(
                fetch_width,
                decode_width,
                rename_width,
                issue_width,
                wb_width,
                commit_width,
                rob_size,
                num_int_regs,
                num_fp_regs,
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
        l2_assoc: int = 16,
        membus: Optional[BaseXBar] = None,
        PrefetcherCls: Optional[Type[BasePrefetcher]] = None,
    ):
        super().__init__(
            l1d_size, l1i_size, l2_size, l1d_assoc, l1i_assoc, l2_assoc, membus
        )
        self._l1d_prefetcher_cls = PrefetcherCls

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
    def __init__(self, degree: int = 0):
        super().__init__()
        self.degree = degree


processor = MyOutOfOrderProcessor(
    fetch_width=8,
    decode_width=8,
    rename_width=8,
    issue_width=8,
    wb_width=8,
    commit_width=8,
    rob_size=192,
    num_int_regs=256,
    num_fp_regs=256,
    lq_entries=128,
    sq_entries=128,
    branch_predictor=BranchPredictor(conditionalBranchPred=TAGE()),
)

main_memory = SingleChannelDDR4_2400(size="4GiB")
cache_hierarchy = MyCacheHierarchy(
    l1d_size="32KiB",
    l1i_size="32KiB",
    l2_size="1MiB",
    l1d_assoc=8,
    l1i_assoc=8,
    l2_assoc=16,
    PrefetcherCls=MyPrefetcher,
)

board = SimpleBoard(
    processor=processor,
    memory=main_memory,
    cache_hierarchy=cache_hierarchy,
    clk_freq="3GHz",
)

binary = BinaryResource(
    local_path="/home/ubuntu/peregrine-gem5/tests/peregrine-bmarks/whetstone"
)
board.set_se_binary_workload(binary)

simulator = Simulator(board=board)
simulator.run()
