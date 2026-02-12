from m5.objects import (
    TAGE,
    X86O3CPU,
    BranchPredictor,
    TournamentBP,
)
from m5.params import Param

from gem5.components.boards.simple_board import SimpleBoard
from gem5.components.cachehierarchies.classic.private_l1_shared_l2_cache_hierarchy import (
    PrivateL1SharedL2CacheHierarchy,
)
from gem5.components.memory.single_channel import SingleChannelDDR4_2400
from gem5.components.processors.base_cpu_core import BaseCPUCore
from gem5.components.processors.base_cpu_processor import BaseCPUProcessor
from gem5.isas import ISA
from gem5.resources.resource import (
    BinaryResource,
    obtain_resource,
)
from gem5.simulate.simulator import Simulator


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
        self.core.fetchWidth = fetch_width
        self.core.decodeWidth = decode_width
        self.core.renameWidth = rename_width
        self.core.issueWidth = issue_width
        self.core.wbWidth = wb_width
        self.core.commitWidth = commit_width

        self.core.numROBEntries = rob_size

        self.core.numPhysIntRegs = num_int_regs
        self.core.numPhysFloatRegs = num_fp_regs

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
cache_hierarchy = PrivateL1SharedL2CacheHierarchy(
    l1d_size="32KiB", l1i_size="32KiB", l2_size="1MiB"
)

board = SimpleBoard(
    processor=processor,
    memory=main_memory,
    cache_hierarchy=cache_hierarchy,
    clk_freq="3GHz",
)

# binary = obtain_resource("x86-hello64-static")
binary = BinaryResource(
    local_path="/home/ubuntu/gem5/tests/peregrine-bmarks/branch_storm"
)
board.set_se_binary_workload(binary)

simulator = Simulator(board=board)
simulator.run()
