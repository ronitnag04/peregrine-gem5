from m5.objects.Probe import ProbeListenerObject
from m5.params import *
from m5.proxy import *


class InstructionTracer(ProbeListenerObject):
    type = "InstructionTracer"
    cxx_class = "gem5::InstructionTracer"
    cxx_header = "cpu/o3/probe/instruction_tracer.hh"

    # Output CSV file for committed instruction trace
    output_file = Param.String("trace.csv", "Instruction trace output file")
