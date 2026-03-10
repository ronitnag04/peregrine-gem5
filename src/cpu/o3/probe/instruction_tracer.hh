#ifndef __CPU_O3_PROBE_INSTRUCTION_TRACER_HH__
#define __CPU_O3_PROBE_INSTRUCTION_TRACER_HH__

#include <map>
#include <unordered_map>
#include <vector>

#include "base/output.hh"
#include "cpu/inst_seq.hh"
#include "cpu/o3/dyn_inst_ptr.hh"
#include "cpu/reg_class.hh"
#include "mem/packet.hh"
#include "params/InstructionTracer.hh"
#include "sim/probe/probe_listener_object.hh"

namespace gem5
{

/**
 * InstructionTracer listens to O3 pipeline probe points (Fetch, Dispatch,
 * Execute, ToCommit, Commit, Mispredict, Squash) and writes a unified CSV
 * trace with tick, stage, seq num, thread, PC, assembly, opcode, opclass,
 * microop flag, regs, memory address, and branch/squash flags.
 */
class InstructionTracer : public ProbeListenerObject
{
  public:
    InstructionTracer(const InstructionTracerParams &p);
    ~InstructionTracer();

    void regProbeListeners() override;

  private:
    void onExecute(const o3::DynInstPtr &inst);
    void onDataAccess(const o3::DynInstPtr &inst, PacketPtr pkt);
    void onCommit(const o3::DynInstPtr &inst);
    std::string regIdxToStr(const RegId &reg);

    void traceExecute(const o3::DynInstPtr &inst);
    void traceDataAccess(const std::pair<o3::DynInstPtr, PacketPtr> &arg);
    void traceCommit(const o3::DynInstPtr &inst);

    OutputStream *traceStream;

    struct MemWriteEntry
    {
        ThreadID tid;
        Addr addr;
        unsigned size;
        Addr writerIp;
    };

    std::map<std::string, Addr> lastRegWriteIp;
    std::vector<MemWriteEntry> lastMemWrites;

    struct DepSnapshot
    {
        std::string regDeps;   // semicolon-separated hex IPs
        std::string readAddrs; // semicolon-separated 0xADDR(SIZE)
        std::string writeAddrs;
        std::string memDeps;   // semicolon-separated hex IPs
    };

    static uint64_t
    snapshotKey(ThreadID tid, InstSeqNum sn)
    {
        const uint64_t tid_part = static_cast<uint64_t>(tid) << 48;
        const uint64_t sn_part = sn & 0x0000FFFFFFFFFFFFULL;
        return tid_part | sn_part;
    }

    std::unordered_map<uint64_t, DepSnapshot> depSnapshots;
};

} // namespace gem5

#endif
