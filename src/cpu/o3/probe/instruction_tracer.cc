#include "cpu/o3/probe/instruction_tracer.hh"

#include "base/logging.hh"
#include "cpu/o3/dyn_inst.hh"
#include "cpu/op_class.hh"
#include "enums/OpClass.hh"
#include "sim/sim_exit.hh"

namespace gem5
{

InstructionTracer::InstructionTracer(const InstructionTracerParams &p)
    : ProbeListenerObject(p), traceStream(nullptr)
{
    traceStream = simout.create(p.output_file, false);
    if (!traceStream) {
        fatal(
            "InstructionTracer: unable to create output file '%s'",
            p.output_file);
    }
    *traceStream->stream()
        << "IP,Assembly,Category,Opcode,Branch Type,Branch Taken,"
        << "Branch Target Address,Instruction Sync,"
        << "Read Registers,Write Registers,Register Dependent IPs,"
        << "Read Addresses,Write Addresses,Memory Dependent IPs\n";
    traceStream->stream()->flush();
}

InstructionTracer::~InstructionTracer()
{
    if (traceStream)
        simout.close(traceStream);
}

void
InstructionTracer::regProbeListeners()
{
    using DynInstListener =
        ProbeListenerArg<InstructionTracer, o3::DynInstPtr>;

    connectListener<DynInstListener>(
        this, "Execute", &InstructionTracer::traceExecute);

    connectListener<ProbeListenerArg<
        InstructionTracer, std::pair<o3::DynInstPtr, PacketPtr>>>(
            this, "DataAccessComplete",
            &InstructionTracer::traceDataAccess);

    connectListener<DynInstListener>(
        this, "Commit", &InstructionTracer::traceCommit);
}

void
InstructionTracer::traceExecute(const o3::DynInstPtr &inst)
{
    onExecute(inst);
}

void
InstructionTracer::traceDataAccess(
        const std::pair<o3::DynInstPtr, PacketPtr> &arg)
{
    onDataAccess(arg.first, arg.second);
}

void
InstructionTracer::traceCommit(const o3::DynInstPtr &inst)
{
    onCommit(inst);
}

void
InstructionTracer::onCommit(const o3::DynInstPtr &inst)
{
    auto &pc = inst->pcState();
    auto &sInst = inst->staticInst;

    Addr ip = pc.instAddr();
    std::string ipStr = csprintf("0x%x", ip);
    std::string asm_str = sInst->disassemble(ip);
    std::string opcode = sInst->getName();

    OpClass opc = inst->opClass();
    const char *opClassStr =
        (static_cast<unsigned>(opc) < enums::Num_OpClass) ?
        enums::OpClassStrings[opc] : "Unknown";

    // Branch classification + resolved taken (commit-time).
    std::string branchType;
    std::string branchTaken;
    std::string branchTarget;
    if (inst->isControl()) {
        if (sInst->isIndirectCtrl())
            branchType = "indirect";
        else if (sInst->isCondCtrl())
            branchType = "direct_conditional";
        else if (sInst->isUncondCtrl())
            branchType = "direct_unconditional";

        branchTaken = pc.branching() ? "true" : "false";
        branchTarget = "";
    }

    bool instSync =
        sInst->isFullMemBarrier() ||
        sInst->isReadBarrier() ||
        sInst->isWriteBarrier() ||
        sInst->isHtmStart() ||
        sInst->isHtmStop() ||
        sInst->isHtmCancel() ||
        sInst->isSyscall();

    // Register sets (always available).
    std::vector<std::string> readRegs;
    std::vector<std::string> writeRegs;
    for (size_t i = 0; i < inst->numSrcRegs(); i++) {
        readRegs.push_back(regIdxToStr(inst->srcRegIdx(i)));
    }
    for (size_t i = 0; i < inst->numDestRegs(); i++) {
        writeRegs.push_back(regIdxToStr(inst->destRegIdx(i)));
    }

    // Fetch precomputed dependency/memory fields captured at Execute.
    const uint64_t depKey = snapshotKey(inst->threadNumber, inst->seqNum);
    DepSnapshot snap;
    auto it = depSnapshots.find(depKey);
    if (it != depSnapshots.end()) {
        snap = it->second;
        depSnapshots.erase(it);
    }

    // `DataAccessComplete` does not fire for all regular stores (which may
    // not receive a data response). As a fallback, use the committed
    // instruction's effective address when available so `Write Addresses`
    // is populated for stores in the final single-row trace.
    if (snap.writeAddrs.empty() && inst->isStore() && inst->effAddrValid()) {
        snap.writeAddrs = csprintf("0x%x(%u)", inst->effAddr, inst->effSize);
    }

    auto escapeCsv = [](const std::string &field) {
        std::string escaped;
        bool needs_quotes = false;
        for (char c : field) {
            if (c == '\"') {
                escaped += "\"\"";
                needs_quotes = true;
            } else if (c == ',' || c == '\n' || c == '\r') {
                escaped += c;
                needs_quotes = true;
            } else {
                escaped += c;
            }
        }
        if (needs_quotes)
            return std::string("\"") + escaped + "\"";
        return escaped;
    };

    auto joinRegList = [](const std::vector<std::string> &regs) {
        std::string out;
        for (size_t i = 0; i < regs.size(); ++i) {
            if (i) out += ";";
            out += regs[i];
        }
        return out;
    };

    std::ostream *out = traceStream->stream();
    *out << ipStr << ","
         << escapeCsv(asm_str) << ","
         << escapeCsv(opClassStr) << ","
         << escapeCsv(opcode) << ","
         << escapeCsv(branchType) << ","
         << (branchType.empty() ? "" : branchTaken) << ","
         << (branchType.empty() ? "" : branchTarget) << ","
         << (instSync ? "true" : "false") << ","
         << escapeCsv(joinRegList(readRegs)) << ","
         << escapeCsv(joinRegList(writeRegs)) << ","
         << escapeCsv(snap.regDeps) << ","
         << escapeCsv(snap.readAddrs) << ","
         << escapeCsv(snap.writeAddrs) << ","
         << escapeCsv(snap.memDeps) << "\n";
    out->flush();
}

void
InstructionTracer::onExecute(const o3::DynInstPtr &inst)
{
    auto &pc = inst->pcState();

    Addr ip = pc.instAddr();

    // Register read/write sets for this dynamic instruction
    std::vector<std::string> readRegs;
    std::vector<std::string> writeRegs;
    for (size_t i = 0; i < inst->numSrcRegs(); i++) {
        auto name = regIdxToStr(inst->srcRegIdx(i));
        readRegs.push_back(name);
    }
    for (size_t i = 0; i < inst->numDestRegs(); i++) {
        auto name = regIdxToStr(inst->destRegIdx(i));
        writeRegs.push_back(name);
    }

    // Register dependencies (like peregrine-trace reg_dependent_ips)
    std::vector<Addr> regDeps;
    for (const auto &reg_name : readRegs) {
        const std::string key =
            csprintf("t%d:%s", inst->threadNumber, reg_name);
        auto it = lastRegWriteIp.find(key);
        if (it != lastRegWriteIp.end()) {
            Addr depIp = it->second;
            if (depIp != ip &&
                std::find(regDeps.begin(), regDeps.end(), depIp) ==
                    regDeps.end()) {
                regDeps.push_back(depIp);
            }
        }
    }
    for (const auto &reg_name : writeRegs) {
        const std::string key =
            csprintf("t%d:%s", inst->threadNumber, reg_name);
        lastRegWriteIp[key] = ip;
    }

    auto joinIpList = [](const std::vector<Addr> &ips) {
        std::string out;
        for (size_t i = 0; i < ips.size(); ++i) {
            if (i) out += ";";
            out += csprintf("0x%x", ips[i]);
        }
        return out;
    };

    const uint64_t depKey = snapshotKey(inst->threadNumber, inst->seqNum);
    depSnapshots[depKey] = DepSnapshot{
        joinIpList(regDeps),
        "", "", "",
    };
}

void
InstructionTracer::onDataAccess(const o3::DynInstPtr &inst, PacketPtr pkt)
{
    if (!pkt)
        return;

    // Use the instruction effective address (virtual) when available to
    // match the semantics of the PIN tracer and to keep overlap checks
    // consistent with the addresses we print in the CSV.
    Addr addr = inst->effAddrValid() ? inst->effAddr : pkt->getAddr();
    unsigned size = inst->effAddrValid() ? inst->effSize : pkt->getSize();

    // For stores, the packet we see here may be a response which doesn't
    // necessarily identify as "write" via pkt->isWrite(). Use the instruction
    // classification primarily, and fall back to the packet helpers.
    bool is_read  = inst->isLoad() || inst->isStoreConditional() ||
                    inst->isAtomic() || pkt->isRead();
    bool is_write = inst->isStore() || inst->isStoreConditional() ||
                    inst->isAtomic() || pkt->isWrite();

    if (!is_read && !is_write)
        return;

    auto rangesOverlap = [](Addr a, unsigned as, Addr b, unsigned bs) {
        Addr a_end = a + as;
        Addr b_end = b + bs;
        return !(a_end <= b || b_end <= a);
    };

    std::vector<std::pair<Addr, unsigned>> readAddrs;
    std::vector<std::pair<Addr, unsigned>> writeAddrs;
    if (is_read)
        readAddrs.emplace_back(addr, size);
    if (is_write)
        writeAddrs.emplace_back(addr, size);

    std::vector<Addr> memDeps;
    if (is_read) {
        for (const auto &r : readAddrs) {
            for (const auto &w : lastMemWrites) {
                if (w.tid != inst->threadNumber)
                    continue;
                if (rangesOverlap(r.first, r.second, w.addr, w.size)) {
                    const bool already_tracked =
                        std::find(
                            memDeps.begin(), memDeps.end(), w.writerIp) !=
                        memDeps.end();
                    if (!already_tracked) {
                        memDeps.push_back(w.writerIp);
                    }
                }
            }
        }
    }
    for (const auto &w : writeAddrs) {
        MemWriteEntry entry;
        entry.tid = inst->threadNumber;
        entry.addr = w.first;
        entry.size = w.second;
        entry.writerIp = inst->pcState().instAddr();
        lastMemWrites.push_back(entry);
    }

    auto joinIpList = [](const std::vector<Addr> &ips) {
        std::string out;
        for (size_t i = 0; i < ips.size(); ++i) {
            if (i) out += ";";
            out += csprintf("0x%x", ips[i]);
        }
        return out;
    };

    auto joinAddrList =
        [](const std::vector<std::pair<Addr, unsigned>> &addrs) {
            std::string out;
            for (size_t i = 0; i < addrs.size(); ++i) {
                if (i) out += ";";
                out += csprintf(
                    "0x%x(%u)", addrs[i].first, addrs[i].second);
            }
            return out;
        };

    const uint64_t dep_key = snapshotKey(inst->threadNumber, inst->seqNum);
    auto &snap = depSnapshots[dep_key];
    // regDeps is already set by onExecute (if any)
    // append/overwrite addresses.
    snap.readAddrs = joinAddrList(readAddrs);
    snap.writeAddrs = joinAddrList(writeAddrs);
    snap.memDeps = joinIpList(memDeps);
}

// Helper: convert RegId → human-readable name (x86 / ARM)
std::string
InstructionTracer::regIdxToStr(const RegId &reg)
{
    // Use gem5's register naming helpers so ISA-specific RegIds don't
    // degrade into "?" placeholders.
    return reg.regClass().regName(reg);
}

} // namespace gem5
