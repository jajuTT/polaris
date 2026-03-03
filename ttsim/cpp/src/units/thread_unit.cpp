// SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "neosim/units/thread_unit.hpp"

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <memory>

namespace neosim::units {

// ============================================================
// Construction
// ============================================================

ThreadUnit::ThreadUnit(const Config&       cfg,
                       TensixFunc&         tensix_func,
                       risc::TriscFunc&    trisc_func,
                       TensixReg&          tensix_reg,
                       RiscReg&            risc_reg,
                       PipeResource&       pipe_resource,
                       TensixSplReg&       spl_regs,
                       risc::TriscRegs&    trisc_regs)
    : cfg_(cfg)
    , tensix_func_(tensix_func)
    , trisc_func_(trisc_func)
    , tensix_reg_(tensix_reg)
    , risc_reg_(risc_reg)
    , pipe_resource_(pipe_resource)
    , spl_regs_(spl_regs)
    , trisc_regs_(trisc_regs)
{}

// ============================================================
// Kernel loading / sequencing
// ============================================================

void ThreadUnit::load_kernel(const std::string& name,
                              const InstrMap&    instrs,
                              uint32_t           start_addr,
                              uint32_t           end_addr)
{
    kernel_instrs_[name] = instrs;
    kernel_ranges_[name] = KernelRange{name, start_addr, end_addr};
}

void ThreadUnit::enqueue_kernel(const std::string& name)
{
    assert(kernel_ranges_.count(name) && "enqueue_kernel: kernel not loaded");
    kernel_queue_.push_back(name);
}

// ============================================================
// Helpers
// ============================================================

isa::InstrPtr ThreadUnit::find_ins(uint32_t addr) const
{
    if (active_kernel_.empty()) return nullptr;
    const auto kit = kernel_instrs_.find(active_kernel_);
    if (kit == kernel_instrs_.end()) return nullptr;
    const auto iit = kit->second.find(addr);
    if (iit == kit->second.end()) return nullptr;
    // Deep-copy the instruction so each execution has its own annotation state.
    return std::make_shared<isa::Instruction>(*iit->second);
}

bool ThreadUnit::is_load_op(const std::string& op)
{
    return op == "LW"  || op == "LH"  || op == "LB" ||
           op == "LHU" || op == "LBU";
}

bool ThreadUnit::is_branch_op(const std::string& op)
{
    return op == "BEQ" || op == "BNE" || op == "BLT" ||
           op == "BGE" || op == "BLTU"|| op == "BGEU";
}

bool ThreadUnit::try_acquire_exe_pipe(const std::string& pipe_name, int thread_id)
{
    if (pipe_name.empty() || pipe_name == "NONE") return true;
    const auto it = std::find(cfg_.pipes.begin(), cfg_.pipes.end(), pipe_name);
    if (it == cfg_.pipes.end()) return true;  // unknown pipe — skip check
    const int pipe_id = static_cast<int>(it - cfg_.pipes.begin());
    return pipe_resource_.set_rsrc_state(pipe_id, thread_id, 1);
}

void ThreadUnit::release_exe_pipe(const std::string& pipe_name, int thread_id)
{
    if (pipe_name.empty() || pipe_name == "NONE") return;
    const auto it = std::find(cfg_.pipes.begin(), cfg_.pipes.end(), pipe_name);
    if (it == cfg_.pipes.end()) return;
    const int pipe_id = static_cast<int>(it - cfg_.pipes.begin());
    pipe_resource_.set_rsrc_state(pipe_id, thread_id, 0);
}

// ============================================================
// T6.1: Fetch
// ============================================================

bool ThreadUnit::step_fetch(int /*cycle*/)
{
    // Activate next kernel if none is running.
    if (active_kernel_.empty()) {
        if (kernel_queue_.empty()) return false;  // nothing to run
        active_kernel_ = kernel_queue_.front();
        kernel_queue_.pop_front();
        pc_           = kernel_ranges_.at(active_kernel_).start_addr;
        prev_pc_      = static_cast<uint32_t>(-1);
        fetch_done_   = false;
    }

    // Stall if input_buf is full.
    if (static_cast<int>(input_buf_.size()) >= INPUT_BUF_CAP) return false;

    // If PC hasn't changed since last cycle, stall (waiting for branch/JALR
    // to update PC via decode).
    if (pc_ == prev_pc_) return false;

    // PC == 0 marks end of kernel (final JAL/JALR returns to 0).
    if (pc_ == 0) {
        fetch_done_    = true;
        active_kernel_ = "";
        return false;
    }

    isa::InstrPtr ins = find_ins(pc_);
    if (!ins) {
        // Address not in instruction map — skip (matches Python warning path).
        std::printf("WARNING: ThreadUnit: no instruction at PC=0x%08x\n", pc_);
        pc_ += 4;
        return false;
    }

    prev_pc_ = pc_;
    input_buf_.push_back(ins);
    return true;
}

// ============================================================
// T6.2: Decode
// ============================================================

bool ThreadUnit::step_decode(int cycle)
{
    if (input_buf_.empty()) return false;

    if (input_buf_.front()->is_tt() || input_buf_.front()->is_mop()) {
        // TT / MOP: route to mop_buf
        if (static_cast<int>(mop_buf_.size()) >= MOP_BUF_CAP) return false;
        isa::InstrPtr ins = input_buf_.front();
        input_buf_.pop_front();
        mop_buf_.push_back(ins);
        return true;
    }

    // RISC instruction: execute functionally via trisc_func.
    // Output queue for scoreboard: stall if full.
    if (cfg_.enable_scoreboard &&
        static_cast<int>(risc_check_buf_.size()) >= RCHECK_BUF_CAP) {
        return false;
    }

    isa::InstrPtr ins = input_buf_.front();
    input_buf_.pop_front();

    // Functional execution — updates PC.
    // Save the old PC as prev_pc_ (mirrors Python: prevPC = pc; pc = nextAddr).
    // This ensures fetch sees pc_ != prev_pc_ and proceeds to the new address.
    const uint32_t old_pc = pc_;
    int next_pc = trisc_func_.exec_r_ins(*ins, cycle);
    pc_      = static_cast<uint32_t>(next_pc);
    prev_pc_ = old_pc;

    // Handle instruction buffer (TT instruction written via SW to instrBuf).
    int32_t decoded_word = trisc_func_.read_instruction_buf_mem();
    if (decoded_word != -1) {
        // Decoded TT instruction from instrBuf write.
        // Wrap in an Instruction and queue into mop_buf.
        // (Caller is responsible for full decode via tensix_func in Track 8.)
        auto tt_ins = std::make_shared<isa::Instruction>();
        tt_ins->set_mem_info("raw_word", static_cast<int64_t>(decoded_word));
        tt_ins->set_thread_id(static_cast<uint32_t>(cfg_.thread_id));
        tt_ins->set_core_id(static_cast<uint32_t>(cfg_.core_id));
        if (static_cast<int>(mop_buf_.size()) < MOP_BUF_CAP) {
            mop_buf_.push_back(tt_ins);
        }
    }

    // Route to scoreboard check pipeline.
    if (cfg_.enable_scoreboard) {
        risc_check_buf_.push_back(ins);
    }

    return true;
}

// ============================================================
// T6.3a: Scoreboard check
// ============================================================

bool ThreadUnit::step_scoreboard_check(int cycle)
{
    // ── Load a new instruction if the stage is idle ──────────────────
    if (!sboard_check_.ins) {
        if (risc_check_buf_.empty()) return false;
        sboard_check_.ins = risc_check_buf_.front();
        risc_check_buf_.pop_front();

        // Apply branch misprediction penalty.
        sboard_check_.branch_penalty_remaining =
            is_branch_op(sboard_check_.ins->get_op()) ? cfg_.branch_mispredict_lat : 0;
        sboard_check_.src_checked = false;
        sboard_check_.dst_set     = false;
        sboard_check_.ready       = false;
    }

    // ── Branch penalty ───────────────────────────────────────────────
    if (sboard_check_.branch_penalty_remaining > 0) {
        --sboard_check_.branch_penalty_remaining;
        return false;
    }

    // ── Check src and dst registers not in-use ───────────────────────
    if (!sboard_check_.src_checked) {
        bool all_free = true;
        for (int r : sboard_check_.ins->get_src_int()) {
            if (r != 0 && risc_reg_.check_in_use(cfg_.thread_id, r)) {
                all_free = false;
                break;
            }
        }
        if (all_free) {
            for (int r : sboard_check_.ins->get_dst_int()) {
                if (r != 0 && risc_reg_.check_in_use(cfg_.thread_id, r)) {
                    all_free = false;
                    break;
                }
            }
        }
        if (!all_free) return false;  // stall, retry next cycle
        sboard_check_.src_checked = true;
    }

    // ── Set dst registers in-use (for load ops, or when forwarding off) ─
    if (!sboard_check_.dst_set) {
        const std::string& op = sboard_check_.ins->get_op();
        const bool should_set = (!cfg_.enable_forwarding ||
                                 (cfg_.enable_forwarding && is_load_op(op)));
        if (should_set) {
            for (int r : sboard_check_.ins->get_dst_int()) {
                if (r != 0) {
                    risc_reg_.set_in_use(cfg_.thread_id, r, true);
                }
            }
        }
        sboard_check_.dst_set = true;
    }

    // ── Forward to the risc execution tracker ────────────────────────
    if (static_cast<int>(risc_exec_trk_buf_.size()) >= cfg_.risc_pipe_depth) {
        return false;  // tracker buffer full
    }

    sboard_check_.ins->set_mem_info("scoreboard_enter_cycle",
                                    static_cast<int64_t>(cycle));
    risc_exec_trk_buf_.push_back(sboard_check_.ins);
    sboard_check_.ins.reset();
    return true;
}

// ============================================================
// T6.3b: Scoreboard reset
// ============================================================

bool ThreadUnit::step_scoreboard_reset(int cycle)
{
    if (risc_exec_trk_buf_.empty()) return false;

    // Count down the pipeline latency: pop on the risc_pipe_depth-th call.
    // With risc_pipe_depth = N, calls 1..(N-1) are delay ticks; call N pops.
    if (exec_trk_delay_ < cfg_.risc_pipe_depth - 1) {
        ++exec_trk_delay_;
        return false;
    }
    exec_trk_delay_ = 0;

    isa::InstrPtr ins = risc_exec_trk_buf_.front();
    risc_exec_trk_buf_.pop_front();

    // Clear in-use for dst registers (for load ops or when forwarding off).
    const std::string& op     = ins->get_op();
    const bool         do_clr = (!cfg_.enable_forwarding ||
                                  (cfg_.enable_forwarding && is_load_op(op)));
    if (do_clr) {
        for (int r : ins->get_dst_int()) {
            if (r != 0) {
                risc_reg_.set_in_use(cfg_.thread_id, r, false);
            }
        }
    }

    (void)cycle;  // unused parameter; retained for API consistency
    return true;
}

// ============================================================
// T6.4: MOP decode
// ============================================================

bool ThreadUnit::step_mop_decode(int cycle)
{
    // ── Continue emitting words from an in-progress MOP expansion ────
    if (mop_state_.active) {
        if (static_cast<int>(instr_buf_.size()) >= INSTR_BUF_CAP) return false;

        if (mop_state_.idx < static_cast<int>(mop_state_.words.size())) {
            // Emit the next sub-instruction word as an Instruction token.
            auto sub = std::make_shared<isa::Instruction>();
            sub->set_mem_info("raw_word",
                static_cast<int64_t>(mop_state_.words[
                    static_cast<std::size_t>(mop_state_.idx)]));
            sub->set_thread_id(static_cast<uint32_t>(cfg_.thread_id));
            sub->set_core_id(static_cast<uint32_t>(cfg_.core_id));
            instr_buf_.push_back(sub);
            ++mop_state_.idx;
            return true;
        }

        // All MOP words emitted — clear state, clear mopSync, advance PC.
        // Mirrors Python mopDecode: "prevPC = pc; pc = prevPC + 4" after expansion.
        spl_regs_.write_reg(cfg_.thread_id, 0,
                            TensixSplReg::SplRegType::MOP_SYNC);
        mop_state_.active = false;
        mop_state_.words.clear();
        mop_state_.idx    = 0;
        prev_pc_ = pc_;  // save aligned TT instruction address before advancing
        pc_     += 4;
        return true;
    }

    // ── Pick up a new instruction from mop_buf ───────────────────────
    if (mop_buf_.empty()) return false;
    if (static_cast<int>(instr_buf_.size()) >= INSTR_BUF_CAP) return false;

    isa::InstrPtr ins = mop_buf_.front();
    mop_buf_.pop_front();

    if (ins->is_mop()) {
        // Set mopSync = 1 while expanding.
        spl_regs_.write_reg(cfg_.thread_id, 1,
                            TensixSplReg::SplRegType::MOP_SYNC);

        // Expand MOP into raw instruction words (one per future cycle).
        mop_state_.words  = tensix_func_.build_ins_from_mop(*ins);
        mop_state_.idx    = 0;
        mop_state_.active = true;

        // Emit the first word this cycle (if buffer has room — already checked).
        if (!mop_state_.words.empty()) {
            auto sub = std::make_shared<isa::Instruction>();
            sub->set_mem_info("raw_word",
                static_cast<int64_t>(mop_state_.words[0]));
            sub->set_thread_id(static_cast<uint32_t>(cfg_.thread_id));
            sub->set_core_id(static_cast<uint32_t>(cfg_.core_id));
            instr_buf_.push_back(sub);
            mop_state_.idx = 1;
        }
        return true;
    }

    // Non-MOP pass-through: clear mopSync, forward to instr_buf_.
    // For true TT instructions (kind is set), advance PC here since step_decode
    // did not update pc_ for TT instructions (only RISC decode updates pc_).
    // For SW-triggered raw instructions (kind not set, is_tt()=false), step_decode
    // already advanced pc_, so we do NOT advance again.
    spl_regs_.write_reg(cfg_.thread_id, 0, TensixSplReg::SplRegType::MOP_SYNC);
    instr_buf_.push_back(ins);
    if (ins->is_tt()) {
        // Mirrors Python mopDecode: "prevPC = pc; pc = prevPC + 4".
        prev_pc_ = pc_;  // save aligned TT instruction address before advancing
        pc_     += 4;
    }
    (void)cycle;
    return true;
}

// ============================================================
// T6.5: Arbiter
// ============================================================

ThreadUnit::RouteResult ThreadUnit::step_arbiter(int cycle)
{
    using Phase = ArbiterPhase;

    // ── IDLE: fetch next instruction ─────────────────────────────────
    if (arbiter_.phase == Phase::IDLE) {
        isa::InstrPtr ins;

        if (replay_.mode() == ReplayState::Mode::EXECUTE) {
            ins = replay_.exec_replay_list();
            if (!ins) return {};  // replay buffer empty
        } else {
            if (instr_buf_.empty()) return {};
            ins = instr_buf_.front();
            instr_buf_.pop_front();

            if (ins->is_replay()) {
                // REPLAY instruction: update replay state, do NOT execute.
                int load      = static_cast<int>(ins->get_mem_info("replay_load_mode"));
                int exec_whl  = static_cast<int>(ins->get_mem_info("replay_execute_while_loading"));
                int start_idx = static_cast<int>(ins->get_mem_info("replay_start_idx"));
                int len       = static_cast<int>(ins->get_mem_info("replay_len"));
                replay_.update_mode(load, exec_whl, start_idx, len);
                return {};
            }

            // In LOAD or LOAD_EXECUTE mode: store instruction.
            if (replay_.mode() == ReplayState::Mode::LOAD ||
                replay_.mode() == ReplayState::Mode::LOAD_EXECUTE) {
                // Copy before storing (replay list owns a separate instance).
                auto copy = std::make_shared<isa::Instruction>(*ins);
                replay_.load_replay_list(copy);
                if (replay_.mode() == ReplayState::Mode::LOAD) {
                    return {};  // LOAD: skip execution
                }
                // LOAD_EXECUTE: fall through to execute
            }
        }

        // Functionally execute the TT instruction.
        tensix_func_.exec_tt_ins(*ins, cycle);
        ins->set_thread_id(static_cast<uint32_t>(cfg_.thread_id));
        ins->set_core_id(static_cast<uint32_t>(cfg_.core_id));

        // Insert into ROB.
        arbiter_.rob_id = rob_.append(ins);
        ins->set_ins_id(arbiter_.rob_id);

        arbiter_.ins        = ins;
        arbiter_.src_consec = 0;
        arbiter_.exe_consec = 0;

        // If pipe stall/sync checks are disabled, skip straight to routing.
        if (!cfg_.enable_pipe_stall && !cfg_.enable_sync) {
            arbiter_.phase = Phase::ROUTING;
        } else {
            arbiter_.phase = Phase::STALL_DST_PIPES;
        }
    }

    isa::InstrPtr& ins = arbiter_.ins;

    // ── STALL_DST_PIPES ──────────────────────────────────────────────
    if (arbiter_.phase == Phase::STALL_DST_PIPES) {
        if (!cfg_.enable_pipe_stall) {
            arbiter_.phase = Phase::CHECK_SRC_PIPES;
        } else {
            bool all_set = true;
            const auto& dst_pipes = ins->get_dst_pipes();
            for (const auto& pname : dst_pipes) {
                const auto it = std::find(cfg_.pipes.begin(), cfg_.pipes.end(), pname);
                if (it == cfg_.pipes.end()) continue;
                const int pid = static_cast<int>(it - cfg_.pipes.begin());
                const int tid = ins->get_pipes_thread_id()
                                ? static_cast<int>(*ins->get_pipes_thread_id())
                                : cfg_.thread_id;
                if (!pipe_resource_.set_rsrc_state(pid, tid, 1)) {
                    all_set = false;
                    break;
                }
            }
            if (!all_set) return {};
            arbiter_.phase = Phase::CHECK_SRC_PIPES;
        }
    }

    // ── CHECK_SRC_PIPES ──────────────────────────────────────────────
    if (arbiter_.phase == Phase::CHECK_SRC_PIPES) {
        if (!cfg_.enable_pipe_stall) {
            arbiter_.phase = Phase::WAIT_EXE_PIPE;
        } else {
            const std::string& exe_pipe = ins->get_ex_pipe();
            bool all_idle = true;

            for (const auto& pname : ins->get_src_pipes()) {
                // Skip if exe_pipe is one of the src pipes (avoid self-block).
                if (!exe_pipe.empty() && pname == exe_pipe) continue;
                const auto it = std::find(cfg_.pipes.begin(), cfg_.pipes.end(), pname);
                if (it == cfg_.pipes.end()) continue;
                const int pid = static_cast<int>(it - cfg_.pipes.begin());
                const int tid = ins->get_pipes_thread_id()
                                ? static_cast<int>(*ins->get_pipes_thread_id())
                                : cfg_.thread_id;
                auto result = pipe_resource_.check_rsrc_state(pid, tid, 0,
                                                              arbiter_.src_consec);
                arbiter_.src_consec = result.consec_count;
                if (!result.done) { all_idle = false; break; }
            }
            if (!all_idle) return {};
            arbiter_.phase = Phase::WAIT_EXE_PIPE;
        }
    }

    // ── WAIT_EXE_PIPE ────────────────────────────────────────────────
    if (arbiter_.phase == Phase::WAIT_EXE_PIPE) {
        if (!cfg_.enable_pipe_stall) {
            arbiter_.phase = Phase::CHECK_VALIDS;
        } else {
            const std::string& exe_pipe = ins->get_ex_pipe();
            if (!exe_pipe.empty() && exe_pipe != "NONE") {
                const auto it = std::find(cfg_.pipes.begin(), cfg_.pipes.end(), exe_pipe);
                if (it != cfg_.pipes.end()) {
                    const int pid = static_cast<int>(it - cfg_.pipes.begin());
                    // Skip if exe_pipe is in dst_pipes for the same thread
                    // (Python: avoid self-deadlock when src releases dst).
                    bool skip = false;
                    if (ins->get_pipes_thread_id()) {
                        const int ptid = static_cast<int>(*ins->get_pipes_thread_id());
                        const auto& dst_pipes = ins->get_dst_pipes();
                        if (ptid == cfg_.thread_id &&
                            std::find(dst_pipes.begin(), dst_pipes.end(), exe_pipe)
                                != dst_pipes.end()) {
                            skip = true;
                        }
                    }
                    if (!skip) {
                        auto result = pipe_resource_.check_rsrc_state(
                            pid, cfg_.thread_id, 0, arbiter_.exe_consec);
                        arbiter_.exe_consec = result.consec_count;
                        if (!result.done) return {};
                        // Acquire exe pipe.
                        if (!try_acquire_exe_pipe(exe_pipe, cfg_.thread_id)) return {};
                    }
                }
            }
            arbiter_.phase = Phase::CHECK_VALIDS;
        }
    }

    // ── CHECK_VALIDS ─────────────────────────────────────────────────
    if (arbiter_.phase == Phase::CHECK_VALIDS) {
        if (!cfg_.enable_sync) {
            arbiter_.phase = Phase::WAIT_ROB_HEAD;
        } else {
            const int  context    = ins->get_context();
            const bool skip_valid = (ins->get_op() == "STALLWAIT" ||
                                     ins->get_op() == "SETRWC");
            bool all_ok = true;

            // Check source registers.
            // skip_valid ops (STALLWAIT, SETRWC): mode=2 (acquire in_use only).
            // All others: mode=1 (check valid==cond_chk AND not in_use).
            // condCheckValid==-1 means "not programmed" → skip with no stall.
            const int src_mode = skip_valid ? 2 : 1;
            for (int r : ins->get_src_int()) {
                int cond_chk;
                if (ins->has_cond_chk_vld_upd(r, context)) {
                    cond_chk = ins->get_cond_chk_vld_upd(r, context);
                } else {
                    cond_chk = tensix_reg_.cond_check_valid(r, context);
                }
                if (cond_chk < 0) continue;  // -1: not programmed → skip
                if (!tensix_reg_.check_valid(r, context, cond_chk, src_mode)) {
                    all_ok = false;
                    break;
                }
            }
            if (all_ok) {
                // Check destination registers.
                // skip_valid ops: mode=2; all others: mode=3 (check valid AND set in_use).
                const int dst_mode = skip_valid ? 2 : 3;
                for (int r : ins->get_dst_int()) {
                    int cond_chk;
                    if (ins->has_cond_chk_vld_upd(r, context)) {
                        cond_chk = ins->get_cond_chk_vld_upd(r, context);
                    } else {
                        cond_chk = tensix_reg_.cond_check_valid(r, context);
                    }
                    if (cond_chk < 0) continue;  // -1: not programmed → skip
                    if (!tensix_reg_.check_valid(r, context, cond_chk, dst_mode)) {
                        all_ok = false;
                        break;
                    }
                }
            }
            if (!all_ok) return {};
            arbiter_.phase = Phase::WAIT_ROB_HEAD;
        }
    }

    // ── WAIT_ROB_HEAD (in-order issue) ───────────────────────────────
    if (arbiter_.phase == Phase::WAIT_ROB_HEAD) {
        if (cfg_.enable_in_order_issue) {
            // Only issue if this instruction is at the head of the ROB.
            if (!rob_.is_head(arbiter_.rob_id)) return {};
        }
        arbiter_.phase = Phase::ROUTING;
    }

    // ── ROUTING ─────────────────────────────────────────────────────
    assert(arbiter_.phase == Phase::ROUTING);

    isa::InstrPtr routed_ins = arbiter_.ins;
    arbiter_.ins.reset();
    arbiter_.phase = Phase::IDLE;

    const std::string& ex_pipe = routed_ins->get_ex_pipe();
    if (ex_pipe.empty() || ex_pipe == "NONE") {
        // No pipe needed: remove from ROB immediately.
        rob_.remove(arbiter_.rob_id);
        return {routed_ins, ""};
    }

    return {routed_ins, ex_pipe};
}

// ============================================================
// Main step
// ============================================================

ThreadUnit::RouteResult ThreadUnit::step(int cycle)
{
    // Process stages in reverse pipeline order to allow flow in one step.
    RouteResult r = step_arbiter(cycle);
    step_mop_decode(cycle);
    step_scoreboard_reset(cycle);
    step_scoreboard_check(cycle);
    step_decode(cycle);
    step_fetch(cycle);
    return r;
}

// ============================================================
// all_done
// ============================================================

bool ThreadUnit::all_done() const
{
    return fetch_done_
        && input_buf_.empty()
        && mop_buf_.empty()
        && risc_check_buf_.empty()
        && risc_exec_trk_buf_.empty()
        && !mop_state_.active
        && instr_buf_.empty()
        && arbiter_.phase == ArbiterPhase::IDLE
        && rob_.empty();
}

} // namespace neosim::units
