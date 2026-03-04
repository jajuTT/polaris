// SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "neosim/units/pipe_unit.hpp"
#include "neosim/isa/instruction.hpp"
#include "neosim/units/mem_req.hpp"

#include <algorithm>
#include <cassert>
#include <cstdint>

namespace neosim::units {

// ============================================================
// Construction
// ============================================================

PipeUnit::PipeUnit(const Config& cfg, TensixReg& tensix_reg,
                   PipeResource& pipe_resource, Rob& rob,
                   Scratchpad& scratchpad)
    : cfg_(cfg)
    , tensix_reg_(tensix_reg)
    , pipe_resource_(pipe_resource)
    , rob_(rob)
    , scratchpad_(scratchpad)
{}

// ============================================================
// Public interface
// ============================================================

void PipeUnit::push(isa::InstrPtr ins)
{
    assert(static_cast<int>(pipe_buf_.size()) < PIPE_BUF_CAP);
    pipe_buf_.push_back(std::move(ins));
}

bool PipeUnit::is_idle() const
{
    return phase_ == Phase::IDLE && pipe_buf_.empty();
}

int PipeUnit::buf_size() const
{
    return static_cast<int>(pipe_buf_.size());
}

// ============================================================
// Utilities
// ============================================================

int PipeUnit::find_pipe_id(const std::string& name) const
{
    const auto it = std::find(cfg_.pipes.begin(), cfg_.pipes.end(), name);
    if (it == cfg_.pipes.end()) return -1;
    return static_cast<int>(it - cfg_.pipes.begin());
}

int PipeUnit::thread_of_ins() const
{
    if (active_ins_ && active_ins_->get_pipes_thread_id()) {
        return static_cast<int>(*active_ins_->get_pipes_thread_id());
    }
    return cfg_.thread_id;
}

uint32_t PipeUnit::align_bytes(uint32_t bytes, uint32_t port_width)
{
    if (port_width == 0) return bytes;
    return ((bytes + port_width - 1) / port_width) * port_width;
}

// ============================================================
// has_memory
// ============================================================

bool PipeUnit::has_memory() const
{
    if (!active_ins_) return false;
    const std::string& op = active_ins_->get_op();
    if (cfg_.flavor == Flavor::UNPACK) {
        if (op == "POP_TILES") return false;
        return active_ins_->get_src_size() > 0;
    }
    if (cfg_.flavor == Flavor::PACK) {
        if (op == "PUSH_TILES") return false;
        return active_ins_->get_src_size() > 0;
    }
    return false;  // COMPUTE never uses L1
}

// ============================================================
// Phase transition helpers
// ============================================================

PipeUnit::Phase PipeUnit::next_after_src_pipes() const
{
    switch (cfg_.flavor) {
    case Flavor::COMPUTE:
        return cfg_.enable_sync ? Phase::CHECK_VALIDS : Phase::EXECUTE;

    case Flavor::UNPACK:
        if (has_memory()) return Phase::L1_READ;
        return cfg_.enable_sync ? Phase::CHECK_VALIDS : Phase::CLEANUP;

    case Flavor::PACK:
        return Phase::FE_DELAY;
    }
    return Phase::CLEANUP;  // unreachable
}

PipeUnit::Phase PipeUnit::next_after_fe_delay() const
{
    // PACK only
    if (cfg_.enable_sync) return Phase::CHECK_VALIDS;
    if (has_memory())     return Phase::REG_READ;
    return Phase::CLEANUP;
}

PipeUnit::Phase PipeUnit::next_after_valids() const
{
    switch (cfg_.flavor) {
    case Flavor::COMPUTE:
        return Phase::EXECUTE;

    case Flavor::UNPACK:
        return has_memory() ? Phase::REG_WRITE : Phase::CLEANUP;

    case Flavor::PACK:
        return has_memory() ? Phase::REG_READ : Phase::CLEANUP;
    }
    return Phase::CLEANUP;  // unreachable
}

// ============================================================
// enter_phase — set up cycles_remaining_ for a new phase
//
// Cascading rules (see doc comment in header):
//   Timed phases that RETURN when countdown hits 0:
//     FE_DELAY, L1_READ, L1_WRITE
//   Instantaneous phases that fall through when at 0:
//     CHECK_SRC_PIPES, CHECK_VALIDS, EXECUTE, REG_WRITE, REG_READ, CLEANUP
//
// For FE_DELAY / L1_READ / L1_WRITE, the total occupancy is
// (cycles_remaining_ + 1) step() calls: the phase is entered, then
// cycles_remaining_ decrements once per step until 0, at which point the
// phase completes and the next phase is entered before returning.
// ============================================================

void PipeUnit::enter_phase(Phase p)
{
    phase_ = p;

    switch (p) {
    case Phase::FE_DELAY:
        cycles_remaining_ = PACKERFE_DELAY_CYCLES - 1;
        break;

    case Phase::EXECUTE: {
        int delay = active_ins_->get_pipe_delay();
        cycles_remaining_ = (delay > 0) ? (delay - 1) : 0;
        break;
    }

    case Phase::L1_READ: {
        const int tid = thread_of_ins();
        uint32_t bytes = static_cast<uint32_t>(active_ins_->get_src_size());
        bytes = align_bytes(bytes, static_cast<uint32_t>(cfg_.l1_port_width));
        if (bytes == 0) bytes = static_cast<uint32_t>(cfg_.l1_port_width);
        MemReq req(MemOp::READ, 0, bytes);
        req.set_thread_id(tid);
        req.set_core_id(cfg_.core_id);
        req.set_pipe_id(cfg_.pipe_id);
        if (active_ins_->get_ins_id() != 0) {
            req.set_ins_id(active_ins_->get_ins_id());
        }
        int lat = scratchpad_.submit(req);
        pending_req_id_ = req.req_id();
        cycles_remaining_ = lat - 1;
        break;
    }

    case Phase::REG_WRITE:
        cycles_remaining_ = 0;
        break;

    case Phase::REG_READ:
        cycles_remaining_ = 0;
        break;

    case Phase::L1_WRITE: {
        const int tid = thread_of_ins();
        uint32_t bytes = static_cast<uint32_t>(active_ins_->get_dst_size());
        bytes = align_bytes(bytes, static_cast<uint32_t>(cfg_.l1_port_width));
        if (bytes == 0) bytes = static_cast<uint32_t>(cfg_.l1_port_width);
        MemReq req(MemOp::WRITE, 0, bytes);
        req.set_thread_id(tid);
        req.set_core_id(cfg_.core_id);
        req.set_pipe_id(cfg_.pipe_id);
        if (active_ins_->get_ins_id() != 0) {
            req.set_ins_id(active_ins_->get_ins_id());
        }
        int lat = scratchpad_.submit(req);
        pending_req_id_ = req.req_id();
        cycles_remaining_ = lat - 1;
        break;
    }

    default:
        cycles_remaining_ = 0;
        break;
    }
}

// ============================================================
// do_check_src_pipes
// ============================================================

bool PipeUnit::do_check_src_pipes()
{
    if (!cfg_.enable_pipe_stall) return true;

    const std::string& exe_pipe = active_ins_->get_ex_pipe();
    for (const auto& pname : active_ins_->get_src_pipes()) {
        // Skip the exe_pipe in src_pipes to avoid self-block.
        if (!exe_pipe.empty() && pname == exe_pipe) continue;
        const int pid = find_pipe_id(pname);
        if (pid < 0) continue;
        const int tid = thread_of_ins();
        auto result = pipe_resource_.check_rsrc_state(pid, tid, 0, src_consec_);
        src_consec_ = result.consec_count;
        if (!result.done) return false;
    }
    return true;
}

// ============================================================
// do_check_valids
// ============================================================

bool PipeUnit::do_check_valids()
{
    if (!cfg_.enable_sync) return true;

    const int context = active_ins_->get_context();

    // Source registers: mode=1 (check valid AND not in_use).
    for (int r : active_ins_->get_src_int()) {
        int cond_chk;
        if (active_ins_->has_cond_chk_vld_upd(r, context)) {
            cond_chk = active_ins_->get_cond_chk_vld_upd(r, context);
        } else {
            cond_chk = tensix_reg_.cond_check_valid(r, context);
        }
        if (cond_chk < 0) continue;  // -1: not programmed → skip
        if (!tensix_reg_.check_valid(r, context, cond_chk, 1)) return false;
    }

    // Destination registers: mode=3 (check valid AND set in_use).
    for (int r : active_ins_->get_dst_int()) {
        int cond_chk;
        if (active_ins_->has_cond_chk_vld_upd(r, context)) {
            cond_chk = active_ins_->get_cond_chk_vld_upd(r, context);
        } else {
            cond_chk = tensix_reg_.cond_check_valid(r, context);
        }
        if (cond_chk < 0) continue;  // -1: not programmed → skip
        if (!tensix_reg_.check_valid(r, context, cond_chk, 3)) return false;
    }

    return true;
}

// ============================================================
// apply_cleanup — mirrors Python _common_pipe_cleanup
// ============================================================

void PipeUnit::apply_cleanup()
{
    assert(active_ins_);
    const int context = active_ins_->get_context();

    // 1. Update valid bits via vld/bank update masks.
    for (int r = 0; r < TensixReg::NUM_REGS; ++r) {
        const bool v_mask = active_ins_->has_vld_upd_mask(r) &&
                            active_ins_->get_vld_upd_mask(r) != 0;
        const bool b_mask = active_ins_->has_bank_upd_mask(r) &&
                            active_ins_->get_bank_upd_mask(r) != 0;
        if (v_mask || b_mask) {
            int val;
            if (active_ins_->has_cond_wri_vld_upd(r, context)) {
                val = active_ins_->get_cond_wri_vld_upd(r, context);
            } else {
                val = tensix_reg_.cond_write_valid(r, context);
            }
            // mode=3: update valid + rotate bank + clear in_use (requires in_use was set
            // by CHECK_VALIDS).  When enable_sync=false CHECK_VALIDS is skipped so in_use
            // is never set — use mode=1 (update valid + rotate bank, skip in_use).
            const int wv_mode = cfg_.enable_sync ? 3 : 1;
            tensix_reg_.write_valid(r, context, val, v_mask, b_mask, wv_mode);
        }
    }

    // 2. Update conditional valid programming.
    for (int r = 0; r < TensixReg::NUM_REGS; ++r) {
        if (active_ins_->has_cond_chk_vld_upd(r, context)) {
            int chk = active_ins_->get_cond_chk_vld_upd(r, context);
            int wri;
            if (active_ins_->has_cond_wri_vld_upd(r, context)) {
                wri = active_ins_->get_cond_wri_vld_upd(r, context);
            } else {
                wri = tensix_reg_.cond_write_valid(r, context);
            }
            tensix_reg_.write_cond_valid(r, context, chk, wri);
        }
    }

    // 3. Release dst_pipes for the instruction's thread.
    // Exception: SEMWAIT holds dst pipes busy — freed later by SEMGET/SEMPOST
    // (Python doNotFreeDstPipesforInstrs = ["SEMWAIT"]).
    const int tid = thread_of_ins();
    if (active_ins_->get_op() != "SEMWAIT") {
        for (const auto& pname : active_ins_->get_dst_pipes()) {
            const int pid = find_pipe_id(pname);
            if (pid >= 0) {
                pipe_resource_.set_rsrc_state(pid, tid, 0);
            }
        }
    }

    // 4. Remove from ROB.
    rob_.remove(active_ins_->get_ins_id());

    // 5. Release exe_pipe unless it is already in dst_pipes for the same thread.
    const std::string& exe_pipe = active_ins_->get_ex_pipe();
    if (!exe_pipe.empty() && exe_pipe != "NONE") {
        const auto& dst_pipes = active_ins_->get_dst_pipes();
        bool already_released = false;
        if (active_ins_->get_pipes_thread_id()) {
            const int ptid = static_cast<int>(*active_ins_->get_pipes_thread_id());
            if (ptid == cfg_.thread_id &&
                std::find(dst_pipes.begin(), dst_pipes.end(), exe_pipe)
                    != dst_pipes.end()) {
                already_released = true;
            }
        }
        if (!already_released) {
            const int pid = find_pipe_id(exe_pipe);
            if (pid >= 0) {
                pipe_resource_.set_rsrc_state(pid, tid, 0);
            }
        }
    }
}

// ============================================================
// Main step
//
// Cascading rule summary:
//   CHECK_SRC_PIPES, CHECK_VALIDS, EXECUTE(0), REG_WRITE(0), REG_READ(0):
//     fall through to next phase in the same step() call.
//   FE_DELAY, L1_READ, L1_WRITE:
//     when countdown > 0: decrement and RETURN.
//     when countdown == 0: call complete() if needed, enter next phase, RETURN.
//     The next phase is then processed starting from the FOLLOWING step() call.
//
// This ensures that L1 latency and FE front-end delay consume the correct
// number of step() calls as modelled by Python yield env.timeout(N).
// ============================================================

PipeUnit::StepResult PipeUnit::step(int /*cycle*/)
{
    StepResult result;

    // ── IDLE: dequeue next instruction ──────────────────────────────
    if (phase_ == Phase::IDLE) {
        if (pipe_buf_.empty()) return result;
        active_ins_ = pipe_buf_.front();
        pipe_buf_.pop_front();
        src_consec_ = 0;

        if (cfg_.enable_pipe_stall) {
            enter_phase(Phase::CHECK_SRC_PIPES);
        } else {
            enter_phase(next_after_src_pipes());
        }
    }

    // ── CHECK_SRC_PIPES ─────────────────────────────────────────────
    if (phase_ == Phase::CHECK_SRC_PIPES) {
        if (!do_check_src_pipes()) return result;
        enter_phase(next_after_src_pipes());
        // fall through to next phase
    }

    // ── FE_DELAY (PACK only) ─────────────────────────────────────────
    // Timed: decrement each step; when 0, enter next phase and RETURN.
    if (phase_ == Phase::FE_DELAY) {
        if (cycles_remaining_ > 0) {
            --cycles_remaining_;
            return result;
        }
        // countdown reached 0 — enter next phase but do NOT cascade this step
        enter_phase(next_after_fe_delay());
        return result;
    }

    // ── CHECK_VALIDS ─────────────────────────────────────────────────
    if (phase_ == Phase::CHECK_VALIDS) {
        if (!do_check_valids()) return result;
        enter_phase(next_after_valids());
        // fall through to next phase
    }

    // ── L1_READ (UNPACK) — timed, submit in enter_phase ─────────────
    if (phase_ == Phase::L1_READ) {
        if (cycles_remaining_ > 0) {
            --cycles_remaining_;
            return result;
        }
        // countdown complete: finalise the scratchpad request
        scratchpad_.complete(pending_req_id_);
        pending_req_id_ = 0;
        // enter the next phase — CHECK_VALIDS (if sync) or REG_WRITE (if not)
        if (cfg_.enable_sync) {
            enter_phase(Phase::CHECK_VALIDS);
        } else {
            enter_phase(Phase::REG_WRITE);
        }
        // Do NOT cascade: let the next phase run in the following step()
        return result;
    }

    // ── EXECUTE (COMPUTE) — instantaneous when at 0, falls through ──
    if (phase_ == Phase::EXECUTE) {
        if (cycles_remaining_ > 0) {
            --cycles_remaining_;
            return result;
        }
        enter_phase(Phase::CLEANUP);
        // fall through to CLEANUP
    }

    // ── REG_WRITE (UNPACK) — instantaneous, falls through ───────────
    if (phase_ == Phase::REG_WRITE) {
        // cycles_remaining_ == 0 always; fall through
        enter_phase(Phase::CLEANUP);
    }

    // ── REG_READ (PACK) — instantaneous, falls through to L1_WRITE ──
    if (phase_ == Phase::REG_READ) {
        enter_phase(Phase::L1_WRITE);
        // fall through to L1_WRITE
    }

    // ── L1_WRITE (PACK) — timed, submit in enter_phase ──────────────
    if (phase_ == Phase::L1_WRITE) {
        if (cycles_remaining_ > 0) {
            --cycles_remaining_;
            return result;
        }
        // countdown complete: finalise the scratchpad request
        scratchpad_.complete(pending_req_id_);
        pending_req_id_ = 0;
        enter_phase(Phase::CLEANUP);
        // Do NOT cascade: let CLEANUP run in the following step()
        return result;
    }

    // ── CLEANUP ──────────────────────────────────────────────────────
    if (phase_ == Phase::CLEANUP) {
        apply_cleanup();
        result.completed_ins = active_ins_;
        active_ins_.reset();
        phase_ = Phase::IDLE;
        return result;
    }

    return result;
}

} // namespace neosim::units
