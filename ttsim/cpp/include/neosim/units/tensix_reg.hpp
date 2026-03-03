// SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <cassert>
#include <cstdint>
#include <vector>

namespace neosim::units {

/// Tensix wide register file state — replaces Python class `ttReg` (t3sim.py).
///
/// Models 4 named registers (srcA=0, srcB=1, srcS=2, dst=3) each with 2 banks,
/// accessed by up to NUM_CONTEXTS (4) contexts (UNPACKER/MATH/SFPU/PACKER).
///
/// Each bank has an independent valid bit and in-use flag used for
/// producer-consumer synchronisation across Tensix threads.  Bank selection
/// rotates per context so that successive writes go to alternating banks.
///
/// Blocking poll semantics (Python `yield env.timeout(1)`) are intentionally
/// absent here: check_valid / check_rsrc_state return bool immediately.
/// The caller (ThreadUnit / PipeUnit in Track 6) is responsible for
/// re-scheduling via Sparta events when false is returned.
class TensixReg {
public:
    static constexpr int NUM_REGS     = 4; ///< srcA, srcB, srcS, dst
    static constexpr int NUM_BANKS    = 2;
    static constexpr int NUM_CONTEXTS = 4; ///< UNPACKER=0 MATH=1 SFPU=2 PACKER=3

    TensixReg();

    /// Initialise from the config `orderScheme` array.
    ///
    /// @param core_id       Core identifier used in diagnostic messages.
    /// @param order_scheme  order_scheme[reg] = vector of thread-context IDs
    ///                      that access that register in priority order.
    ///                      Empty outer vector → all cond_*_valid initialised to -1.
    void init(int core_id, const std::vector<std::vector<int>>& order_scheme);

    // ----------------------------------------------------------------
    // Bank selection
    // ----------------------------------------------------------------

    /// Return the current bank for (reg, context) without advancing.
    int peek_curr_bank(int reg, int context) const;

    /// Advance bank selection for (reg, context) and return the PREVIOUS bank
    /// (the one that was current before rotation).  Matches Python getNextBankId.
    int get_next_bank(int reg, int context);

    // ----------------------------------------------------------------
    // Non-blocking read accessors (no side-effects)
    // ----------------------------------------------------------------
    int read_valid(int reg, int context) const;
    int read_curr_bank(int reg, int context) const;

    // ----------------------------------------------------------------
    // write_valid — update register bank state.
    //
    // mode 0 — skip valid and in_use (only update access-count tracking)
    // mode 1 — update valid + rotate bank; skip in_use
    // mode 2 — skip valid; reset in_use (no bank rotation)
    // mode 3 — update valid + rotate bank; reset in_use
    //
    // v_mask / b_mask — must be equal (asserted). When both are true a real
    // valid/bank update is performed (bank rotates).  When both are false,
    // only acc_cnt tracking is updated and no bank rotation occurs.
    // ----------------------------------------------------------------
    void write_valid(int reg, int context, int val, bool v_mask, bool b_mask, int mode);

    // ----------------------------------------------------------------
    // check_valid — synchronous condition check; returns true if met.
    //
    // mode 0 — always returns true
    // mode 1 — true iff valid[reg][curr_bank] == val AND NOT in_use
    // mode 2 — if NOT in_use: sets in_use = true, returns true; else false
    // mode 3 — if valid == val AND NOT in_use: sets in_use = true, returns true; else false
    //
    // Returns false when the condition is not yet satisfied.  The caller must
    // retry on the next simulated cycle.
    // ----------------------------------------------------------------
    bool check_valid(int reg, int context, int val, int mode);

    // ----------------------------------------------------------------
    // Conditional valid programming (SETDVALID / CLEARDVALID side-effects)
    // ----------------------------------------------------------------

    /// Update cond_check_valid / cond_write_valid for (reg, context).
    /// A cond_chk_val of ValueStatus::IGNORE (-2) is a no-op.
    void write_cond_valid(int reg, int context, int cond_chk_val, int cond_wri_val);

    int cond_check_valid(int reg, int context) const;
    int cond_write_valid(int reg, int context) const;

    // ----------------------------------------------------------------
    // Diagnostic accessors (used by unit tests and TensixRegUnit)
    // ----------------------------------------------------------------
    bool in_use(int reg, int bank) const;
    int  acc_thread_id(int reg, int bank) const;
    int  acc_cnt_to_valid(int reg, int bank) const;

    void print_state() const;

private:
    int core_id_ = 0;

    // Indexed [reg][bank]
    std::array<std::array<int,  NUM_BANKS>, NUM_REGS> valids_{};
    std::array<std::array<bool, NUM_BANKS>, NUM_REGS> in_use_{};
    std::array<std::array<int,  NUM_BANKS>, NUM_REGS> acc_thread_id_{};
    std::array<std::array<int,  NUM_BANKS>, NUM_REGS> acc_cnt_to_valid_{};

    // Indexed [reg][context]
    std::array<std::array<int, NUM_CONTEXTS>, NUM_REGS> bank_sel_{};
    std::array<std::array<int, NUM_CONTEXTS>, NUM_REGS> cond_check_valid_{};
    std::array<std::array<int, NUM_CONTEXTS>, NUM_REGS> cond_write_valid_{};
};

} // namespace neosim::units
