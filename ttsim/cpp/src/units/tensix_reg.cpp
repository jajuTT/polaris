// SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "neosim/units/tensix_reg.hpp"

#include <cstdio>

namespace neosim::units {

TensixReg::TensixReg() {
    // valids_ default to 0 (zero-init); in_use_ default to false.
    // bank_sel_ default to 0 (both contexts start on bank 0).
    // cond_check_valid_ / cond_write_valid_ initialised in init().
    for (int r = 0; r < NUM_REGS; ++r) {
        for (int b = 0; b < NUM_BANKS; ++b) {
            valids_[r][b]          = 0;
            in_use_[r][b]          = false;
            acc_thread_id_[r][b]   = -1;
            acc_cnt_to_valid_[r][b] = -1;
        }
        for (int c = 0; c < NUM_CONTEXTS; ++c) {
            bank_sel_[r][c]        = 0;
            cond_check_valid_[r][c] = -1;
            cond_write_valid_[r][c] = -1;
        }
    }
}

void TensixReg::init(int core_id, const std::vector<std::vector<int>>& order_scheme) {
    core_id_ = core_id;

    for (int r = 0; r < NUM_REGS; ++r) {
        int order_index = 0;
        for (int c = 0; c < NUM_CONTEXTS; ++c) {
            bank_sel_[r][c] = 0;

            bool no_scheme = order_scheme.empty()
                             || r >= static_cast<int>(order_scheme.size())
                             || order_scheme[r].empty();

            if (no_scheme) {
                cond_check_valid_[r][c] = -1;
                cond_write_valid_[r][c] = -1;
            } else if (c == order_scheme[r][order_index]) {
                cond_check_valid_[r][c] = order_index;
                int last = static_cast<int>(order_scheme[r].size()) - 1;
                if (order_index == last) {
                    order_index = 0;
                    cond_write_valid_[r][c] = 0;
                } else {
                    ++order_index;
                    cond_write_valid_[r][c] = order_index;
                }
            } else {
                cond_check_valid_[r][c] = -1;
                cond_write_valid_[r][c] = -1;
            }
        }
    }
}

// ----------------------------------------------------------------
// Bank selection
// ----------------------------------------------------------------

int TensixReg::peek_curr_bank(int reg, int context) const {
    return bank_sel_[reg][context];
}

int TensixReg::get_next_bank(int reg, int context) {
    int curr = bank_sel_[reg][context];
    bank_sel_[reg][context] = (curr + 1) % NUM_BANKS;
    return curr; // return OLD bank (same as Python getNextBankId)
}

// ----------------------------------------------------------------
// Non-blocking read accessors
// ----------------------------------------------------------------

int TensixReg::read_valid(int reg, int context) const {
    return valids_[reg][bank_sel_[reg][context]];
}

int TensixReg::read_curr_bank(int reg, int context) const {
    return bank_sel_[reg][context];
}

// ----------------------------------------------------------------
// write_valid
// ----------------------------------------------------------------

void TensixReg::write_valid(int reg, int context, int val,
                            bool v_mask, bool b_mask, int mode) {
    assert(val >= 0);
    assert(!(v_mask && !b_mask) && !(!v_mask && b_mask));

    int cb = bank_sel_[reg][context]; // current bank (peek)

    // Early return 1: valid at current bank already equals val.
    if (valids_[reg][cb] == val) {
        if (mode == 2 || mode == 3) {
            assert(in_use_[reg][cb]);
            in_use_[reg][cb] = false;
        }
        acc_thread_id_[reg][cb] = context;
        if (!v_mask && !b_mask) {
            acc_cnt_to_valid_[reg][cb] += 1;
        } else {
            acc_cnt_to_valid_[reg][cb] = 0;
        }
        return;
    }

    // Early return 2: masks are both false → no real update.
    if (!v_mask && !b_mask) {
        if (mode == 2 || mode == 3) {
            assert(in_use_[reg][cb]);
            in_use_[reg][cb] = false;
        }
        acc_thread_id_[reg][cb]   = context;
        acc_cnt_to_valid_[reg][cb] += 1;
        return;
    }

    // Real update path (v_mask == b_mask == true).
    switch (mode) {
        case 0: // Skip valid, skip in_use — access-count tracking only
            acc_thread_id_[reg][cb]   = context;
            acc_cnt_to_valid_[reg][cb] += 1;
            break;

        case 1: // Update valid + rotate bank; skip in_use
            acc_thread_id_[reg][cb]    = context;
            acc_cnt_to_valid_[reg][cb] = 0;
            // get_next_bank returns the current bank and advances bank_sel.
            valids_[reg][get_next_bank(reg, context)] = val;
            break;

        case 2: // Skip valid; reset in_use — no bank rotation
            assert(in_use_[reg][cb]);
            acc_thread_id_[reg][cb]   = context;
            acc_cnt_to_valid_[reg][cb] += 1;
            in_use_[reg][cb] = false;
            break;

        case 3: // Update valid + rotate bank; reset in_use
            assert(in_use_[reg][cb]);
            acc_thread_id_[reg][cb]    = context;
            acc_cnt_to_valid_[reg][cb] = 0;
            in_use_[reg][cb] = false;
            valids_[reg][get_next_bank(reg, context)] = val;
            break;

        default:
            assert(false);
    }
}

// ----------------------------------------------------------------
// check_valid
// ----------------------------------------------------------------

bool TensixReg::check_valid(int reg, int context, int val, int mode) {
    assert(val >= 0);
    int cb = bank_sel_[reg][context];

    switch (mode) {
        case 0: // Always true
            return true;

        case 1: // valid == val AND not in_use
            return (valids_[reg][cb] == val) && !in_use_[reg][cb];

        case 2: // If not in_use: set in_use, return true; else false
            if (!in_use_[reg][cb]) {
                in_use_[reg][cb] = true;
                return true;
            }
            return false;

        case 3: // valid == val AND not in_use: set in_use, return true; else false
            if ((valids_[reg][cb] == val) && !in_use_[reg][cb]) {
                in_use_[reg][cb] = true;
                return true;
            }
            return false;

        default:
            assert(false);
            return false;
    }
}

// ----------------------------------------------------------------
// Conditional valid programming
// ----------------------------------------------------------------

void TensixReg::write_cond_valid(int reg, int context,
                                 int cond_chk_val, int cond_wri_val) {
    // Sentinel -2 (IGNORE) means "no update".
    assert(cond_chk_val >= -2 && cond_chk_val <= 3);
    assert(cond_wri_val >= -2 && cond_wri_val <= 3);
    if (cond_chk_val != -2) {
        cond_check_valid_[reg][context] = cond_chk_val;
        cond_write_valid_[reg][context] = cond_wri_val;
    }
}

int TensixReg::cond_check_valid(int reg, int context) const {
    return cond_check_valid_[reg][context];
}

int TensixReg::cond_write_valid(int reg, int context) const {
    return cond_write_valid_[reg][context];
}

// ----------------------------------------------------------------
// Diagnostic accessors
// ----------------------------------------------------------------

bool TensixReg::in_use(int reg, int bank) const {
    return in_use_[reg][bank];
}

int TensixReg::acc_thread_id(int reg, int bank) const {
    return acc_thread_id_[reg][bank];
}

int TensixReg::acc_cnt_to_valid(int reg, int bank) const {
    return acc_cnt_to_valid_[reg][bank];
}

void TensixReg::print_state() const {
    std::printf("TensixReg[core=%d]:\n", core_id_);
    for (int r = 0; r < NUM_REGS; ++r) {
        std::printf("  reg[%d]: ", r);
        for (int b = 0; b < NUM_BANKS; ++b) {
            std::printf("bank[%d]={valid=%d,in_use=%d,accThId=%d,accCnt=%d} ",
                        b, valids_[r][b], in_use_[r][b],
                        acc_thread_id_[r][b], acc_cnt_to_valid_[r][b]);
        }
        std::printf("\n");
        std::printf("  bank_sel: ");
        for (int c = 0; c < NUM_CONTEXTS; ++c) {
            std::printf("ctx[%d]=%d ", c, bank_sel_[r][c]);
        }
        std::printf("\n");
    }
}

} // namespace neosim::units
