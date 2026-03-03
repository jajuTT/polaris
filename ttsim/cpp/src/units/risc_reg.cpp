// SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "neosim/units/risc_reg.hpp"

#include <cstdio>

namespace neosim::units {

RiscReg::RiscReg(int num_threads)
    : num_threads_(num_threads)
    , in_use_(num_threads, std::vector<bool>(NUM_REGS, false))
    , valid_(num_threads, std::vector<int>(NUM_REGS, 0))
{}

// ----------------------------------------------------------------
// in_use management
// ----------------------------------------------------------------

void RiscReg::set_in_use(int thread, int reg, bool v) {
    assert(thread >= 0 && thread < num_threads_);
    assert(reg >= 0 && reg < NUM_REGS);
    assert(in_use_[thread][reg] != v); // toggle semantics: must not already be v
    in_use_[thread][reg] = v;
}

bool RiscReg::check_in_use(int thread, int reg) const {
    assert(thread >= 0 && thread < num_threads_);
    assert(reg >= 0 && reg < NUM_REGS);
    return in_use_[thread][reg];
}

// ----------------------------------------------------------------
// valid management
// ----------------------------------------------------------------

void RiscReg::set_valid(int thread, int reg, int v) {
    assert(thread >= 0 && thread < num_threads_);
    assert(reg >= 0 && reg < NUM_REGS);
    assert(v >= 0);
    valid_[thread][reg] = v;
}

int RiscReg::check_valid(int thread, int reg) const {
    assert(thread >= 0 && thread < num_threads_);
    assert(reg >= 0 && reg < NUM_REGS);
    return valid_[thread][reg];
}

void RiscReg::reset_valid(int thread, int reg, [[maybe_unused]] int v) {
    assert(thread >= 0 && thread < num_threads_);
    assert(reg >= 0 && reg < NUM_REGS);
    assert(v >= 0);
    assert(valid_[thread][reg] == v);
    valid_[thread][reg] = 0;
}

// ----------------------------------------------------------------
// Diagnostic
// ----------------------------------------------------------------

void RiscReg::print_state() const {
    std::printf("RiscReg[%d threads]:\n", num_threads_);
    for (int t = 0; t < num_threads_; ++t) {
        std::printf("  thread[%d]: ", t);
        for (int r = 0; r < NUM_REGS; ++r) {
            if (in_use_[t][r] || valid_[t][r] != 0) {
                std::printf("x%d={in_use=%d,valid=%d} ", r, in_use_[t][r], valid_[t][r]);
            }
        }
        std::printf("\n");
    }
}

} // namespace neosim::units
