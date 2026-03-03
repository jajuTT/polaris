// SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cassert>
#include <vector>

namespace neosim::units {

/// RISC-V GPR scoreboard — replaces Python class `riscRegState` (t3sim.py).
///
/// Models a 32-entry scoreboard replicated across num_threads threads.
/// Each GPR entry has:
///   - in_use (bool) — register is being written by a long-latency operation
///   - valid  (int)  — generation counter; 0 = no outstanding write
///
/// Non-blocking: all methods return immediately.  Callers (ThreadUnit) are
/// responsible for re-scheduling via Sparta events when a condition is not met.
class RiscReg {
public:
    static constexpr int NUM_REGS = 32;

    /// @param num_threads  Number of RISC threads (TRISC-0/1/2 + NCRISC).
    explicit RiscReg(int num_threads);

    // ----------------------------------------------------------------
    // in_use management
    // ----------------------------------------------------------------

    /// Toggle in_use for (thread, reg).  Asserts that the register was NOT
    /// already set to v (toggle semantics — setting twice is a bug).
    void set_in_use(int thread, int reg, bool v);

    /// @return true if the register is in-use (busy), false if free.
    bool check_in_use(int thread, int reg) const;

    // ----------------------------------------------------------------
    // valid management
    // ----------------------------------------------------------------

    void set_valid(int thread, int reg, int v);

    int  check_valid(int thread, int reg) const;

    /// Assert that valid[thread][reg] == v, then reset to 0.
    void reset_valid(int thread, int reg, int v);

    // ----------------------------------------------------------------
    // Diagnostic accessors
    // ----------------------------------------------------------------
    int  num_threads() const { return num_threads_; }

    void print_state() const;

private:
    int num_threads_;

    // Indexed [thread][reg]
    std::vector<std::vector<bool>> in_use_;
    std::vector<std::vector<int>>  valid_;
};

} // namespace neosim::units
