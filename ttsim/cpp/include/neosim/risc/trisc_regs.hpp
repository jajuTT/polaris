// SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <cassert>
#include <cstdint>
#include <set>

namespace neosim::risc {

/// TRISC register file — replaces Python class `triscRegs` (triscFunc.py).
///
/// Models the register file for a single TRISC thread:
///   riscgpr[64]  — RISC-V GPRs (x0–x31 standard + x32–x63 extensions).
///                  Initialised from Config: x0=0, x2=stack_ptr, x3=global_ptr.
///   csr[4096]    — Control/Status Registers.
///   trisc_id[1]  — Thread identifier register.
///
/// "Temporary" registers {0,2,5,6,7,28,29,30,31} must be written before they
/// are read (assert fires on first read if not yet written).  All other riscgpr
/// registers auto-zero on first read if not yet written.
/// Initialization is tracked via a separate bool array, so -1 is a valid value.
///
/// `is_mmr(addr)` maps a byte address into a register type + word offset,
/// enabling TRISC load/store instructions to detect MMR accesses.
class TriscRegs {
public:
    // ----------------------------------------------------------------
    // Register counts
    // ----------------------------------------------------------------
    static constexpr int NUM_RISCGPR = 64;
    static constexpr int NUM_CSR     = 4096;
    static constexpr int NUM_TRISC_ID = 1;

    // ----------------------------------------------------------------
    // MMR address ranges (match Python triscRegs defaults)
    // ----------------------------------------------------------------
    static constexpr uint32_t RISCGPR_BASE = 0xee000000u;
    static constexpr uint32_t RISCGPR_SIZE = 64u;          ///< bytes
    static constexpr uint32_t CSR_BASE     = 0xef000000u;
    static constexpr uint32_t CSR_SIZE     = 4096u;        ///< bytes
    static constexpr uint32_t TRISC_ID_BASE = 0x0000a71cu;
    static constexpr uint32_t TRISC_ID_SIZE = 1u;          ///< bytes

    // ----------------------------------------------------------------
    // Register type tag
    // ----------------------------------------------------------------
    enum class RegType { NONE, RISCGPR, CSR, TRISC_ID };

    struct MmrInfo {
        RegType type   = RegType::NONE;
        int     offset = -1; ///< word offset from region base; -1 if NONE
    };

    // ----------------------------------------------------------------
    // Construction configuration
    // ----------------------------------------------------------------
    struct Config {
        int      core_id    = 0;
        int      thread_id  = 0;
        uint32_t stack_ptr  = 0; ///< initial value for x2 (sp)
        uint32_t global_ptr = 0; ///< initial value for x3 (gp)
    };

    explicit TriscRegs(const Config& cfg);

    // ----------------------------------------------------------------
    // Register read / write
    // ----------------------------------------------------------------

    /// Read a RISC-V GPR.
    ///
    /// For "temporary" registers {0,2,5,6,7,28,29,30,31}: asserts if the
    /// register has not been written yet.
    /// For all other registers: auto-zeros on first read if not yet written.
    int32_t read_riscgpr(int r);

    /// Write a RISC-V GPR.
    void write_riscgpr(int r, int32_t value);

    int32_t read_csr(int idx) const;
    void    write_csr(int idx, int32_t value);

    int32_t read_trisc_id() const;

    // ----------------------------------------------------------------
    // Generic typed accessors (used by triscFunc load/store helpers)
    // ----------------------------------------------------------------
    int32_t read_reg(int offset, RegType type);
    void    write_reg(int offset, int32_t value, RegType type);

    // ----------------------------------------------------------------
    // MMR address resolution  (mirrors Python triscRegs.__ismmr__)
    // ----------------------------------------------------------------

    /// Map a byte address to a register type + word offset.
    /// Returns {NONE, -1} if the address is not in any known MMR range.
    MmrInfo is_mmr(uint32_t addr) const;

    // ----------------------------------------------------------------
    // Diagnostic
    // ----------------------------------------------------------------
    int core_id()   const { return core_id_; }
    int thread_id() const { return thread_id_; }

    void print_state(RegType type) const;

private:
    int core_id_;
    int thread_id_;

    /// Register indices that must be initialised before they are read.
    static const std::set<int> TEMP_REGS; // {0, 2, 5, 6, 7, 28, 29, 30, 31}

    std::array<int32_t, NUM_RISCGPR>   riscgpr_{};
    std::array<bool,    NUM_RISCGPR>   riscgpr_initialized_{}; ///< tracks which GPRs have been written
    std::array<int32_t, NUM_CSR>       csr_{};
    std::array<int32_t, NUM_TRISC_ID>  trisc_id_{};
};

} // namespace neosim::risc
