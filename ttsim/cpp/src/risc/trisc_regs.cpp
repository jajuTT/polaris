// SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "neosim/risc/trisc_regs.hpp"

#include <cstdio>

namespace neosim::risc {

// "Temporary" RISC-V registers that must be initialised before first read.
const std::set<int> TriscRegs::TEMP_REGS = {0, 2, 5, 6, 7, 28, 29, 30, 31};

TriscRegs::TriscRegs(const Config& cfg)
    : core_id_(cfg.core_id)
    , thread_id_(cfg.thread_id)
{
    riscgpr_.fill(-1);
    csr_.fill(-1);
    trisc_id_.fill(-1);

    // Architectural initial values
    riscgpr_[0] = 0;                         // x0 = zero register
    riscgpr_[2] = static_cast<int32_t>(cfg.stack_ptr);   // x2 = sp
    riscgpr_[3] = static_cast<int32_t>(cfg.global_ptr);  // x3 = gp

    trisc_id_[0] = cfg.thread_id;
}

// ----------------------------------------------------------------
// riscgpr read/write
// ----------------------------------------------------------------

int32_t TriscRegs::read_riscgpr(int r) {
    assert(r >= 0 && r < NUM_RISCGPR);
    if (TEMP_REGS.count(r)) {
        // Temporary registers must be initialised before read.
        assert(riscgpr_[r] != -1);
    } else if (riscgpr_[r] == -1) {
        // Non-temporary registers auto-zero on first read.
        riscgpr_[r] = 0;
    }
    return riscgpr_[r];
}

void TriscRegs::write_riscgpr(int r, int32_t value) {
    assert(r >= 0 && r < NUM_RISCGPR);
    riscgpr_[r] = value;
}

// ----------------------------------------------------------------
// csr read/write
// ----------------------------------------------------------------

int32_t TriscRegs::read_csr(int idx) const {
    assert(idx >= 0 && idx < NUM_CSR);
    return csr_[idx];
}

void TriscRegs::write_csr(int idx, int32_t value) {
    assert(idx >= 0 && idx < NUM_CSR);
    csr_[idx] = value;
}

// ----------------------------------------------------------------
// trisc_id
// ----------------------------------------------------------------

int32_t TriscRegs::read_trisc_id() const {
    return trisc_id_[0];
}

// ----------------------------------------------------------------
// Generic typed accessors
// ----------------------------------------------------------------

int32_t TriscRegs::read_reg(int offset, RegType type) {
    switch (type) {
        case RegType::RISCGPR:  return read_riscgpr(offset);
        case RegType::CSR:      return read_csr(offset);
        case RegType::TRISC_ID: return read_trisc_id();
        default:
            assert(false && "Unknown RegType");
            return -1;
    }
}

void TriscRegs::write_reg(int offset, int32_t value, RegType type) {
    switch (type) {
        case RegType::RISCGPR:
            write_riscgpr(offset, value);
            break;
        case RegType::CSR:
            write_csr(offset, value);
            break;
        case RegType::TRISC_ID:
            assert(offset == 0);
            trisc_id_[0] = value;
            break;
        default:
            assert(false && "Unknown RegType");
    }
}

// ----------------------------------------------------------------
// MMR address resolution
// ----------------------------------------------------------------

TriscRegs::MmrInfo TriscRegs::is_mmr(uint32_t addr) const {
    if (addr >= RISCGPR_BASE && addr < RISCGPR_BASE + RISCGPR_SIZE) {
        return {RegType::RISCGPR, static_cast<int>((addr - RISCGPR_BASE) / 4)};
    }
    if (addr >= CSR_BASE && addr < CSR_BASE + CSR_SIZE) {
        return {RegType::CSR, static_cast<int>((addr - CSR_BASE) / 4)};
    }
    if (addr >= TRISC_ID_BASE && addr < TRISC_ID_BASE + TRISC_ID_SIZE) {
        return {RegType::TRISC_ID, 0};
    }
    return {RegType::NONE, -1};
}

// ----------------------------------------------------------------
// Diagnostic
// ----------------------------------------------------------------

void TriscRegs::print_state(RegType type) const {
    switch (type) {
        case RegType::RISCGPR:
            std::printf("TriscRegs[core=%d,thread=%d] riscgpr:\n",
                        core_id_, thread_id_);
            for (int r = 0; r < NUM_RISCGPR; ++r) {
                if (riscgpr_[r] != -1) {
                    std::printf("  x%d=0x%x\n", r, riscgpr_[r]);
                }
            }
            break;
        case RegType::CSR:
            std::printf("TriscRegs[core=%d,thread=%d] csr (non -1):\n",
                        core_id_, thread_id_);
            for (int i = 0; i < NUM_CSR; ++i) {
                if (csr_[i] != -1) {
                    std::printf("  csr[%d]=0x%x\n", i, csr_[i]);
                }
            }
            break;
        default:
            break;
    }
}

} // namespace neosim::risc
