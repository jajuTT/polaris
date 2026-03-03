// SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "neosim/isa/instruction.hpp"
#include "neosim/risc/trisc_mem_func.hpp"
#include "neosim/risc/trisc_regs.hpp"
#include "neosim/units/tensix_spl_reg.hpp"

namespace neosim::risc {

/// TRISC RV32 instruction execution engine — replaces Python class `triscFunc`.
///
/// Dispatches all RV32 opcodes observed in LLK kernels:
///   - R-type  : ADD, SUB, AND, OR, XOR, SLL, SRL, SRA, SLT, SLTU, MUL
///   - I-type  : ADDI, SUBI, ANDI, ORI, XORI, SLLI, SRLI, SRAI, SLTI, SLTIU
///   - Load    : LW, LH, LB, LHU, LBU  (with TensixSplReg MMR intercept)
///   - Store   : SW, SH, SB            (with TensixSplReg MMR intercept)
///   - Branch  : BEQ, BNE, BLT, BGE, BLTU, BGEU
///   - Jump    : JAL, JALR
///   - U-type  : LUI, AUIPC
///   - Other   : FENCE, CSRRW, CSRRS, CSRRC
///
/// Loads and stores that hit a memory-mapped register address are routed to
/// TensixSplReg (cfg, mop, mopSync, idleSync, instrBuffer, tileCounters).
/// Cfg-register stores also apply instruction-state side-effects (vld/bank masks,
/// conditional-valid maps) to the Instruction token — same logic as
/// TensixFunc::apply_cfg_reg_write.
///
/// No Sparta dependency — pure C++20 data model, testable on macOS ARM64.
class TriscFunc {
public:
    struct Config {
        int core_id   = 0;
        int thread_id = 0;
    };

    explicit TriscFunc(const Config&        cfg,
                       TriscMemFunc&         mem,
                       units::TensixSplReg&  spl_regs,
                       TriscRegs&            regs);

    // ----------------------------------------------------------------
    // Main entry point
    // ----------------------------------------------------------------

    /// Execute one RISC-V instruction.
    ///
    /// Updates TriscRegs, TriscMemFunc, and TensixSplReg as appropriate.
    /// For cfg-register stores, also annotates @p ins (vld/bank/cond masks).
    ///
    /// Returns the next program-counter address.
    int exec_r_ins(isa::Instruction& ins, int cycle);

    // ----------------------------------------------------------------
    // Instruction-buffer helpers (used by ThreadUnit)
    // ----------------------------------------------------------------

    /// Read the instruction-buffer register.  Clears the register on read.
    /// Returns -1 when the buffer is empty.
    int32_t read_instruction_buf_mem();

private:
    // ── R-type ──────────────────────────────────────────────────────
    int exec_add (isa::Instruction& ins);
    int exec_sub (isa::Instruction& ins);
    int exec_and (isa::Instruction& ins);
    int exec_or  (isa::Instruction& ins);
    int exec_xor (isa::Instruction& ins);
    int exec_sll (isa::Instruction& ins);
    int exec_srl (isa::Instruction& ins);
    int exec_sra (isa::Instruction& ins);
    int exec_slt (isa::Instruction& ins);
    int exec_sltu(isa::Instruction& ins);
    int exec_mul (isa::Instruction& ins);

    // ── I-type arithmetic ────────────────────────────────────────────
    int exec_addi (isa::Instruction& ins);
    int exec_subi (isa::Instruction& ins);
    int exec_andi (isa::Instruction& ins);
    int exec_ori  (isa::Instruction& ins);
    int exec_xori (isa::Instruction& ins);
    int exec_slli (isa::Instruction& ins);
    int exec_srli (isa::Instruction& ins);
    int exec_srai (isa::Instruction& ins);
    int exec_slti (isa::Instruction& ins);
    int exec_sltiu(isa::Instruction& ins);

    // ── Load ─────────────────────────────────────────────────────────
    int exec_lw (isa::Instruction& ins, int cycle);
    int exec_lh (isa::Instruction& ins);
    int exec_lb (isa::Instruction& ins);
    int exec_lhu(isa::Instruction& ins);
    int exec_lbu(isa::Instruction& ins);

    // ── Store ────────────────────────────────────────────────────────
    int exec_sw(isa::Instruction& ins, int cycle);
    int exec_sh(isa::Instruction& ins);
    int exec_sb(isa::Instruction& ins);

    // ── Branch ───────────────────────────────────────────────────────
    int exec_beq (isa::Instruction& ins);
    int exec_bne (isa::Instruction& ins);
    int exec_blt (isa::Instruction& ins);
    int exec_bge (isa::Instruction& ins);
    int exec_bltu(isa::Instruction& ins);
    int exec_bgeu(isa::Instruction& ins);

    // ── Jump / upper immediate / other ───────────────────────────────
    int exec_jal  (isa::Instruction& ins);
    int exec_jalr (isa::Instruction& ins);
    int exec_lui  (isa::Instruction& ins);
    int exec_auipc(isa::Instruction& ins);
    int exec_fence(isa::Instruction& ins);
    int exec_csr  (isa::Instruction& ins); ///< CSRRW, CSRRS, CSRRC

    // ── Store / load MMR routing helpers ─────────────────────────────

    /// Common store routing for SW (@p width=32), SH (16), SB (8).
    /// Applies width masking before writing to any register type.
    void do_store(uint32_t addr, int32_t raw_val, int width,
                  isa::Instruction& ins, int cycle);

    /// Common load routing for LW (32), LH/LHU (16), LB/LBU (8).
    /// @p sign_extend = true for LH/LB; false for LHU/LBU/LW.
    int32_t do_load(uint32_t addr, int width, bool sign_extend);

    /// Apply instruction-state side-effects of a cfg-register write.
    /// Mirrors TensixFunc::apply_cfg_reg_write.
    void apply_cfg_write_effects(int offset, int32_t val,
                                 isa::Instruction& ins);

    // ── Data members ─────────────────────────────────────────────────
    Config               cfg_;
    TriscMemFunc&        mem_;
    units::TensixSplReg& spl_regs_;
    TriscRegs&           regs_;
};

} // namespace neosim::risc
