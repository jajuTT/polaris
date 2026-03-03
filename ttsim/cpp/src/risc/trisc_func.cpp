// SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "neosim/risc/trisc_func.hpp"

#include <cassert>
#include <cstdint>

namespace neosim::risc {

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

TriscFunc::TriscFunc(const Config&        cfg,
                     TriscMemFunc&         mem,
                     units::TensixSplReg&  spl_regs,
                     TriscRegs&            regs)
    : cfg_(cfg)
    , mem_(mem)
    , spl_regs_(spl_regs)
    , regs_(regs)
{}

// ---------------------------------------------------------------------------
// Main dispatcher
// ---------------------------------------------------------------------------

int TriscFunc::exec_r_ins(isa::Instruction& ins, int cycle)
{
    const std::string& op = ins.get_op();
    int next_addr = -1;

    if      (op == "ADD")   next_addr = exec_add(ins);
    else if (op == "SUB")   next_addr = exec_sub(ins);
    else if (op == "AND")   next_addr = exec_and(ins);
    else if (op == "OR")    next_addr = exec_or(ins);
    else if (op == "XOR")   next_addr = exec_xor(ins);
    else if (op == "SLL")   next_addr = exec_sll(ins);
    else if (op == "SRL")   next_addr = exec_srl(ins);
    else if (op == "SRA")   next_addr = exec_sra(ins);
    else if (op == "SLT")   next_addr = exec_slt(ins);
    else if (op == "SLTU")  next_addr = exec_sltu(ins);
    else if (op == "MUL")   next_addr = exec_mul(ins);
    else if (op == "ADDI")  next_addr = exec_addi(ins);
    else if (op == "SUBI")  next_addr = exec_subi(ins);
    else if (op == "ANDI")  next_addr = exec_andi(ins);
    else if (op == "ORI")   next_addr = exec_ori(ins);
    else if (op == "XORI")  next_addr = exec_xori(ins);
    else if (op == "SLLI")  next_addr = exec_slli(ins);
    else if (op == "SRLI")  next_addr = exec_srli(ins);
    else if (op == "SRAI")  next_addr = exec_srai(ins);
    else if (op == "SLTI")  next_addr = exec_slti(ins);
    else if (op == "SLTIU") next_addr = exec_sltiu(ins);
    else if (op == "LW")    next_addr = exec_lw(ins, cycle);
    else if (op == "LH")    next_addr = exec_lh(ins);
    else if (op == "LB")    next_addr = exec_lb(ins);
    else if (op == "LHU")   next_addr = exec_lhu(ins);
    else if (op == "LBU")   next_addr = exec_lbu(ins);
    else if (op == "SW")    next_addr = exec_sw(ins, cycle);
    else if (op == "SH")    next_addr = exec_sh(ins);
    else if (op == "SB")    next_addr = exec_sb(ins);
    else if (op == "BEQ")   next_addr = exec_beq(ins);
    else if (op == "BNE")   next_addr = exec_bne(ins);
    else if (op == "BLT")   next_addr = exec_blt(ins);
    else if (op == "BGE")   next_addr = exec_bge(ins);
    else if (op == "BLTU")  next_addr = exec_bltu(ins);
    else if (op == "BGEU")  next_addr = exec_bgeu(ins);
    else if (op == "JAL")   next_addr = exec_jal(ins);
    else if (op == "JALR")  next_addr = exec_jalr(ins);
    else if (op == "LUI")   next_addr = exec_lui(ins);
    else if (op == "AUIPC") next_addr = exec_auipc(ins);
    else if (op == "FENCE") next_addr = exec_fence(ins);
    else if (op == "CSRRW" || op == "CSRRS" || op == "CSRRC")
        next_addr = exec_csr(ins);
    else {
        // Unknown opcode — step over and continue (matches Python warning + skip)
        next_addr = static_cast<int>(ins.get_addr()) + 4;
    }

    assert(next_addr != -1 && "TriscFunc::exec_r_ins: no handler matched");
    return next_addr;
}

// ---------------------------------------------------------------------------
// Instruction-buffer helper
// ---------------------------------------------------------------------------

int32_t TriscFunc::read_instruction_buf_mem()
{
    int32_t val = spl_regs_.read_reg(0, units::TensixSplReg::SplRegType::INSTR_BUF);
    if (val == -1) return -1;
    spl_regs_.write_reg(0, -1, units::TensixSplReg::SplRegType::INSTR_BUF);
    return val;
}

// ---------------------------------------------------------------------------
// R-type handlers
// ---------------------------------------------------------------------------

int TriscFunc::exec_add(isa::Instruction& ins)
{
    auto src = ins.get_src_int();
    auto dst = ins.get_dst_int();
    regs_.write_riscgpr(dst[0], regs_.read_riscgpr(src[0]) + regs_.read_riscgpr(src[1]));
    return static_cast<int>(ins.get_addr()) + 4;
}

int TriscFunc::exec_sub(isa::Instruction& ins)
{
    auto src = ins.get_src_int();
    auto dst = ins.get_dst_int();
    regs_.write_riscgpr(dst[0], regs_.read_riscgpr(src[0]) - regs_.read_riscgpr(src[1]));
    return static_cast<int>(ins.get_addr()) + 4;
}

int TriscFunc::exec_and(isa::Instruction& ins)
{
    auto src = ins.get_src_int();
    auto dst = ins.get_dst_int();
    regs_.write_riscgpr(dst[0], regs_.read_riscgpr(src[0]) & regs_.read_riscgpr(src[1]));
    return static_cast<int>(ins.get_addr()) + 4;
}

int TriscFunc::exec_or(isa::Instruction& ins)
{
    auto src = ins.get_src_int();
    auto dst = ins.get_dst_int();
    regs_.write_riscgpr(dst[0], regs_.read_riscgpr(src[0]) | regs_.read_riscgpr(src[1]));
    return static_cast<int>(ins.get_addr()) + 4;
}

int TriscFunc::exec_xor(isa::Instruction& ins)
{
    auto src = ins.get_src_int();
    auto dst = ins.get_dst_int();
    regs_.write_riscgpr(dst[0], regs_.read_riscgpr(src[0]) ^ regs_.read_riscgpr(src[1]));
    return static_cast<int>(ins.get_addr()) + 4;
}

int TriscFunc::exec_sll(isa::Instruction& ins)
{
    auto src = ins.get_src_int();
    auto dst = ins.get_dst_int();
    int32_t a = regs_.read_riscgpr(src[0]);
    int32_t shamt = regs_.read_riscgpr(src[1]) & 0x1F;
    regs_.write_riscgpr(dst[0], static_cast<int32_t>(static_cast<uint32_t>(a) << shamt));
    return static_cast<int>(ins.get_addr()) + 4;
}

int TriscFunc::exec_srl(isa::Instruction& ins)
{
    auto src = ins.get_src_int();
    auto dst = ins.get_dst_int();
    int32_t a = regs_.read_riscgpr(src[0]);
    int32_t shamt = regs_.read_riscgpr(src[1]) & 0x1F;
    regs_.write_riscgpr(dst[0], static_cast<int32_t>(static_cast<uint32_t>(a) >> shamt));
    return static_cast<int>(ins.get_addr()) + 4;
}

int TriscFunc::exec_sra(isa::Instruction& ins)
{
    auto src = ins.get_src_int();
    auto dst = ins.get_dst_int();
    int32_t a = regs_.read_riscgpr(src[0]);
    int32_t shamt = regs_.read_riscgpr(src[1]) & 0x1F;
    // In C++20, signed right-shift of int32_t is arithmetic (sign-extending).
    regs_.write_riscgpr(dst[0], a >> shamt);
    return static_cast<int>(ins.get_addr()) + 4;
}

int TriscFunc::exec_slt(isa::Instruction& ins)
{
    auto src = ins.get_src_int();
    auto dst = ins.get_dst_int();
    int32_t a = regs_.read_riscgpr(src[0]);
    int32_t b = regs_.read_riscgpr(src[1]);
    regs_.write_riscgpr(dst[0], (a < b) ? 1 : 0);
    return static_cast<int>(ins.get_addr()) + 4;
}

int TriscFunc::exec_sltu(isa::Instruction& ins)
{
    auto src = ins.get_src_int();
    auto dst = ins.get_dst_int();
    uint32_t a = static_cast<uint32_t>(regs_.read_riscgpr(src[0]));
    uint32_t b = static_cast<uint32_t>(regs_.read_riscgpr(src[1]));
    regs_.write_riscgpr(dst[0], (a < b) ? 1 : 0);
    return static_cast<int>(ins.get_addr()) + 4;
}

int TriscFunc::exec_mul(isa::Instruction& ins)
{
    auto src = ins.get_src_int();
    auto dst = ins.get_dst_int();
    // Low 32 bits of 64-bit product.
    int64_t prod = static_cast<int64_t>(regs_.read_riscgpr(src[0]))
                 * static_cast<int64_t>(regs_.read_riscgpr(src[1]));
    regs_.write_riscgpr(dst[0], static_cast<int32_t>(prod));
    return static_cast<int>(ins.get_addr()) + 4;
}

// ---------------------------------------------------------------------------
// I-type arithmetic handlers
// ---------------------------------------------------------------------------

int TriscFunc::exec_addi(isa::Instruction& ins)
{
    auto src = ins.get_src_int();
    auto dst = ins.get_dst_int();
    auto imm = ins.get_imm();
    regs_.write_riscgpr(dst[0], regs_.read_riscgpr(src[0]) + imm[0]);
    return static_cast<int>(ins.get_addr()) + 4;
}

int TriscFunc::exec_subi(isa::Instruction& ins)
{
    auto src = ins.get_src_int();
    auto dst = ins.get_dst_int();
    auto imm = ins.get_imm();
    regs_.write_riscgpr(dst[0], regs_.read_riscgpr(src[0]) - imm[0]);
    return static_cast<int>(ins.get_addr()) + 4;
}

int TriscFunc::exec_andi(isa::Instruction& ins)
{
    auto src = ins.get_src_int();
    auto dst = ins.get_dst_int();
    auto imm = ins.get_imm();
    regs_.write_riscgpr(dst[0], regs_.read_riscgpr(src[0]) & imm[0]);
    return static_cast<int>(ins.get_addr()) + 4;
}

int TriscFunc::exec_ori(isa::Instruction& ins)
{
    auto src = ins.get_src_int();
    auto dst = ins.get_dst_int();
    auto imm = ins.get_imm();
    regs_.write_riscgpr(dst[0], regs_.read_riscgpr(src[0]) | imm[0]);
    return static_cast<int>(ins.get_addr()) + 4;
}

int TriscFunc::exec_xori(isa::Instruction& ins)
{
    auto src = ins.get_src_int();
    auto dst = ins.get_dst_int();
    auto imm = ins.get_imm();
    regs_.write_riscgpr(dst[0], regs_.read_riscgpr(src[0]) ^ imm[0]);
    return static_cast<int>(ins.get_addr()) + 4;
}

int TriscFunc::exec_slli(isa::Instruction& ins)
{
    auto src = ins.get_src_int();
    auto dst = ins.get_dst_int();
    auto imm = ins.get_imm();
    int32_t a = regs_.read_riscgpr(src[0]);
    int32_t shamt = imm[0] & 0x1F;
    regs_.write_riscgpr(dst[0], static_cast<int32_t>(static_cast<uint32_t>(a) << shamt));
    return static_cast<int>(ins.get_addr()) + 4;
}

int TriscFunc::exec_srli(isa::Instruction& ins)
{
    auto src = ins.get_src_int();
    auto dst = ins.get_dst_int();
    auto imm = ins.get_imm();
    int32_t a = regs_.read_riscgpr(src[0]);
    int32_t shamt = imm[0] & 0x1F;
    regs_.write_riscgpr(dst[0], static_cast<int32_t>(static_cast<uint32_t>(a) >> shamt));
    return static_cast<int>(ins.get_addr()) + 4;
}

int TriscFunc::exec_srai(isa::Instruction& ins)
{
    auto src = ins.get_src_int();
    auto dst = ins.get_dst_int();
    auto imm = ins.get_imm();
    int32_t a = regs_.read_riscgpr(src[0]);
    int32_t shamt = imm[0] & 0x1F;
    regs_.write_riscgpr(dst[0], a >> shamt);
    return static_cast<int>(ins.get_addr()) + 4;
}

int TriscFunc::exec_slti(isa::Instruction& ins)
{
    auto src = ins.get_src_int();
    auto dst = ins.get_dst_int();
    auto imm = ins.get_imm();
    regs_.write_riscgpr(dst[0], (regs_.read_riscgpr(src[0]) < imm[0]) ? 1 : 0);
    return static_cast<int>(ins.get_addr()) + 4;
}

int TriscFunc::exec_sltiu(isa::Instruction& ins)
{
    auto src = ins.get_src_int();
    auto dst = ins.get_dst_int();
    auto imm = ins.get_imm();
    uint32_t a = static_cast<uint32_t>(regs_.read_riscgpr(src[0]));
    uint32_t b = static_cast<uint32_t>(imm[0]);
    regs_.write_riscgpr(dst[0], (a < b) ? 1 : 0);
    return static_cast<int>(ins.get_addr()) + 4;
}

// ---------------------------------------------------------------------------
// Load handlers
// ---------------------------------------------------------------------------

int TriscFunc::exec_lw(isa::Instruction& ins, int /*cycle*/)
{
    auto src = ins.get_src_int();
    auto dst = ins.get_dst_int();
    auto imm = ins.get_imm();
    uint32_t addr = static_cast<uint32_t>(regs_.read_riscgpr(src[0]) + imm[0]);
    regs_.write_riscgpr(dst[0], do_load(addr, 32, false));
    return static_cast<int>(ins.get_addr()) + 4;
}

int TriscFunc::exec_lh(isa::Instruction& ins)
{
    auto src = ins.get_src_int();
    auto dst = ins.get_dst_int();
    auto imm = ins.get_imm();
    uint32_t addr = static_cast<uint32_t>(regs_.read_riscgpr(src[0]) + imm[0]);
    regs_.write_riscgpr(dst[0], do_load(addr, 16, /*sign_extend=*/true));
    return static_cast<int>(ins.get_addr()) + 4;
}

int TriscFunc::exec_lb(isa::Instruction& ins)
{
    auto src = ins.get_src_int();
    auto dst = ins.get_dst_int();
    auto imm = ins.get_imm();
    uint32_t addr = static_cast<uint32_t>(regs_.read_riscgpr(src[0]) + imm[0]);
    regs_.write_riscgpr(dst[0], do_load(addr, 8, /*sign_extend=*/true));
    return static_cast<int>(ins.get_addr()) + 4;
}

int TriscFunc::exec_lhu(isa::Instruction& ins)
{
    auto src = ins.get_src_int();
    auto dst = ins.get_dst_int();
    auto imm = ins.get_imm();
    uint32_t addr = static_cast<uint32_t>(regs_.read_riscgpr(src[0]) + imm[0]);
    regs_.write_riscgpr(dst[0], do_load(addr, 16, /*sign_extend=*/false));
    return static_cast<int>(ins.get_addr()) + 4;
}

int TriscFunc::exec_lbu(isa::Instruction& ins)
{
    auto src = ins.get_src_int();
    auto dst = ins.get_dst_int();
    auto imm = ins.get_imm();
    uint32_t addr = static_cast<uint32_t>(regs_.read_riscgpr(src[0]) + imm[0]);
    regs_.write_riscgpr(dst[0], do_load(addr, 8, /*sign_extend=*/false));
    return static_cast<int>(ins.get_addr()) + 4;
}

// ---------------------------------------------------------------------------
// Store handlers
// ---------------------------------------------------------------------------

int TriscFunc::exec_sw(isa::Instruction& ins, int cycle)
{
    auto src = ins.get_src_int();
    auto imm = ins.get_imm();
    uint32_t addr = static_cast<uint32_t>(regs_.read_riscgpr(src[0]) + imm[0]);
    int32_t  val  = regs_.read_riscgpr(src[1]);
    do_store(addr, val, 32, ins, cycle);
    return static_cast<int>(ins.get_addr()) + 4;
}

int TriscFunc::exec_sh(isa::Instruction& ins)
{
    auto src = ins.get_src_int();
    auto imm = ins.get_imm();
    uint32_t addr = static_cast<uint32_t>(regs_.read_riscgpr(src[0]) + imm[0]);
    int32_t  val  = regs_.read_riscgpr(src[1]);
    do_store(addr, val, 16, ins, /*cycle=*/0);
    return static_cast<int>(ins.get_addr()) + 4;
}

int TriscFunc::exec_sb(isa::Instruction& ins)
{
    auto src = ins.get_src_int();
    auto imm = ins.get_imm();
    uint32_t addr = static_cast<uint32_t>(regs_.read_riscgpr(src[0]) + imm[0]);
    int32_t  val  = regs_.read_riscgpr(src[1]);
    do_store(addr, val, 8, ins, /*cycle=*/0);
    return static_cast<int>(ins.get_addr()) + 4;
}

// ---------------------------------------------------------------------------
// Branch handlers
// ---------------------------------------------------------------------------

int TriscFunc::exec_beq(isa::Instruction& ins)
{
    auto src = ins.get_src_int();
    auto imm = ins.get_imm();
    int32_t a = regs_.read_riscgpr(src[0]);
    int32_t b = regs_.read_riscgpr(src[1]);
    int base  = static_cast<int>(ins.get_addr());
    return (a == b) ? (base + imm[0]) : (base + 4);
}

int TriscFunc::exec_bne(isa::Instruction& ins)
{
    auto src = ins.get_src_int();
    auto imm = ins.get_imm();
    int32_t a = regs_.read_riscgpr(src[0]);
    int32_t b = regs_.read_riscgpr(src[1]);
    int base  = static_cast<int>(ins.get_addr());
    return (a != b) ? (base + imm[0]) : (base + 4);
}

int TriscFunc::exec_blt(isa::Instruction& ins)
{
    auto src = ins.get_src_int();
    auto imm = ins.get_imm();
    int32_t a = regs_.read_riscgpr(src[0]);
    int32_t b = regs_.read_riscgpr(src[1]);
    int base  = static_cast<int>(ins.get_addr());
    return (a < b) ? (base + imm[0]) : (base + 4);
}

int TriscFunc::exec_bge(isa::Instruction& ins)
{
    auto src = ins.get_src_int();
    auto imm = ins.get_imm();
    int32_t a = regs_.read_riscgpr(src[0]);
    int32_t b = regs_.read_riscgpr(src[1]);
    int base  = static_cast<int>(ins.get_addr());
    return (a >= b) ? (base + imm[0]) : (base + 4);
}

int TriscFunc::exec_bltu(isa::Instruction& ins)
{
    auto src = ins.get_src_int();
    auto imm = ins.get_imm();
    uint32_t a = static_cast<uint32_t>(regs_.read_riscgpr(src[0]));
    uint32_t b = static_cast<uint32_t>(regs_.read_riscgpr(src[1]));
    int base   = static_cast<int>(ins.get_addr());
    return (a < b) ? (base + imm[0]) : (base + 4);
}

int TriscFunc::exec_bgeu(isa::Instruction& ins)
{
    auto src = ins.get_src_int();
    auto imm = ins.get_imm();
    uint32_t a = static_cast<uint32_t>(regs_.read_riscgpr(src[0]));
    uint32_t b = static_cast<uint32_t>(regs_.read_riscgpr(src[1]));
    int base   = static_cast<int>(ins.get_addr());
    return (a >= b) ? (base + imm[0]) : (base + 4);
}

// ---------------------------------------------------------------------------
// Jump / upper-immediate / other handlers
// ---------------------------------------------------------------------------

int TriscFunc::exec_jal(isa::Instruction& ins)
{
    auto dst = ins.get_dst_int();
    auto imm = ins.get_imm();
    int pc = static_cast<int>(ins.get_addr());

    if (dst[0] != 0) {
        regs_.write_riscgpr(dst[0], pc + 4);
        return pc + imm[0];
    }
    // dst == x0: return address discarded.
    if (imm[0] == 0) {
        // JAL x0, 0 — treat as "end of kernel" (matches Python behaviour).
        return 0;
    }
    return pc + imm[0];
}

int TriscFunc::exec_jalr(isa::Instruction& ins)
{
    auto src = ins.get_src_int();
    auto dst = ins.get_dst_int();
    auto imm = ins.get_imm();
    int pc = static_cast<int>(ins.get_addr());

    // Compute target first (spec: compute before writing rd).
    int target = regs_.read_riscgpr(src[0]) + imm[0];
    if (dst[0] != 0) {
        regs_.write_riscgpr(dst[0], pc + 4);
    }
    return target;
}

int TriscFunc::exec_lui(isa::Instruction& ins)
{
    auto dst = ins.get_dst_int();
    auto imm = ins.get_imm();
    regs_.write_riscgpr(dst[0], imm[0] << 12);
    return static_cast<int>(ins.get_addr()) + 4;
}

int TriscFunc::exec_auipc(isa::Instruction& ins)
{
    auto dst = ins.get_dst_int();
    auto imm = ins.get_imm();
    int pc = static_cast<int>(ins.get_addr());
    regs_.write_riscgpr(dst[0], pc + (imm[0] << 12));
    return pc + 4;
}

int TriscFunc::exec_fence(isa::Instruction& ins)
{
    // No-op in this functional model (memory ordering not modelled).
    return static_cast<int>(ins.get_addr()) + 4;
}

int TriscFunc::exec_csr(isa::Instruction& ins)
{
    auto src  = ins.get_src_int();
    auto dst  = ins.get_dst_int();
    auto attr = ins.get_attr();

    int csr_idx = attr.at("csr");

    // Read old CSR value into destination GPR (unless rd == x0).
    if (dst[0] != 0) {
        regs_.write_riscgpr(dst[0], regs_.read_csr(csr_idx));
    }

    const std::string& op = ins.get_op();
    if (op == "CSRRW") {
        regs_.write_csr(csr_idx, regs_.read_riscgpr(src[0]));
    } else if (op == "CSRRS") {
        // Set bits in CSR: csr |= rs1  (Python ported faithfully: uses & with src[0] as index,
        // but we implement the correct RISC-V semantics since src[0] is a register index).
        regs_.write_csr(csr_idx, regs_.read_csr(csr_idx) | regs_.read_riscgpr(src[0]));
    } else { // CSRRC
        regs_.write_csr(csr_idx, regs_.read_csr(csr_idx) & ~regs_.read_riscgpr(src[0]));
    }

    return static_cast<int>(ins.get_addr()) + 4;
}

// ---------------------------------------------------------------------------
// Store routing helper
// ---------------------------------------------------------------------------

void TriscFunc::do_store(uint32_t addr, int32_t raw_val, int width,
                         isa::Instruction& ins, int /*cycle*/)
{
    using ST  = units::TensixSplReg::SplRegType;
    using TCF = units::TensixSplReg::TileCounterField;

    // Apply width masking (SW: full 32-bit; SH: 16-bit; SB: 8-bit).
    int32_t val = raw_val;
    if (width == 16) val &= 0xFFFF;
    else if (width == 8) val &= 0xFF;

    auto trisc_mmr = regs_.is_mmr(addr);
    auto spl_mmr   = spl_regs_.is_mmr(addr);

    if (trisc_mmr.type == TriscRegs::RegType::NONE && spl_mmr.type == ST::NONE) {
        // ── Plain memory ────────────────────────────────────────────────
        if (width == 32) {
            mem_.write_mem(addr, val);
        } else if (width == 16) {
            int32_t old = mem_.read_mem(addr);
            mem_.write_mem(addr, (old & static_cast<int32_t>(0xFFFF0000)) | (val & 0xFFFF));
        } else { // 8
            int32_t old = mem_.read_mem(addr);
            mem_.write_mem(addr, (old & static_cast<int32_t>(0xFFFFFF00)) | (val & 0xFF));
        }

    } else if (spl_mmr.type == ST::CFG) {
        // ── Cfg register write (with instruction-state side-effects) ────
        spl_regs_.write_reg(spl_mmr.offset, val, ST::CFG);
        apply_cfg_write_effects(spl_mmr.offset, val, ins);

    } else if (spl_mmr.type == ST::MOP) {
        // MOP: absolute index = thread * MOP_PER_THREAD + local_offset
        int idx = cfg_.thread_id * units::TensixSplReg::MOP_PER_THREAD + spl_mmr.offset;
        spl_regs_.write_reg(idx, val, ST::MOP);

    } else if (spl_mmr.type == ST::MOP_SYNC) {
        // Only valid for SW (width==32); assertion captures SH/SB misuse.
        assert(width == 32 && "MOP_SYNC write via SH/SB not supported");
        spl_regs_.write_reg(cfg_.thread_id + spl_mmr.offset, val, ST::MOP_SYNC);

    } else if (spl_mmr.type == ST::IDLE_SYNC) {
        assert(width == 32 && "IDLE_SYNC write via SH/SB not supported");
        spl_regs_.write_reg(cfg_.thread_id + spl_mmr.offset, val, ST::IDLE_SYNC);

    } else if (spl_mmr.type == ST::INSTR_BUF) {
        assert(spl_mmr.offset == 0);
        spl_regs_.write_reg(0, val, ST::INSTR_BUF);

    } else if (spl_mmr.type == ST::TILE_COUNTERS && width == 32) {
        // ── Tile counter write (only for SW) ────────────────────────────
        int buf_idx   = spl_mmr.offset / 8;
        int field_idx = spl_mmr.offset % 8;
        switch (field_idx) {
            case static_cast<int>(TCF::RESET): {
                spl_regs_.write_tile_counter(buf_idx, TCF::RESET, val);
                spl_regs_.write_tile_counter(buf_idx, TCF::TILES_AVAILABLE, 0);
                int32_t cap = spl_regs_.read_tile_counter(buf_idx, TCF::BUFFER_CAPACITY);
                int32_t avail = spl_regs_.read_tile_counter(buf_idx, TCF::TILES_AVAILABLE);
                spl_regs_.write_tile_counter(buf_idx, TCF::SPACE_AVAILABLE, cap - avail);
                break;
            }
            case static_cast<int>(TCF::TILES_AVAILABLE): {
                spl_regs_.write_tile_counter(buf_idx, TCF::TILES_AVAILABLE, val);
                int32_t cap   = spl_regs_.read_tile_counter(buf_idx, TCF::BUFFER_CAPACITY);
                int32_t avail = spl_regs_.read_tile_counter(buf_idx, TCF::TILES_AVAILABLE);
                spl_regs_.write_tile_counter(buf_idx, TCF::SPACE_AVAILABLE, cap - avail);
                break;
            }
            case static_cast<int>(TCF::SPACE_AVAILABLE):
                spl_regs_.write_tile_counter(buf_idx, TCF::SPACE_AVAILABLE, val);
                break;
            case static_cast<int>(TCF::BUFFER_CAPACITY):
                spl_regs_.write_tile_counter(buf_idx, TCF::BUFFER_CAPACITY, val);
                break;
            case static_cast<int>(TCF::TILES_AVAIL_IRQ_THRESH):
                spl_regs_.write_tile_counter(buf_idx, TCF::TILES_AVAIL_IRQ_THRESH, val);
                break;
            case static_cast<int>(TCF::SPACE_AVAIL_IRQ_THRESH):
                spl_regs_.write_tile_counter(buf_idx, TCF::SPACE_AVAIL_IRQ_THRESH, val);
                break;
            default:
                assert(false && "Unhandled tileCounters sub-field index");
        }

    } else if (trisc_mmr.type != TriscRegs::RegType::NONE) {
        // ── TRISC-local MMR (riscgpr / csr / trisc_id) ──────────────────
        regs_.write_reg(trisc_mmr.offset, val, trisc_mmr.type);
    }
    // else: unrecognised MMR combination — silently ignored (matches Python warning path)
}

// ---------------------------------------------------------------------------
// Load routing helper
// ---------------------------------------------------------------------------

int32_t TriscFunc::do_load(uint32_t addr, int width, bool sign_extend)
{
    using ST  = units::TensixSplReg::SplRegType;
    using TCF = units::TensixSplReg::TileCounterField;

    auto trisc_mmr = regs_.is_mmr(addr);
    auto spl_mmr   = spl_regs_.is_mmr(addr);

    int32_t raw = 0;

    if (trisc_mmr.type == TriscRegs::RegType::NONE && spl_mmr.type == ST::NONE) {
        raw = mem_.read_mem(addr);

    } else if (spl_mmr.type == ST::CFG) {
        raw = spl_regs_.read_reg(spl_mmr.offset, ST::CFG);

    } else if (spl_mmr.type == ST::MOP) {
        int idx = cfg_.thread_id * units::TensixSplReg::MOP_PER_THREAD + spl_mmr.offset;
        raw = spl_regs_.read_reg(idx, ST::MOP);

    } else if (spl_mmr.type == ST::MOP_SYNC) {
        raw = spl_regs_.read_reg(cfg_.thread_id + spl_mmr.offset, ST::MOP_SYNC);

    } else if (spl_mmr.type == ST::IDLE_SYNC) {
        raw = spl_regs_.read_reg(cfg_.thread_id + spl_mmr.offset, ST::IDLE_SYNC);

    } else if (spl_mmr.type == ST::INSTR_BUF) {
        raw = spl_regs_.read_reg(spl_mmr.offset, ST::INSTR_BUF);

    } else if (spl_mmr.type == ST::SEMAPHORES) {
        raw = spl_regs_.read_reg(spl_mmr.offset, ST::SEMAPHORES);

    } else if (spl_mmr.type == ST::TILE_COUNTERS) {
        int buf_idx   = spl_mmr.offset / 8;
        int field_idx = spl_mmr.offset % 8;
        switch (field_idx) {
            case static_cast<int>(TCF::RESET):
                raw = spl_regs_.read_tile_counter(buf_idx, TCF::RESET); break;
            case static_cast<int>(TCF::TILES_AVAILABLE):
                raw = spl_regs_.read_tile_counter(buf_idx, TCF::TILES_AVAILABLE); break;
            case static_cast<int>(TCF::SPACE_AVAILABLE):
                raw = spl_regs_.read_tile_counter(buf_idx, TCF::SPACE_AVAILABLE); break;
            case static_cast<int>(TCF::BUFFER_CAPACITY):
                raw = spl_regs_.read_tile_counter(buf_idx, TCF::BUFFER_CAPACITY); break;
            case static_cast<int>(TCF::TILES_AVAIL_IRQ_THRESH):
                raw = spl_regs_.read_tile_counter(buf_idx, TCF::TILES_AVAIL_IRQ_THRESH); break;
            case static_cast<int>(TCF::SPACE_AVAIL_IRQ_THRESH):
                raw = spl_regs_.read_tile_counter(buf_idx, TCF::SPACE_AVAIL_IRQ_THRESH); break;
            default:
                assert(false && "Unhandled tileCounters sub-field index");
        }

    } else {
        // TRISC-local MMR
        raw = regs_.read_reg(trisc_mmr.offset, trisc_mmr.type);
    }

    // Apply width masking / sign extension.
    if (width == 32) return raw;
    if (width == 16) {
        if (sign_extend)
            return (raw & 0x8000) ? (raw | static_cast<int32_t>(0xFFFF0000u)) : (raw & 0xFFFF);
        return raw & 0xFFFF;
    }
    // width == 8
    if (sign_extend)
        return (raw & 0x80) ? (raw | static_cast<int32_t>(0xFFFFFF00u)) : (raw & 0xFF);
    return raw & 0xFF;
}

// ---------------------------------------------------------------------------
// Cfg-write instruction-state side-effects
// ---------------------------------------------------------------------------

void TriscFunc::apply_cfg_write_effects(int offset, int32_t /*val*/,
                                        isa::Instruction& ins)
{
    using CfgClass = units::TensixSplReg::CfgRegUpdateClass;
    int dst_reg_id = static_cast<int>(isa::RegIndex::DST);

    switch (spl_regs_.get_cfg_reg_update_class(offset)) {
        case CfgClass::DEST_TARGET_REG_CFG_MATH: {
            if (spl_regs_.is_dst_reg_programmed() &&
                spl_regs_.update_dst_reg_bank_id(offset)) {
                ins.set_vld_upd_mask({{dst_reg_id, 1}});
                ins.set_bank_upd_mask({{dst_reg_id, 1}});
            }
            break;
        }
        case CfgClass::DEST_DVALID_CTRL: {
            auto cvi = spl_regs_.get_dst_reg_cond_valids(offset);
            const int ignore = static_cast<int>(isa::ValueStatus::IGNORE);
            std::map<int, std::map<int, int>> chk, wri;
            for (int ctx = 0; ctx < 4; ++ctx) {
                chk[ctx] = {{0, ignore}, {1, ignore}, {2, ignore}, {3, ignore}};
                wri[ctx] = chk[ctx];
            }
            chk[cvi.context_id][dst_reg_id] = cvi.read_mask;
            wri[cvi.context_id][dst_reg_id] = cvi.write_mask;
            ins.set_cond_chk_vld_upd(chk);
            ins.set_cond_wri_vld_upd(wri);
            break;
        }
        default:
            break;
    }
}

} // namespace neosim::risc
