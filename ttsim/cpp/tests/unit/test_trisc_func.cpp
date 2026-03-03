// SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Unit tests for TriscFunc (Track 5a).

#include "neosim/risc/trisc_func.hpp"
#include "neosim/risc/trisc_mem_func.hpp"
#include "neosim/risc/trisc_regs.hpp"
#include "neosim/units/tensix_spl_reg.hpp"
#include "neosim/isa/instruction.hpp"

#include <gtest/gtest.h>
#include <cstdint>
#include <string>
#include <vector>
#include <map>

using namespace neosim;
using namespace neosim::risc;
using namespace neosim::units;
using namespace neosim::isa;

// ---------------------------------------------------------------------------
// Test fixture
// ---------------------------------------------------------------------------

class TriscFuncTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        // ── TensixSplReg ──────────────────────────────────────────────
        TensixSplReg::Config sc;
        sc.core_id         = 0;
        sc.cfg_start       = 0x1000;
        sc.cfg_end         = 0x2000;
        sc.mop_start       = 0x10000;
        sc.pcbuf_start     = 0x20000;
        sc.tile_cnt_start  = 0x30000;
        sc.tile_cnt_end    = 0x31000;
        spl_regs_ = std::make_unique<TensixSplReg>(sc);

        // ── TriscRegs ─────────────────────────────────────────────────
        TriscRegs::Config rc;
        rc.core_id    = 0;
        rc.thread_id  = 0;
        rc.stack_ptr  = 0x8000;
        rc.global_ptr = 0x4000;
        regs_ = std::make_unique<TriscRegs>(rc);

        // ── TriscMemFunc ──────────────────────────────────────────────
        TriscMemFunc::Config mc;
        mc.arch = "ttqs";
        mem_ = std::make_unique<TriscMemFunc>(mc);

        // ── TriscFunc ─────────────────────────────────────────────────
        TriscFunc::Config fc;
        fc.core_id   = 0;
        fc.thread_id = 0;
        func_ = std::make_unique<TriscFunc>(fc, *mem_, *spl_regs_, *regs_);
    }

    /// Build a minimal Instruction with the given opcode, src registers, dst
    /// register, and optional immediate.  PC is set to @p pc.
    Instruction make_ins(const std::string& op,
                         std::vector<int> src,
                         std::vector<int> dst,
                         std::vector<int> imm = {},
                         uint32_t pc = 0x1000u)
    {
        Instruction ins;
        ins.set_op(op);
        ins.program_counter = pc;  // decoded_instruction::program_counter
        if (!src.empty())  ins.set_src_int(src);
        if (!dst.empty())  ins.set_dst_int(dst);
        if (!imm.empty())  ins.set_imm(imm);
        return ins;
    }

    std::unique_ptr<TensixSplReg>  spl_regs_;
    std::unique_ptr<TriscRegs>     regs_;
    std::unique_ptr<TriscMemFunc>  mem_;
    std::unique_ptr<TriscFunc>     func_;
};

// ---------------------------------------------------------------------------
// TriscMemFunc tests
// ---------------------------------------------------------------------------

TEST_F(TriscFuncTest, TriscMemFunc_WriteRead)
{
    mem_->write_mem(0xDEADBEEFu, 0x12345678);
    EXPECT_EQ(mem_->read_mem(0xDEADBEEFu), 0x12345678);
}

TEST_F(TriscFuncTest, TriscMemFunc_IsInitialized)
{
    EXPECT_FALSE(mem_->is_initialized(0xAAAA0000u));
    mem_->write_mem(0xAAAA0000u, 99);
    EXPECT_TRUE(mem_->is_initialized(0xAAAA0000u));
}

TEST_F(TriscFuncTest, TriscMemFunc_TtqsPreseeded)
{
    // ttqs arch pre-seeds tile-counter region to 0
    EXPECT_TRUE(mem_->is_initialized(0x0080b000u));
    EXPECT_EQ(mem_->read_mem(0x0080b000u), 0);
}

// ---------------------------------------------------------------------------
// R-type arithmetic
// ---------------------------------------------------------------------------

TEST_F(TriscFuncTest, ExecRIns_ADD)
{
    regs_->write_riscgpr(10, 5);
    regs_->write_riscgpr(11, 3);
    auto ins = make_ins("ADD", {10, 11}, {12});
    int next = func_->exec_r_ins(ins, 0);
    EXPECT_EQ(next, 0x1004);
    EXPECT_EQ(regs_->read_riscgpr(12), 8);
}

TEST_F(TriscFuncTest, ExecRIns_SUB)
{
    regs_->write_riscgpr(10, 10);
    regs_->write_riscgpr(11, 4);
    auto ins = make_ins("SUB", {10, 11}, {12});
    func_->exec_r_ins(ins, 0);
    EXPECT_EQ(regs_->read_riscgpr(12), 6);
}

TEST_F(TriscFuncTest, ExecRIns_AND)
{
    regs_->write_riscgpr(10, 0b1100);
    regs_->write_riscgpr(11, 0b1010);
    auto ins = make_ins("AND", {10, 11}, {12});
    func_->exec_r_ins(ins, 0);
    EXPECT_EQ(regs_->read_riscgpr(12), 0b1000);
}

TEST_F(TriscFuncTest, ExecRIns_OR)
{
    regs_->write_riscgpr(10, 0b1100);
    regs_->write_riscgpr(11, 0b0011);
    auto ins = make_ins("OR", {10, 11}, {12});
    func_->exec_r_ins(ins, 0);
    EXPECT_EQ(regs_->read_riscgpr(12), 0b1111);
}

TEST_F(TriscFuncTest, ExecRIns_XOR)
{
    regs_->write_riscgpr(10, 0b1100);
    regs_->write_riscgpr(11, 0b1010);
    auto ins = make_ins("XOR", {10, 11}, {12});
    func_->exec_r_ins(ins, 0);
    EXPECT_EQ(regs_->read_riscgpr(12), 0b0110);
}

TEST_F(TriscFuncTest, ExecRIns_SLL)
{
    regs_->write_riscgpr(10, 1);
    regs_->write_riscgpr(11, 4);
    auto ins = make_ins("SLL", {10, 11}, {12});
    func_->exec_r_ins(ins, 0);
    EXPECT_EQ(regs_->read_riscgpr(12), 16);
}

TEST_F(TriscFuncTest, ExecRIns_SRL)
{
    regs_->write_riscgpr(10, static_cast<int32_t>(0x80000010u)); // negative int
    regs_->write_riscgpr(11, 4);
    auto ins = make_ins("SRL", {10, 11}, {12});
    func_->exec_r_ins(ins, 0);
    // SRL: logical shift — MSB fills with 0
    EXPECT_EQ(regs_->read_riscgpr(12), static_cast<int32_t>(0x08000001u));
}

TEST_F(TriscFuncTest, ExecRIns_SRA_Negative)
{
    regs_->write_riscgpr(10, static_cast<int32_t>(0x80000010u)); // negative
    regs_->write_riscgpr(11, 4);
    auto ins = make_ins("SRA", {10, 11}, {12});
    func_->exec_r_ins(ins, 0);
    // SRA: arithmetic shift — MSB fills with 1
    EXPECT_EQ(regs_->read_riscgpr(12), static_cast<int32_t>(0xF8000001u));
}

TEST_F(TriscFuncTest, ExecRIns_SLT_Signed)
{
    regs_->write_riscgpr(10, static_cast<int32_t>(-1)); // negative in signed
    regs_->write_riscgpr(11, 0);
    auto ins = make_ins("SLT", {10, 11}, {12});
    func_->exec_r_ins(ins, 0);
    EXPECT_EQ(regs_->read_riscgpr(12), 1); // -1 < 0 signed
}

TEST_F(TriscFuncTest, ExecRIns_SLTU_Unsigned)
{
    regs_->write_riscgpr(10, static_cast<int32_t>(-1)); // 0xFFFFFFFF unsigned
    regs_->write_riscgpr(11, 0);
    auto ins = make_ins("SLTU", {10, 11}, {12});
    func_->exec_r_ins(ins, 0);
    EXPECT_EQ(regs_->read_riscgpr(12), 0); // 0xFFFFFFFF > 0 unsigned
}

TEST_F(TriscFuncTest, ExecRIns_MUL)
{
    regs_->write_riscgpr(10, 6);
    regs_->write_riscgpr(11, 7);
    auto ins = make_ins("MUL", {10, 11}, {12});
    func_->exec_r_ins(ins, 0);
    EXPECT_EQ(regs_->read_riscgpr(12), 42);
}

// ---------------------------------------------------------------------------
// I-type arithmetic
// ---------------------------------------------------------------------------

TEST_F(TriscFuncTest, ExecRIns_ADDI)
{
    regs_->write_riscgpr(10, 100);
    auto ins = make_ins("ADDI", {10}, {12}, {-50});
    func_->exec_r_ins(ins, 0);
    EXPECT_EQ(regs_->read_riscgpr(12), 50);
}

TEST_F(TriscFuncTest, ExecRIns_SUBI)
{
    regs_->write_riscgpr(10, 100);
    auto ins = make_ins("SUBI", {10}, {12}, {40});
    func_->exec_r_ins(ins, 0);
    EXPECT_EQ(regs_->read_riscgpr(12), 60);
}

TEST_F(TriscFuncTest, ExecRIns_ANDI)
{
    regs_->write_riscgpr(10, 0xFF);
    auto ins = make_ins("ANDI", {10}, {12}, {0x0F});
    func_->exec_r_ins(ins, 0);
    EXPECT_EQ(regs_->read_riscgpr(12), 0x0F);
}

TEST_F(TriscFuncTest, ExecRIns_ORI)
{
    regs_->write_riscgpr(10, 0xF0);
    auto ins = make_ins("ORI", {10}, {12}, {0x0F});
    func_->exec_r_ins(ins, 0);
    EXPECT_EQ(regs_->read_riscgpr(12), 0xFF);
}

TEST_F(TriscFuncTest, ExecRIns_XORI)
{
    regs_->write_riscgpr(10, 0xFF);
    auto ins = make_ins("XORI", {10}, {12}, {0x0F});
    func_->exec_r_ins(ins, 0);
    EXPECT_EQ(regs_->read_riscgpr(12), 0xF0);
}

TEST_F(TriscFuncTest, ExecRIns_SLLI)
{
    regs_->write_riscgpr(10, 1);
    auto ins = make_ins("SLLI", {10}, {12}, {3});
    func_->exec_r_ins(ins, 0);
    EXPECT_EQ(regs_->read_riscgpr(12), 8);
}

TEST_F(TriscFuncTest, ExecRIns_SRLI)
{
    regs_->write_riscgpr(10, 0x10);
    auto ins = make_ins("SRLI", {10}, {12}, {2});
    func_->exec_r_ins(ins, 0);
    EXPECT_EQ(regs_->read_riscgpr(12), 4);
}

TEST_F(TriscFuncTest, ExecRIns_SRAI_Negative)
{
    regs_->write_riscgpr(10, static_cast<int32_t>(0x80000000u));
    auto ins = make_ins("SRAI", {10}, {12}, {1});
    func_->exec_r_ins(ins, 0);
    EXPECT_EQ(regs_->read_riscgpr(12), static_cast<int32_t>(0xC0000000u));
}

TEST_F(TriscFuncTest, ExecRIns_SLTI)
{
    regs_->write_riscgpr(10, static_cast<int32_t>(-5));
    auto ins = make_ins("SLTI", {10}, {12}, {0});
    func_->exec_r_ins(ins, 0);
    EXPECT_EQ(regs_->read_riscgpr(12), 1);
}

TEST_F(TriscFuncTest, ExecRIns_SLTIU)
{
    regs_->write_riscgpr(10, 3);
    auto ins = make_ins("SLTIU", {10}, {12}, {10});
    func_->exec_r_ins(ins, 0);
    EXPECT_EQ(regs_->read_riscgpr(12), 1);
}

// ---------------------------------------------------------------------------
// Load / store — plain memory
// ---------------------------------------------------------------------------

TEST_F(TriscFuncTest, ExecRIns_SW_Memory)
{
    // Write sentinel to memory then read it back via LW.
    regs_->write_riscgpr(5, 0x00C0000);  // base
    regs_->write_riscgpr(6, 0xABCD);      // data
    auto sw = make_ins("SW", {5, 6}, {}, {0});
    func_->exec_r_ins(sw, 0);
    EXPECT_TRUE(mem_->is_initialized(0x00C0000u));
    EXPECT_EQ(mem_->read_mem(0x00C0000u), 0xABCD);
}

TEST_F(TriscFuncTest, ExecRIns_LW_Memory)
{
    mem_->write_mem(0x00D0000u, 0x12345678);
    regs_->write_riscgpr(5, static_cast<int32_t>(0x00D0000u));
    auto lw = make_ins("LW", {5}, {10}, {0});
    func_->exec_r_ins(lw, 0);
    EXPECT_EQ(regs_->read_riscgpr(10), 0x12345678);
}

TEST_F(TriscFuncTest, ExecRIns_SH_Memory)
{
    // Write a known 32-bit word, then SH should merge the low 16 bits.
    mem_->write_mem(0x00E0000u, static_cast<int32_t>(0xFFFF0000u));
    regs_->write_riscgpr(5, static_cast<int32_t>(0x00E0000u));
    regs_->write_riscgpr(6, 0x1234ABCD); // only low 16 bits stored
    auto sh = make_ins("SH", {5, 6}, {}, {0});
    func_->exec_r_ins(sh, 0);
    EXPECT_EQ(mem_->read_mem(0x00E0000u), static_cast<int32_t>(0xFFFFABCDu));
}

TEST_F(TriscFuncTest, ExecRIns_SB_Memory)
{
    mem_->write_mem(0x00E0004u, static_cast<int32_t>(0xFFFFFF00u));
    regs_->write_riscgpr(5, static_cast<int32_t>(0x00E0004u));
    regs_->write_riscgpr(6, 0x1234ABCD); // only low 8 bits stored
    auto sb = make_ins("SB", {5, 6}, {}, {0});
    func_->exec_r_ins(sb, 0);
    EXPECT_EQ(mem_->read_mem(0x00E0004u), static_cast<int32_t>(0xFFFFFFCDu));
}

TEST_F(TriscFuncTest, ExecRIns_LH_SignExtend)
{
    // Pre-seed a 32-bit word whose low 16 bits are 0xFFFF (sign-extends to -1)
    mem_->write_mem(0x00F000u, 0x0000FFFF);
    regs_->write_riscgpr(5, static_cast<int32_t>(0x00F000u));
    auto lh = make_ins("LH", {5}, {10}, {0});
    func_->exec_r_ins(lh, 0);
    EXPECT_EQ(regs_->read_riscgpr(10), static_cast<int32_t>(0xFFFFFFFFu));
}

TEST_F(TriscFuncTest, ExecRIns_LB_SignExtend)
{
    mem_->write_mem(0x00F004u, 0x000000FF);
    regs_->write_riscgpr(5, static_cast<int32_t>(0x00F004u));
    auto lb = make_ins("LB", {5}, {10}, {0});
    func_->exec_r_ins(lb, 0);
    EXPECT_EQ(regs_->read_riscgpr(10), static_cast<int32_t>(0xFFFFFFFFu));
}

TEST_F(TriscFuncTest, ExecRIns_LHU_ZeroExtend)
{
    mem_->write_mem(0x00F008u, 0x0000FFFF);
    regs_->write_riscgpr(5, static_cast<int32_t>(0x00F008u));
    auto lhu = make_ins("LHU", {5}, {10}, {0});
    func_->exec_r_ins(lhu, 0);
    EXPECT_EQ(regs_->read_riscgpr(10), 0xFFFF); // zero-extended
}

TEST_F(TriscFuncTest, ExecRIns_LBU_ZeroExtend)
{
    mem_->write_mem(0x00F00Cu, 0x000000FF);
    regs_->write_riscgpr(5, static_cast<int32_t>(0x00F00Cu));
    auto lbu = make_ins("LBU", {5}, {10}, {0});
    func_->exec_r_ins(lbu, 0);
    EXPECT_EQ(regs_->read_riscgpr(10), 0xFF); // zero-extended
}

// ---------------------------------------------------------------------------
// Load / store — TensixSplReg MMR (MOP registers)
// ---------------------------------------------------------------------------

TEST_F(TriscFuncTest, ExecRIns_SW_MopReg)
{
    // mop_start = 0x10000; thread 0 entry 0 → write there
    uint32_t mop_addr = 0x10000u;
    regs_->write_riscgpr(5, static_cast<int32_t>(mop_addr));
    regs_->write_riscgpr(6, 0x5A5A5A5A);
    auto sw = make_ins("SW", {5, 6}, {}, {0});
    func_->exec_r_ins(sw, 0);
    // Thread 0, offset 0 → spl_regs MOP[0]
    EXPECT_EQ(spl_regs_->read_reg(0, TensixSplReg::SplRegType::MOP), 0x5A5A5A5A);
}

TEST_F(TriscFuncTest, ExecRIns_LW_MopReg)
{
    spl_regs_->write_reg(0, static_cast<int32_t>(0xDEADBEEFu), TensixSplReg::SplRegType::MOP);
    uint32_t mop_addr = 0x10000u;
    regs_->write_riscgpr(5, static_cast<int32_t>(mop_addr));
    auto lw = make_ins("LW", {5}, {10}, {0});
    func_->exec_r_ins(lw, 0);
    EXPECT_EQ(regs_->read_riscgpr(10), static_cast<int32_t>(0xDEADBEEFu));
}

// ---------------------------------------------------------------------------
// Branch
// ---------------------------------------------------------------------------

TEST_F(TriscFuncTest, ExecRIns_BEQ_Taken)
{
    regs_->write_riscgpr(10, 42);
    regs_->write_riscgpr(11, 42);
    auto ins = make_ins("BEQ", {10, 11}, {}, {100}, 0x2000u);
    int next = func_->exec_r_ins(ins, 0);
    EXPECT_EQ(next, 0x2000 + 100);
}

TEST_F(TriscFuncTest, ExecRIns_BEQ_NotTaken)
{
    regs_->write_riscgpr(10, 1);
    regs_->write_riscgpr(11, 2);
    auto ins = make_ins("BEQ", {10, 11}, {}, {100}, 0x2000u);
    int next = func_->exec_r_ins(ins, 0);
    EXPECT_EQ(next, 0x2004);
}

TEST_F(TriscFuncTest, ExecRIns_BNE_Taken)
{
    regs_->write_riscgpr(10, 1);
    regs_->write_riscgpr(11, 2);
    auto ins = make_ins("BNE", {10, 11}, {}, {-8}, 0x2000u);
    int next = func_->exec_r_ins(ins, 0);
    EXPECT_EQ(next, 0x2000 - 8);
}

TEST_F(TriscFuncTest, ExecRIns_BLT_Taken)
{
    regs_->write_riscgpr(10, static_cast<int32_t>(-1));
    regs_->write_riscgpr(11, 0);
    auto ins = make_ins("BLT", {10, 11}, {}, {20}, 0x1000u);
    EXPECT_EQ(func_->exec_r_ins(ins, 0), 0x1014);
}

TEST_F(TriscFuncTest, ExecRIns_BGE_Taken)
{
    regs_->write_riscgpr(10, 5);
    regs_->write_riscgpr(11, 5);
    auto ins = make_ins("BGE", {10, 11}, {}, {8}, 0x1000u);
    EXPECT_EQ(func_->exec_r_ins(ins, 0), 0x1008);
}

TEST_F(TriscFuncTest, ExecRIns_BLTU_Taken)
{
    regs_->write_riscgpr(10, 1);
    regs_->write_riscgpr(11, static_cast<int32_t>(0xFFFFFFFFu)); // large unsigned
    auto ins = make_ins("BLTU", {10, 11}, {}, {12}, 0x1000u);
    EXPECT_EQ(func_->exec_r_ins(ins, 0), 0x100C);
}

TEST_F(TriscFuncTest, ExecRIns_BGEU_Taken)
{
    regs_->write_riscgpr(10, static_cast<int32_t>(0xFFFFFFFFu));
    regs_->write_riscgpr(11, 1);
    auto ins = make_ins("BGEU", {10, 11}, {}, {8}, 0x1000u);
    EXPECT_EQ(func_->exec_r_ins(ins, 0), 0x1008);
}

// ---------------------------------------------------------------------------
// Jump / upper immediate
// ---------------------------------------------------------------------------

TEST_F(TriscFuncTest, ExecRIns_JAL_SaveReturn)
{
    auto ins = make_ins("JAL", {}, {10}, {100}, 0x3000u);
    int next = func_->exec_r_ins(ins, 0);
    EXPECT_EQ(next, 0x3000 + 100);
    EXPECT_EQ(regs_->read_riscgpr(10), 0x3004); // return address
}

TEST_F(TriscFuncTest, ExecRIns_JAL_X0_EndOfKernel)
{
    // JAL x0, 0 → end-of-kernel sentinel
    auto ins = make_ins("JAL", {}, {0}, {0}, 0x3000u);
    int next = func_->exec_r_ins(ins, 0);
    EXPECT_EQ(next, 0); // end-of-kernel
}

TEST_F(TriscFuncTest, ExecRIns_JALR)
{
    regs_->write_riscgpr(5, 0x5000);
    auto ins = make_ins("JALR", {5}, {10}, {0x10}, 0x3000u);
    int next = func_->exec_r_ins(ins, 0);
    EXPECT_EQ(next, 0x5010);
    EXPECT_EQ(regs_->read_riscgpr(10), 0x3004);
}

TEST_F(TriscFuncTest, ExecRIns_LUI)
{
    auto ins = make_ins("LUI", {}, {10}, {0x12345}, 0x1000u);
    func_->exec_r_ins(ins, 0);
    EXPECT_EQ(regs_->read_riscgpr(10), 0x12345 << 12);
}

TEST_F(TriscFuncTest, ExecRIns_AUIPC)
{
    auto ins = make_ins("AUIPC", {}, {10}, {1}, 0x4000u);
    func_->exec_r_ins(ins, 0);
    EXPECT_EQ(regs_->read_riscgpr(10), static_cast<int32_t>(0x4000u + (1 << 12)));
}

// ---------------------------------------------------------------------------
// FENCE
// ---------------------------------------------------------------------------

TEST_F(TriscFuncTest, ExecRIns_FENCE_Trivial)
{
    regs_->write_riscgpr(5, 0);
    regs_->write_riscgpr(6, 0);
    auto ins = make_ins("FENCE", {5, 6}, {5}, {}, 0x1000u);
    ins.set_attr({{"pred", 0}, {"succ", 0}});
    int next = func_->exec_r_ins(ins, 0);
    EXPECT_EQ(next, 0x1004);
}

// ---------------------------------------------------------------------------
// CSR instructions
// ---------------------------------------------------------------------------

TEST_F(TriscFuncTest, ExecRIns_CSRRW)
{
    regs_->write_riscgpr(10, 0xABCD);
    auto ins = make_ins("CSRRW", {10}, {11}, {}, 0x1000u);
    ins.set_attr({{"csr", 5}});
    func_->exec_r_ins(ins, 0);
    EXPECT_EQ(regs_->read_csr(5), 0xABCD);
}

TEST_F(TriscFuncTest, ExecRIns_CSRRS_SetsOldInDst)
{
    regs_->write_csr(3, 0x1234);
    regs_->write_riscgpr(10, 0);  // mask = 0 → no change to CSR
    auto ins = make_ins("CSRRS", {10}, {11}, {}, 0x1000u);
    ins.set_attr({{"csr", 3}});
    func_->exec_r_ins(ins, 0);
    EXPECT_EQ(regs_->read_riscgpr(11), 0x1234); // old value in rd
}

// ---------------------------------------------------------------------------
// Unknown opcode
// ---------------------------------------------------------------------------

TEST_F(TriscFuncTest, ExecRIns_Unknown_ReturnsPcPlus4)
{
    auto ins = make_ins("UNKNOWNOP", {}, {}, {}, 0x2000u);
    int next = func_->exec_r_ins(ins, 0);
    EXPECT_EQ(next, 0x2004);
}

// ---------------------------------------------------------------------------
// Instruction-buffer helpers
// ---------------------------------------------------------------------------

TEST_F(TriscFuncTest, ReadInstructionBufMem_EmptyReturnsMinusOne)
{
    EXPECT_EQ(func_->read_instruction_buf_mem(), -1);
}

TEST_F(TriscFuncTest, ReadInstructionBufMem_ReturnsAndClears)
{
    spl_regs_->write_reg(0, 0xDEAD, TensixSplReg::SplRegType::INSTR_BUF);
    EXPECT_EQ(func_->read_instruction_buf_mem(), 0xDEAD);
    // Second read should return -1 (cleared).
    EXPECT_EQ(func_->read_instruction_buf_mem(), -1);
}

// ---------------------------------------------------------------------------
// Return-address / PC correctness
// ---------------------------------------------------------------------------

TEST_F(TriscFuncTest, ExecRIns_ReturnsPcPlus4_ForArithmetic)
{
    regs_->write_riscgpr(10, 1);
    regs_->write_riscgpr(11, 2);
    auto ins = make_ins("ADD", {10, 11}, {12}, {}, 0xABCDu);
    EXPECT_EQ(func_->exec_r_ins(ins, 0), static_cast<int>(0xABCDu) + 4);
}
