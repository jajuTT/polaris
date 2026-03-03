// SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include "neosim/risc/trisc_regs.hpp"

using namespace neosim::risc;

static TriscRegs::Config make_config(int thread_id = 0,
                                     uint32_t sp = 0xDEAD0000u,
                                     uint32_t gp = 0x00010000u) {
    return {0, thread_id, sp, gp};
}

class TriscRegsTest : public ::testing::Test {
protected:
    TriscRegs regs_{make_config(1, 0xDEAD0000u, 0x00010000u)};
};

// ----------------------------------------------------------------
// Construction / initial values
// ----------------------------------------------------------------

TEST_F(TriscRegsTest, X0_IsZero) {
    EXPECT_EQ(regs_.read_riscgpr(0), 0);
}

TEST_F(TriscRegsTest, X2_IsStackPointer) {
    EXPECT_EQ(regs_.read_riscgpr(2), static_cast<int32_t>(0xDEAD0000u));
}

TEST_F(TriscRegsTest, X3_IsGlobalPointer) {
    EXPECT_EQ(regs_.read_riscgpr(3), static_cast<int32_t>(0x00010000u));
}

TEST_F(TriscRegsTest, TriscId_IsThreadId) {
    EXPECT_EQ(regs_.read_trisc_id(), 1);
}

TEST_F(TriscRegsTest, CoreAndThread) {
    EXPECT_EQ(regs_.core_id(), 0);
    EXPECT_EQ(regs_.thread_id(), 1);
}

// ----------------------------------------------------------------
// riscgpr: non-temporary auto-zero on first read
// ----------------------------------------------------------------

TEST_F(TriscRegsTest, NonTempReg_AutoZeroOnFirstRead) {
    // x4 is not in TEMP_REGS and starts as -1 sentinel
    int32_t val = regs_.read_riscgpr(4);
    EXPECT_EQ(val, 0);
}

TEST_F(TriscRegsTest, NonTempReg_KeepsValueAfterWrite) {
    regs_.write_riscgpr(10, 42);
    EXPECT_EQ(regs_.read_riscgpr(10), 42);
}

// ----------------------------------------------------------------
// riscgpr: temporary regs must be initialised before read
// ----------------------------------------------------------------

TEST_F(TriscRegsTest, TempReg_X5_Uninitialised_Asserts) {
#ifndef NDEBUG
    // x5 is in TEMP_REGS; reading it uninitialised should assert
    EXPECT_DEATH(regs_.read_riscgpr(5), "");
#else
    GTEST_SKIP() << "Assertions disabled in Release build";
#endif
}

TEST_F(TriscRegsTest, TempReg_X5_InitialisedOk) {
    regs_.write_riscgpr(5, 100);
    EXPECT_EQ(regs_.read_riscgpr(5), 100);
}

TEST_F(TriscRegsTest, TempReg_X7_AfterWrite) {
    regs_.write_riscgpr(7, -99);
    EXPECT_EQ(regs_.read_riscgpr(7), -99);
}

// ----------------------------------------------------------------
// riscgpr: extension registers (32–63)
// ----------------------------------------------------------------

TEST_F(TriscRegsTest, ExtensionReg_AutoZeroOnFirstRead) {
    EXPECT_EQ(regs_.read_riscgpr(32), 0); // non-temp, auto-zero
}

TEST_F(TriscRegsTest, ExtensionReg_WriteRead) {
    regs_.write_riscgpr(63, 0xBEEF);
    EXPECT_EQ(regs_.read_riscgpr(63), 0xBEEF);
}

// ----------------------------------------------------------------
// CSR read/write
// ----------------------------------------------------------------

TEST_F(TriscRegsTest, Csr_InitMinus1) {
    EXPECT_EQ(regs_.read_csr(0), -1);
    EXPECT_EQ(regs_.read_csr(4095), -1);
}

TEST_F(TriscRegsTest, Csr_WriteRead) {
    regs_.write_csr(100, 0x1234);
    EXPECT_EQ(regs_.read_csr(100), 0x1234);
}

TEST_F(TriscRegsTest, Csr_AllIndependent) {
    regs_.write_csr(0, 1);
    regs_.write_csr(4095, 2);
    EXPECT_EQ(regs_.read_csr(0), 1);
    EXPECT_EQ(regs_.read_csr(4095), 2);
}

// ----------------------------------------------------------------
// Generic typed accessors
// ----------------------------------------------------------------

TEST_F(TriscRegsTest, ReadWriteReg_Riscgpr) {
    regs_.write_reg(15, 77, TriscRegs::RegType::RISCGPR);
    EXPECT_EQ(regs_.read_reg(15, TriscRegs::RegType::RISCGPR), 77);
}

TEST_F(TriscRegsTest, ReadWriteReg_Csr) {
    regs_.write_reg(200, 0xFACE, TriscRegs::RegType::CSR);
    EXPECT_EQ(regs_.read_reg(200, TriscRegs::RegType::CSR), 0xFACE);
}

TEST_F(TriscRegsTest, ReadReg_TrscId) {
    EXPECT_EQ(regs_.read_reg(0, TriscRegs::RegType::TRISC_ID), 1);
}

// ----------------------------------------------------------------
// is_mmr — address resolution
// ----------------------------------------------------------------

TEST_F(TriscRegsTest, IsMmr_Riscgpr_Base) {
    auto info = regs_.is_mmr(TriscRegs::RISCGPR_BASE);
    EXPECT_EQ(info.type, TriscRegs::RegType::RISCGPR);
    EXPECT_EQ(info.offset, 0);
}

TEST_F(TriscRegsTest, IsMmr_Riscgpr_Mid) {
    auto info = regs_.is_mmr(TriscRegs::RISCGPR_BASE + 8); // offset 8/4 = 2
    EXPECT_EQ(info.type, TriscRegs::RegType::RISCGPR);
    EXPECT_EQ(info.offset, 2);
}

TEST_F(TriscRegsTest, IsMmr_Riscgpr_OutOfRange) {
    auto info = regs_.is_mmr(TriscRegs::RISCGPR_BASE + TriscRegs::RISCGPR_SIZE);
    EXPECT_EQ(info.type, TriscRegs::RegType::NONE);
}

TEST_F(TriscRegsTest, IsMmr_Csr_Base) {
    auto info = regs_.is_mmr(TriscRegs::CSR_BASE);
    EXPECT_EQ(info.type, TriscRegs::RegType::CSR);
    EXPECT_EQ(info.offset, 0);
}

TEST_F(TriscRegsTest, IsMmr_Csr_Mid) {
    auto info = regs_.is_mmr(TriscRegs::CSR_BASE + 16); // offset 16/4 = 4
    EXPECT_EQ(info.type, TriscRegs::RegType::CSR);
    EXPECT_EQ(info.offset, 4);
}

TEST_F(TriscRegsTest, IsMmr_TrscId) {
    auto info = regs_.is_mmr(TriscRegs::TRISC_ID_BASE);
    EXPECT_EQ(info.type, TriscRegs::RegType::TRISC_ID);
    EXPECT_EQ(info.offset, 0);
}

TEST_F(TriscRegsTest, IsMmr_NoMatch) {
    auto info = regs_.is_mmr(0x12345678);
    EXPECT_EQ(info.type, TriscRegs::RegType::NONE);
    EXPECT_EQ(info.offset, -1);
}

// ----------------------------------------------------------------
// Multiple threads independent
// ----------------------------------------------------------------

TEST(TriscRegsMultiThread, TwoThreadsIndependent) {
    TriscRegs t0(make_config(0, 0x1000, 0x2000));
    TriscRegs t1(make_config(1, 0x3000, 0x4000));

    EXPECT_EQ(t0.read_riscgpr(2), static_cast<int32_t>(0x1000));
    EXPECT_EQ(t1.read_riscgpr(2), static_cast<int32_t>(0x3000));
    EXPECT_EQ(t0.read_trisc_id(), 0);
    EXPECT_EQ(t1.read_trisc_id(), 1);
}
