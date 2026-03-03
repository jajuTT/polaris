// SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include "neosim/units/tensix_spl_reg.hpp"

using namespace neosim::units;

// Build a minimal but self-consistent Config for testing.
// Addresses are chosen to be well-separated and easy to reason about.
static TensixSplReg::Config make_test_config() {
    TensixSplReg::Config c;
    c.core_id              = 0;
    c.cfg_start            = 0x1000;
    c.cfg_end              = 0x1100; // 256 bytes / 4 = 64 registers
    c.cfg_bytes_per_reg    = 4;

    c.instr_buf_start         = 0x2000;
    c.instr_buf_bytes_per_reg = 4;

    c.mop_start   = 0x3000;          // 64-byte window [0x3000, 0x3040)
    c.pcbuf_start = 0x4000;
    // idleSync  @ pcbuf + 0x04 = 0x4004
    // mopSync   @ pcbuf + 0x08 = 0x4008
    // semaphores@ pcbuf + 0x80 = 0x4080

    c.tile_cnt_start      = 0x5000;
    c.tile_cnt_end        = 0x51FF; // 512 bytes / 32 bytes per entry = 16 entries
    c.tile_cnt_entry_bytes = 32;

    return c;
}

class TensixSplRegTest : public ::testing::Test {
protected:
    TensixSplReg reg_{make_test_config()};
};

// ----------------------------------------------------------------
// Construction
// ----------------------------------------------------------------

TEST_F(TensixSplRegTest, NumCfgRegs) {
    // (0x1100 - 0x1000) / 4 = 64
    EXPECT_EQ(reg_.num_cfg_regs(), 64);
}

TEST_F(TensixSplRegTest, NumTileCounters) {
    // (0x51FF + 1 - 0x5000) / 32 = 0x200 / 32 = 16
    EXPECT_EQ(reg_.num_tile_counters(), 16);
}

TEST_F(TensixSplRegTest, CfgInitialisedToMinus1) {
    for (int i = 0; i < 64; ++i) {
        EXPECT_EQ(reg_.read_reg(i, TensixSplReg::SplRegType::CFG), -1);
    }
}

TEST_F(TensixSplRegTest, MopInitialisedToMinus1) {
    EXPECT_EQ(reg_.read_reg(0, TensixSplReg::SplRegType::MOP), -1);
}

// ----------------------------------------------------------------
// is_mmr — address resolution
// ----------------------------------------------------------------

TEST_F(TensixSplRegTest, IsMmr_Cfg_FirstReg) {
    auto info = reg_.is_mmr(0x1000);
    EXPECT_EQ(info.type, TensixSplReg::SplRegType::CFG);
    EXPECT_EQ(info.offset, 0);
}

TEST_F(TensixSplRegTest, IsMmr_Cfg_MidReg) {
    auto info = reg_.is_mmr(0x1010); // offset 0x10 / 4 = 4
    EXPECT_EQ(info.type, TensixSplReg::SplRegType::CFG);
    EXPECT_EQ(info.offset, 4);
}

TEST_F(TensixSplRegTest, IsMmr_Cfg_LastByte) {
    auto info = reg_.is_mmr(0x10FC); // last 4-byte register at 0x10FC
    EXPECT_EQ(info.type, TensixSplReg::SplRegType::CFG);
    EXPECT_EQ(info.offset, 63);
}

TEST_F(TensixSplRegTest, IsMmr_Cfg_OutOfRange) {
    auto info = reg_.is_mmr(0x1100); // exclusive upper bound
    EXPECT_EQ(info.type, TensixSplReg::SplRegType::NONE);
}

TEST_F(TensixSplRegTest, IsMmr_InstrBuf) {
    auto info = reg_.is_mmr(0x2000);
    EXPECT_EQ(info.type, TensixSplReg::SplRegType::INSTR_BUF);
    EXPECT_EQ(info.offset, 0);
}

TEST_F(TensixSplRegTest, IsMmr_Mop_Base) {
    auto info = reg_.is_mmr(0x3000);
    EXPECT_EQ(info.type, TensixSplReg::SplRegType::MOP);
    EXPECT_EQ(info.offset, 0);
}

TEST_F(TensixSplRegTest, IsMmr_Mop_Mid) {
    auto info = reg_.is_mmr(0x3008); // offset 8/4 = 2
    EXPECT_EQ(info.type, TensixSplReg::SplRegType::MOP);
    EXPECT_EQ(info.offset, 2);
}

TEST_F(TensixSplRegTest, IsMmr_Mop_OutOfRange) {
    auto info = reg_.is_mmr(0x3040); // 64 bytes window ends at 0x3040
    EXPECT_EQ(info.type, TensixSplReg::SplRegType::NONE);
}

TEST_F(TensixSplRegTest, IsMmr_IdleSync) {
    auto info = reg_.is_mmr(0x4004); // pcbuf + 0x04
    EXPECT_EQ(info.type, TensixSplReg::SplRegType::IDLE_SYNC);
    EXPECT_EQ(info.offset, 0);
}

TEST_F(TensixSplRegTest, IsMmr_MopSync) {
    auto info = reg_.is_mmr(0x4008); // pcbuf + 0x08
    EXPECT_EQ(info.type, TensixSplReg::SplRegType::MOP_SYNC);
    EXPECT_EQ(info.offset, 0);
}

TEST_F(TensixSplRegTest, IsMmr_Semaphores_Base) {
    auto info = reg_.is_mmr(0x4080); // pcbuf + 0x80
    EXPECT_EQ(info.type, TensixSplReg::SplRegType::SEMAPHORES);
    EXPECT_EQ(info.offset, 0);
}

TEST_F(TensixSplRegTest, IsMmr_Semaphores_Mid) {
    auto info = reg_.is_mmr(0x4090); // pcbuf + 0x90 → offset (0x90-0x80)/4 = 4
    EXPECT_EQ(info.type, TensixSplReg::SplRegType::SEMAPHORES);
    EXPECT_EQ(info.offset, 4);
}

TEST_F(TensixSplRegTest, IsMmr_TileCounters) {
    auto info = reg_.is_mmr(0x5000);
    EXPECT_EQ(info.type, TensixSplReg::SplRegType::TILE_COUNTERS);
    EXPECT_EQ(info.offset, 0);
}

TEST_F(TensixSplRegTest, IsMmr_NoMatch) {
    auto info = reg_.is_mmr(0x9999);
    EXPECT_EQ(info.type, TensixSplReg::SplRegType::NONE);
    EXPECT_EQ(info.offset, -1);
}

// ----------------------------------------------------------------
// read / write registers — round trip
// ----------------------------------------------------------------

TEST_F(TensixSplRegTest, CfgReadWrite) {
    reg_.write_reg(5, 0xABCD, TensixSplReg::SplRegType::CFG);
    EXPECT_EQ(reg_.read_reg(5, TensixSplReg::SplRegType::CFG), 0xABCD);
}

TEST_F(TensixSplRegTest, InstrBufReadWrite) {
    reg_.write_reg(0, 0xDEAD, TensixSplReg::SplRegType::INSTR_BUF);
    EXPECT_EQ(reg_.read_reg(0, TensixSplReg::SplRegType::INSTR_BUF), 0xDEAD);
}

TEST_F(TensixSplRegTest, MopReadWrite) {
    reg_.write_reg(10, 0x1234, TensixSplReg::SplRegType::MOP);
    EXPECT_EQ(reg_.read_reg(10, TensixSplReg::SplRegType::MOP), 0x1234);
}

TEST_F(TensixSplRegTest, MopSyncReadWrite) {
    reg_.write_reg(2, 0xFF, TensixSplReg::SplRegType::MOP_SYNC);
    EXPECT_EQ(reg_.read_reg(2, TensixSplReg::SplRegType::MOP_SYNC), 0xFF);
}

TEST_F(TensixSplRegTest, IdleSyncReadWrite) {
    reg_.write_reg(1, 0x01, TensixSplReg::SplRegType::IDLE_SYNC);
    EXPECT_EQ(reg_.read_reg(1, TensixSplReg::SplRegType::IDLE_SYNC), 0x01);
}

TEST_F(TensixSplRegTest, SemaphoresFlatReadWrite) {
    reg_.write_reg(7, 0x42, TensixSplReg::SplRegType::SEMAPHORES);
    EXPECT_EQ(reg_.read_reg(7, TensixSplReg::SplRegType::SEMAPHORES), 0x42);
}

// ----------------------------------------------------------------
// Tile counter sub-field accessors
// ----------------------------------------------------------------

TEST_F(TensixSplRegTest, TileCounterDefaultValues) {
    using TC = TensixSplReg::TileCounterField;
    EXPECT_EQ(reg_.read_tile_counter(0, TC::RESERVED0), -1);
    EXPECT_EQ(reg_.read_tile_counter(0, TC::TILES_AVAILABLE), 0);   // special default
    EXPECT_EQ(reg_.read_tile_counter(0, TC::SPACE_AVAILABLE), 0);   // special default
    EXPECT_EQ(reg_.read_tile_counter(0, TC::BUFFER_CAPACITY), -1);
}

TEST_F(TensixSplRegTest, TileCounterReadWrite) {
    using TC = TensixSplReg::TileCounterField;
    reg_.write_tile_counter(3, TC::TILES_AVAILABLE, 42);
    EXPECT_EQ(reg_.read_tile_counter(3, TC::TILES_AVAILABLE), 42);
    // Other fields unaffected
    EXPECT_EQ(reg_.read_tile_counter(3, TC::SPACE_AVAILABLE), 0);
}

TEST_F(TensixSplRegTest, TileCounterAllEntriesIndependent) {
    using TC = TensixSplReg::TileCounterField;
    for (int i = 0; i < reg_.num_tile_counters(); ++i) {
        reg_.write_tile_counter(i, TC::TILES_AVAILABLE, i * 10);
    }
    for (int i = 0; i < reg_.num_tile_counters(); ++i) {
        EXPECT_EQ(reg_.read_tile_counter(i, TC::TILES_AVAILABLE), i * 10);
    }
}

// ----------------------------------------------------------------
// Semaphore structured accessors
// ----------------------------------------------------------------

TEST_F(TensixSplRegTest, SemaphoreInitialisedToMinus1) {
    using SF = TensixSplReg::SemField;
    EXPECT_EQ(reg_.read_semaphore(0, 0, SF::ID), -1);
    EXPECT_EQ(reg_.read_semaphore(0, 0, SF::CURRENT_VALUE), -1);
}

TEST_F(TensixSplRegTest, SemaphoreReadWrite) {
    using SF = TensixSplReg::SemField;
    reg_.write_semaphore(0, 0, SF::INIT_VALUE, 5);
    reg_.write_semaphore(0, 0, SF::MAX_VALUE, 10);
    EXPECT_EQ(reg_.read_semaphore(0, 0, SF::INIT_VALUE), 5);
    EXPECT_EQ(reg_.read_semaphore(0, 0, SF::MAX_VALUE), 10);
}

TEST_F(TensixSplRegTest, SemaphoreBankIsolation) {
    using SF = TensixSplReg::SemField;
    reg_.write_semaphore(0, 0, SF::CURRENT_VALUE, 1);
    reg_.write_semaphore(1, 0, SF::CURRENT_VALUE, 2);
    EXPECT_EQ(reg_.read_semaphore(0, 0, SF::CURRENT_VALUE), 1);
    EXPECT_EQ(reg_.read_semaphore(1, 0, SF::CURRENT_VALUE), 2);
}

TEST_F(TensixSplRegTest, SemaphoreAllBanksAndSems) {
    using SF = TensixSplReg::SemField;
    // Touch last bank, last semaphore, last field
    reg_.write_semaphore(TensixSplReg::NUM_SEM_BANKS - 1,
                         TensixSplReg::NUM_SEM_PER_BANK - 1,
                         SF::THREAD_ID_OF_PIPES, 99);
    EXPECT_EQ(reg_.read_semaphore(TensixSplReg::NUM_SEM_BANKS - 1,
                                   TensixSplReg::NUM_SEM_PER_BANK - 1,
                                   SF::THREAD_ID_OF_PIPES), 99);
}
