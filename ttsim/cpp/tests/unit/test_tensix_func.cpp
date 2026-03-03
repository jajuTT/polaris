// SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "neosim/units/tensix_func.hpp"
#include "neosim/units/tensix_spl_reg.hpp"
#include "neosim/isa/instruction.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <memory>

using namespace neosim;
using namespace neosim::units;

// ---------------------------------------------------------------------------
// Test fixture
// ---------------------------------------------------------------------------

class TensixFuncTest : public ::testing::Test {
protected:
    TensixSplReg::Config spl_cfg_;
    std::unique_ptr<TensixSplReg> spl_regs_;
    TensixFunc::Config func_cfg_;
    std::unique_ptr<TensixFunc> func_;

    void SetUp() override {
        // Minimal TensixSplReg config — just enough for all test cases
        spl_cfg_.core_id          = 0;
        spl_cfg_.cfg_start        = 0;
        spl_cfg_.cfg_end          = 4096;
        spl_cfg_.cfg_bytes_per_reg = 4;
        spl_cfg_.mop_start        = 0x10000;
        spl_cfg_.pcbuf_start      = 0x20000;
        spl_cfg_.tile_cnt_start   = 0x30000;
        spl_cfg_.tile_cnt_end     = 0x30000 + 32 * 32 - 1;
        spl_cfg_.tile_cnt_entry_bytes = 32;
        spl_regs_ = std::make_unique<TensixSplReg>(spl_cfg_);

        // Pipe config — matches the flat pipe list order used throughout
        func_cfg_.arch      = "ttqs";
        func_cfg_.core_id   = 0;
        func_cfg_.llk_group = 0;
        func_cfg_.pipe_grps = {
            {"TDMA",   {"TDMA0"}},
            {"SYNC",   {"SYNC0"}},
            {"PACK",   {"PACKER0", "PACKER1"}},
            {"UNPACK", {"UNPACKER0", "UNPACKER1", "UNPACKER2"}},
            {"XMOV",   {"XMOV0"}},
            {"THCON",  {"THCON0"}},
            {"MATH",   {"MATH0"}},
            {"CFG",    {"CFG0"}},
            {"SFPU",   {"SFPU0"}},
        };
        // Flat ordered pipe list: bit position i in stall bitmask → pipe at index i
        func_cfg_.pipes = {
            "TDMA0",      // bit 0 (TDMA)
            "SYNC0",      // bit 1 (SYNC)
            "PACKER0",    // bit 2 (PACK)
            "PACKER1",    // bit 2 (PACK, second pipe)
            "UNPACKER0",  // bit 3 (UNPACK)
            "UNPACKER1",  // bit 3 (UNPACK, second)
            "UNPACKER2",  // bit 3 (UNPACK, third)
            "XMOV0",      // bit 4
            "THCON0",     // bit 5
            "MATH0",      // bit 6
            "CFG0",       // bit 7
            "SFPU0",      // bit 8
        };
        func_ = std::make_unique<TensixFunc>(func_cfg_, *spl_regs_);
    }

    /// Build a minimal Instruction with the given mnemonic and attributes.
    isa::Instruction make_ins(
        const std::string& op,
        const std::map<std::string, int>& attrs = {},
        uint32_t addr = 0x1000)
    {
        isa::Instruction ins;
        ins.set_op(op);
        ins.program_counter = addr;
        if (!attrs.empty()) ins.set_attr(attrs);
        return ins;
    }
};

// ---------------------------------------------------------------------------
// Unknown opcode: returns PC+4 without crashing
// ---------------------------------------------------------------------------

TEST_F(TensixFuncTest, ExecTTIns_Unknown_ReturnsPC4) {
    auto ins = make_ins("BOGUS_OPCODE_XYZ");
    EXPECT_EQ(func_->exec_tt_ins(ins, 0), 0x1000 + 4);
}

// ---------------------------------------------------------------------------
// UNPACR0_TILE_INC: SetDatValid=1 → ex_pipe=UNPACKER0, dst=[srcA], vld/bank=1
// ---------------------------------------------------------------------------

TEST_F(TensixFuncTest, ExecTTIns_UNPACR0_Tile_SetDatValid) {
    auto ins = make_ins("UNPACR0_TILE_INC",
                        {{"SetDatValid", 1}, {"Buffer_Descriptor_Table_Sel", 0}});
    EXPECT_EQ(func_->exec_tt_ins(ins, 0), 0x1000 + 4);
    EXPECT_EQ(ins.get_ex_pipe(), "UNPACKER0");
    ASSERT_EQ(ins.get_dst_int().size(), 1u);
    EXPECT_EQ(ins.get_dst_int()[0], 0); // SRC_A
    EXPECT_EQ(ins.get_vld_upd_mask(0), 1);
    EXPECT_EQ(ins.get_bank_upd_mask(0), 1);
    EXPECT_TRUE(ins.get_src_int().empty());
}

// ---------------------------------------------------------------------------
// UNPACR0_TILE_INC: SetDatValid=0 → vld/bank=0
// ---------------------------------------------------------------------------

TEST_F(TensixFuncTest, ExecTTIns_UNPACR0_Tile_NoClear) {
    auto ins = make_ins("UNPACR0_TILE_INC",
                        {{"SetDatValid", 0}, {"Buffer_Descriptor_Table_Sel", 0}});
    func_->exec_tt_ins(ins, 0);
    EXPECT_EQ(ins.get_vld_upd_mask(0), 0);
    EXPECT_EQ(ins.get_bank_upd_mask(0), 0);
}

// ---------------------------------------------------------------------------
// UNPACR1_TILE_INC: dst=[SRC_B=1], ex_pipe=UNPACKER1
// ---------------------------------------------------------------------------

TEST_F(TensixFuncTest, ExecTTIns_UNPACR1_TileInc) {
    auto ins = make_ins("UNPACR1_TILE_INC",
                        {{"SetDatValid", 1}, {"Buffer_Descriptor_Table_Sel", 0}});
    func_->exec_tt_ins(ins, 0);
    EXPECT_EQ(ins.get_ex_pipe(), "UNPACKER1");
    ASSERT_EQ(ins.get_dst_int().size(), 1u);
    EXPECT_EQ(ins.get_dst_int()[0], 1); // SRC_B
    EXPECT_EQ(ins.get_vld_upd_mask(1), 1);
}

// ---------------------------------------------------------------------------
// UNPACR2_STRIDE: dst=[SRC_S=2], ex_pipe=UNPACKER2
// ---------------------------------------------------------------------------

TEST_F(TensixFuncTest, ExecTTIns_UNPACR2_Stride) {
    auto ins = make_ins("UNPACR2_STRIDE",
                        {{"SetDatValid", 1}, {"Buffer_Descriptor_Table_Sel", 0}});
    func_->exec_tt_ins(ins, 0);
    EXPECT_EQ(ins.get_ex_pipe(), "UNPACKER2");
    ASSERT_EQ(ins.get_dst_int().size(), 1u);
    EXPECT_EQ(ins.get_dst_int()[0], 2); // SRC_S
}

// ---------------------------------------------------------------------------
// UNPACR_DEST_TILE: dst=[DST=3], ex_pipe=UNPACKER0
// ---------------------------------------------------------------------------

TEST_F(TensixFuncTest, ExecTTIns_UNPACR_DEST_Tile) {
    auto ins = make_ins("UNPACR_DEST_TILE",
                        {{"SetDatValid", 1}, {"Buffer_Descriptor_Table_Sel", 0}});
    func_->exec_tt_ins(ins, 0);
    EXPECT_EQ(ins.get_ex_pipe(), "UNPACKER0");
    ASSERT_EQ(ins.get_dst_int().size(), 1u);
    EXPECT_EQ(ins.get_dst_int()[0], 3); // DST
}

// ---------------------------------------------------------------------------
// PACR0_TILE: src=[DST=3], ex_pipe=PACKER0, ClrDatValid=1 → vld/bank=1
// ---------------------------------------------------------------------------

TEST_F(TensixFuncTest, ExecTTIns_PACR0_Tile) {
    auto ins = make_ins("PACR0_TILE",
                        {{"ClrDatValid", 1}, {"Buffer_Descriptor_Table_Sel", 0}});
    func_->exec_tt_ins(ins, 0);
    EXPECT_EQ(ins.get_ex_pipe(), "PACKER0");
    ASSERT_EQ(ins.get_src_int().size(), 1u);
    EXPECT_EQ(ins.get_src_int()[0], 3); // DST
    EXPECT_EQ(ins.get_vld_upd_mask(3), 1);
    EXPECT_TRUE(ins.get_dst_int().empty());
}

// ---------------------------------------------------------------------------
// PACR1_TILE: src=[SRC_S=2], ex_pipe=PACKER1
// ---------------------------------------------------------------------------

TEST_F(TensixFuncTest, ExecTTIns_PACR1_Tile) {
    auto ins = make_ins("PACR1_TILE",
                        {{"ClrDatValid", 0}, {"Buffer_Descriptor_Table_Sel", 0}});
    func_->exec_tt_ins(ins, 0);
    EXPECT_EQ(ins.get_ex_pipe(), "PACKER1");
    ASSERT_EQ(ins.get_src_int().size(), 1u);
    EXPECT_EQ(ins.get_src_int()[0], 2); // SRC_S
    EXPECT_EQ(ins.get_vld_upd_mask(2), 0);
}

// ---------------------------------------------------------------------------
// ELWADD: clear_dvalid=0 → all vld/bank=0
// ---------------------------------------------------------------------------

TEST_F(TensixFuncTest, ExecTTIns_ELWADD_ClearDvalid0) {
    auto ins = make_ins("ELWADD", {{"clear_dvalid", 0}});
    func_->exec_tt_ins(ins, 0);
    EXPECT_EQ(ins.get_vld_upd_mask(0), 0); // SRC_A
    EXPECT_EQ(ins.get_vld_upd_mask(1), 0); // SRC_B
    ASSERT_EQ(ins.get_src_int().size(), 2u);
    ASSERT_EQ(ins.get_dst_int().size(), 1u);
    EXPECT_EQ(ins.get_dst_int()[0], 3); // DST
}

// ---------------------------------------------------------------------------
// ELWADD: clear_dvalid=1 → vld[srcA]=1, vld[srcB]=0
// ---------------------------------------------------------------------------

TEST_F(TensixFuncTest, ExecTTIns_ELWADD_ClearDvalid1) {
    auto ins = make_ins("ELWADD", {{"clear_dvalid", 1}});
    func_->exec_tt_ins(ins, 0);
    EXPECT_EQ(ins.get_vld_upd_mask(0), 1); // SRC_A
    EXPECT_EQ(ins.get_bank_upd_mask(0), 1);
    EXPECT_EQ(ins.get_vld_upd_mask(1), 0); // SRC_B
}

// ---------------------------------------------------------------------------
// ELWADD: clear_dvalid=3 → vld[srcA]=vld[srcB]=1
// ---------------------------------------------------------------------------

TEST_F(TensixFuncTest, ExecTTIns_ELWADD_ClearDvalid3) {
    auto ins = make_ins("ELWADD", {{"clear_dvalid", 3}});
    func_->exec_tt_ins(ins, 0);
    EXPECT_EQ(ins.get_vld_upd_mask(0), 1);
    EXPECT_EQ(ins.get_vld_upd_mask(1), 1);
}

// ---------------------------------------------------------------------------
// CLEARDVALID: cleardvalid=1 → src=[srcA], vld[srcA]=1
// ---------------------------------------------------------------------------

TEST_F(TensixFuncTest, ExecTTIns_CLEARDVALID_SrcA) {
    auto ins = make_ins("CLEARDVALID", {{"cleardvalid", 1}});
    func_->exec_tt_ins(ins, 0);
    const auto src = ins.get_src_int();
    ASSERT_EQ(src.size(), 1u);
    EXPECT_EQ(src[0], 0); // SRC_A
    EXPECT_EQ(ins.get_vld_upd_mask(0), 1);
    EXPECT_EQ(ins.get_bank_upd_mask(0), 1);
    EXPECT_TRUE(ins.get_dst_int().empty());
}

// ---------------------------------------------------------------------------
// CLEARDVALID: cleardvalid_S=1 → src includes SRC_S=2
// ---------------------------------------------------------------------------

TEST_F(TensixFuncTest, ExecTTIns_CLEARDVALID_SrcS) {
    auto ins = make_ins("CLEARDVALID", {{"cleardvalid", 0}, {"cleardvalid_S", 1}});
    func_->exec_tt_ins(ins, 0);
    const auto src = ins.get_src_int();
    ASSERT_EQ(src.size(), 1u);
    EXPECT_EQ(src[0], 2); // SRC_S
    EXPECT_EQ(ins.get_vld_upd_mask(2), 1);
}

// ---------------------------------------------------------------------------
// CLEARDVALID: dest_pulse_last=2 (MATH) → dst=[DST], vld[dst]=1
// ---------------------------------------------------------------------------

TEST_F(TensixFuncTest, ExecTTIns_CLEARDVALID_DestMath) {
    auto ins = make_ins("CLEARDVALID",
                        {{"cleardvalid", 0}, {"cleardvalid_S", 0}, {"dest_pulse_last", 2}});
    func_->exec_tt_ins(ins, 0);
    const auto dst = ins.get_dst_int();
    ASSERT_EQ(dst.size(), 1u);
    EXPECT_EQ(dst[0], 3); // DST
    EXPECT_EQ(ins.get_vld_upd_mask(3), 1);
    EXPECT_EQ(ins.get_bank_upd_mask(3), 1);
    EXPECT_TRUE(ins.get_src_int().empty());
}

// ---------------------------------------------------------------------------
// CLEARDVALID: dest_pulse_last=8 (PACKER) → src=[DST], vld[dst]=1
// ---------------------------------------------------------------------------

TEST_F(TensixFuncTest, ExecTTIns_CLEARDVALID_Packer) {
    auto ins = make_ins("CLEARDVALID",
                        {{"cleardvalid", 0}, {"cleardvalid_S", 0}, {"dest_pulse_last", 8}});
    func_->exec_tt_ins(ins, 0);
    const auto src = ins.get_src_int();
    ASSERT_EQ(src.size(), 1u);
    EXPECT_EQ(src[0], 3); // DST in src list
    EXPECT_EQ(ins.get_vld_upd_mask(3), 1);
    EXPECT_TRUE(ins.get_dst_int().empty());
}

// ---------------------------------------------------------------------------
// SETRWC: clear_ab_vld=1 → src=[srcA]
// ---------------------------------------------------------------------------

TEST_F(TensixFuncTest, ExecTTIns_SETRWC_ClearSrcA) {
    auto ins = make_ins("SETRWC", {{"clear_ab_vld", 1}});
    func_->exec_tt_ins(ins, 0);
    const auto src = ins.get_src_int();
    ASSERT_EQ(src.size(), 1u);
    EXPECT_EQ(src[0], 0); // SRC_A
    EXPECT_EQ(ins.get_vld_upd_mask(0), 1);
    EXPECT_TRUE(ins.get_dst_int().empty());
}

// ---------------------------------------------------------------------------
// SETRWC: clear_ab_vld=3 → src=[srcA, srcB]
// ---------------------------------------------------------------------------

TEST_F(TensixFuncTest, ExecTTIns_SETRWC_ClearBoth) {
    auto ins = make_ins("SETRWC", {{"clear_ab_vld", 3}});
    func_->exec_tt_ins(ins, 0);
    const auto src = ins.get_src_int();
    ASSERT_EQ(src.size(), 2u);
    EXPECT_EQ(ins.get_vld_upd_mask(0), 1);
    EXPECT_EQ(ins.get_vld_upd_mask(1), 1);
}

// ---------------------------------------------------------------------------
// MOVA2D: src=[srcA=0], dst=[DST=3], vld/bank=0
// ---------------------------------------------------------------------------

TEST_F(TensixFuncTest, ExecTTIns_MOVA2D) {
    auto ins = make_ins("MOVA2D");
    func_->exec_tt_ins(ins, 0);
    ASSERT_EQ(ins.get_src_int().size(), 1u);
    EXPECT_EQ(ins.get_src_int()[0], 0); // SRC_A
    ASSERT_EQ(ins.get_dst_int().size(), 1u);
    EXPECT_EQ(ins.get_dst_int()[0], 3); // DST
    EXPECT_EQ(ins.get_vld_upd_mask(0), 0);
    EXPECT_EQ(ins.get_vld_upd_mask(3), 0);
}

// ---------------------------------------------------------------------------
// MOVD2A: src=[DST=3], dst=[srcA=0]
// ---------------------------------------------------------------------------

TEST_F(TensixFuncTest, ExecTTIns_MOVD2A) {
    auto ins = make_ins("MOVD2A");
    func_->exec_tt_ins(ins, 0);
    ASSERT_EQ(ins.get_src_int().size(), 1u);
    EXPECT_EQ(ins.get_src_int()[0], 3); // DST
    ASSERT_EQ(ins.get_dst_int().size(), 1u);
    EXPECT_EQ(ins.get_dst_int()[0], 0); // SRC_A
}

// ---------------------------------------------------------------------------
// MOVB2A: src=[srcB=1], dst=[srcA=0]
// ---------------------------------------------------------------------------

TEST_F(TensixFuncTest, ExecTTIns_MOVB2A) {
    auto ins = make_ins("MOVB2A");
    func_->exec_tt_ins(ins, 0);
    ASSERT_EQ(ins.get_src_int().size(), 1u);
    EXPECT_EQ(ins.get_src_int()[0], 1); // SRC_B
    ASSERT_EQ(ins.get_dst_int().size(), 1u);
    EXPECT_EQ(ins.get_dst_int()[0], 0); // SRC_A
}

// ---------------------------------------------------------------------------
// DMANOP: trivial — returns PC+4, does not modify any register fields
// ---------------------------------------------------------------------------

TEST_F(TensixFuncTest, ExecTTIns_DMANOP_Trivial) {
    auto ins = make_ins("DMANOP");
    const int next = func_->exec_tt_ins(ins, 5);
    EXPECT_EQ(next, 0x1000 + 4);
    EXPECT_TRUE(ins.get_src_int().empty());
    EXPECT_TRUE(ins.get_dst_int().empty());
    EXPECT_TRUE(ins.get_ex_pipe().empty());
}

// ---------------------------------------------------------------------------
// SEMINIT: writes semaphore fields to TensixSplReg
// ---------------------------------------------------------------------------

TEST_F(TensixFuncTest, ExecTTIns_SEMINIT_Init) {
    isa::Instruction ins = make_ins("SEMINIT");
    // sem_sel=1 → id_in_bank=0; bank_id=0 (non-ttqs)
    ttdecode::decode::operands ops;
    ops.all        = {{"sem_sel", 1}, {"init_value", 3}, {"max_value", 7}};
    ins.operands   = ops;

    const int next = func_->exec_tt_ins(ins, 0);
    EXPECT_EQ(next, 0x1000 + 4);

    using SF = TensixSplReg::SemField;
    EXPECT_EQ(spl_regs_->read_semaphore(0, 0, SF::ID),            0);
    EXPECT_EQ(spl_regs_->read_semaphore(0, 0, SF::INIT_VALUE),    3);
    EXPECT_EQ(spl_regs_->read_semaphore(0, 0, SF::MAX_VALUE),     7);
    EXPECT_EQ(spl_regs_->read_semaphore(0, 0, SF::CURRENT_VALUE), 3);
    EXPECT_EQ(spl_regs_->read_semaphore(0, 0, SF::PIPES_TO_STALL),0);
}

// ---------------------------------------------------------------------------
// SEMGET: decrements current_value by 1
// ---------------------------------------------------------------------------

TEST_F(TensixFuncTest, ExecTTIns_SEMGET_Decrements) {
    using SF = TensixSplReg::SemField;
    // Pre-init semaphore 0 in bank 0 with current=5
    spl_regs_->write_semaphore(0, 0, SF::ID,            0);
    spl_regs_->write_semaphore(0, 0, SF::BANK,          0);
    spl_regs_->write_semaphore(0, 0, SF::INIT_VALUE,    0);
    spl_regs_->write_semaphore(0, 0, SF::MAX_VALUE,     8);
    spl_regs_->write_semaphore(0, 0, SF::CURRENT_VALUE, 5);
    spl_regs_->write_semaphore(0, 0, SF::PIPES_TO_STALL, 0);

    isa::Instruction ins = make_ins("SEMGET");
    ttdecode::decode::operands ops;
    ops.all = {{"sem_sel", 1}};  // sem_sel=1 → id=0
    ins.operands = ops;

    func_->exec_tt_ins(ins, 0);
    EXPECT_EQ(spl_regs_->read_semaphore(0, 0, SF::CURRENT_VALUE), 4);
    EXPECT_TRUE(ins.get_dst_pipes().empty()); // no pipes stalled
}

// ---------------------------------------------------------------------------
// SEMPOST: increments current_value by 1
// ---------------------------------------------------------------------------

TEST_F(TensixFuncTest, ExecTTIns_SEMPOST_Increments) {
    using SF = TensixSplReg::SemField;
    spl_regs_->write_semaphore(0, 0, SF::ID,            0);
    spl_regs_->write_semaphore(0, 0, SF::CURRENT_VALUE, 2);
    spl_regs_->write_semaphore(0, 0, SF::PIPES_TO_STALL, 0);

    isa::Instruction ins = make_ins("SEMPOST");
    ttdecode::decode::operands ops;
    ops.all = {{"sem_sel", 1}};
    ins.operands = ops;

    func_->exec_tt_ins(ins, 0);
    EXPECT_EQ(spl_regs_->read_semaphore(0, 0, SF::CURRENT_VALUE), 3);
}

// ---------------------------------------------------------------------------
// SEMWAIT: stalls when wait_cond==1 && current==0 (condition met)
// ---------------------------------------------------------------------------

TEST_F(TensixFuncTest, ExecTTIns_SEMWAIT_StallsWhenCondMet) {
    using SF = TensixSplReg::SemField;
    // Semaphore 0: max=4, current=0 → wait_cond==1 triggers stall
    spl_regs_->write_semaphore(0, 0, SF::ID,            0);
    spl_regs_->write_semaphore(0, 0, SF::BANK,          0);
    spl_regs_->write_semaphore(0, 0, SF::INIT_VALUE,    0);
    spl_regs_->write_semaphore(0, 0, SF::MAX_VALUE,     4);
    spl_regs_->write_semaphore(0, 0, SF::CURRENT_VALUE, 0);
    spl_regs_->write_semaphore(0, 0, SF::PIPES_TO_STALL, 0);

    isa::Instruction ins = make_ins("SEMWAIT");
    ins.set_thread_id(0);
    ttdecode::decode::operands ops;
    // wait_cond=1 → stall when current==0; stall_res bit 3 = UNPACK
    ops.all = {{"sem_sel", 1}, {"wait_sem_cond", 1}, {"stall_res", 8}};
    ins.operands = ops;

    func_->exec_tt_ins(ins, 0);
    EXPECT_FALSE(ins.get_dst_pipes().empty());
    // UNPACK group should contain UNPACKER0/1/2
    const auto& dp = ins.get_dst_pipes();
    EXPECT_TRUE(std::find(dp.begin(), dp.end(), "UNPACKER0") != dp.end());
}

// ---------------------------------------------------------------------------
// SEMWAIT: no stall when condition is NOT met
// ---------------------------------------------------------------------------

TEST_F(TensixFuncTest, ExecTTIns_SEMWAIT_NoStallWhenCondClear) {
    using SF = TensixSplReg::SemField;
    // current=2, max=4 → wait_cond==1 does NOT trigger stall (current != 0)
    spl_regs_->write_semaphore(0, 0, SF::ID,            0);
    spl_regs_->write_semaphore(0, 0, SF::BANK,          0);
    spl_regs_->write_semaphore(0, 0, SF::INIT_VALUE,    0);
    spl_regs_->write_semaphore(0, 0, SF::MAX_VALUE,     4);
    spl_regs_->write_semaphore(0, 0, SF::CURRENT_VALUE, 2);
    spl_regs_->write_semaphore(0, 0, SF::PIPES_TO_STALL, 0);

    isa::Instruction ins = make_ins("SEMWAIT");
    ttdecode::decode::operands ops;
    ops.all = {{"sem_sel", 1}, {"wait_sem_cond", 1}, {"stall_res", 8}};
    ins.operands = ops;

    func_->exec_tt_ins(ins, 0);
    EXPECT_TRUE(ins.get_dst_pipes().empty());
}

// ---------------------------------------------------------------------------
// STALLWAIT: stall_res bitmask → correct pipe name list
// ---------------------------------------------------------------------------

TEST_F(TensixFuncTest, ExecTTIns_STALLWAIT_DstPipes) {
    isa::Instruction ins = make_ins("STALLWAIT");
    ttdecode::decode::operands ops;
    // bit 3 = UNPACK → UNPACKER0, UNPACKER1, UNPACKER2
    ops.all = {{"stall_res", 8}};
    ins.operands = ops;

    func_->exec_tt_ins(ins, 0);
    const auto& dp = ins.get_dst_pipes();
    ASSERT_EQ(dp.size(), 3u);
    EXPECT_TRUE(std::find(dp.begin(), dp.end(), "UNPACKER0") != dp.end());
    EXPECT_TRUE(std::find(dp.begin(), dp.end(), "UNPACKER1") != dp.end());
    EXPECT_TRUE(std::find(dp.begin(), dp.end(), "UNPACKER2") != dp.end());
}

// ---------------------------------------------------------------------------
// REPLAY: stores params in mem_info_, returns PC+4
// ---------------------------------------------------------------------------

TEST_F(TensixFuncTest, ExecTTIns_REPLAY_StoresParams) {
    auto ins = make_ins("REPLAY", {
        {"load_mode", 1},
        {"execute_while_loading", 0},
        {"start_idx", 5},
        {"len", 10}
    });
    const int next = func_->exec_tt_ins(ins, 7);
    EXPECT_EQ(next, 0x1000 + 4);
    EXPECT_EQ(ins.get_mem_info("replay_load_mode"),             1);
    EXPECT_EQ(ins.get_mem_info("replay_execute_while_loading"), 0);
    EXPECT_EQ(ins.get_mem_info("replay_start_idx"),             5);
    EXPECT_EQ(ins.get_mem_info("replay_len"),                   10);
}

// ---------------------------------------------------------------------------
// MOP expansion: 1×1 loop with only instr0 → [instr0]
// ---------------------------------------------------------------------------

TEST_F(TensixFuncTest, BuildInsFromMop_1x1Loop_InstrOnly) {
    using SR = TensixSplReg::SplRegType;
    const int base     = 64 * 0; // thread 0
    const int32_t NOP  = 0x2000000;
    const int32_t INSTR_A = 0x12345678;

    spl_regs_->write_reg(base + 0, 1,       SR::MOP); // loop0_len
    spl_regs_->write_reg(base + 1, 1,       SR::MOP); // loop1_len
    spl_regs_->write_reg(base + 2, NOP,     SR::MOP); // loop_start_instr0
    spl_regs_->write_reg(base + 3, NOP,     SR::MOP); // loop_end_instr0
    spl_regs_->write_reg(base + 4, NOP,     SR::MOP); // loop_end_instr1
    spl_regs_->write_reg(base + 5, INSTR_A, SR::MOP); // loop_instr0
    spl_regs_->write_reg(base + 6, NOP,     SR::MOP); // loop_instr1
    spl_regs_->write_reg(base + 7, NOP,     SR::MOP); // loop0_last_instr
    spl_regs_->write_reg(base + 8, NOP,     SR::MOP); // loop1_last_instr
    spl_regs_->write_reg(base + 9, 0,       SR::MOP); // mop_sw_ctrl

    isa::Instruction ins = make_ins("MOP", {{"mop_type", 1}});
    ins.set_thread_id(0);
    const auto result = func_->build_ins_from_mop(ins);
    ASSERT_EQ(result.size(), 1u);
    EXPECT_EQ(result[0], static_cast<uint32_t>(INSTR_A));
}

// ---------------------------------------------------------------------------
// MOP expansion: NOP words excluded from output
// ---------------------------------------------------------------------------

TEST_F(TensixFuncTest, BuildInsFromMop_NopFiltered) {
    using SR = TensixSplReg::SplRegType;
    const int base    = 64 * 0;
    const int32_t NOP = 0x2000000;
    const int32_t INSTR_A = 0xAABBCCDD;

    // 1×2 loop: start=NOP, instr0=INSTR_A, instr1=NOP, loop1_last=NOP, loop0_last=NOP
    spl_regs_->write_reg(base + 0, 1,       SR::MOP); // loop0_len
    spl_regs_->write_reg(base + 1, 2,       SR::MOP); // loop1_len
    spl_regs_->write_reg(base + 2, NOP,     SR::MOP); // loop_start_instr0 (NOP)
    spl_regs_->write_reg(base + 3, NOP,     SR::MOP); // loop_end_instr0
    spl_regs_->write_reg(base + 4, NOP,     SR::MOP); // loop_end_instr1
    spl_regs_->write_reg(base + 5, INSTR_A, SR::MOP); // loop_instr0
    spl_regs_->write_reg(base + 6, NOP,     SR::MOP); // loop_instr1 (NOP)
    spl_regs_->write_reg(base + 7, NOP,     SR::MOP); // loop0_last_instr
    spl_regs_->write_reg(base + 8, NOP,     SR::MOP); // loop1_last_instr
    spl_regs_->write_reg(base + 9, 0,       SR::MOP); // mop_sw_ctrl

    isa::Instruction ins = make_ins("MOP", {{"mop_type", 1}});
    ins.set_thread_id(0);
    const auto result = func_->build_ins_from_mop(ins);

    // j=0 (normal): push INSTR_A (instr1 is NOP so skipped)
    // j=1 (last_inner && last_outer, else case): push INSTR_A
    ASSERT_EQ(result.size(), 2u);
    EXPECT_EQ(result[0], static_cast<uint32_t>(INSTR_A));
    EXPECT_EQ(result[1], static_cast<uint32_t>(INSTR_A));
    // NOP (0x2000000) never appears
    for (auto w : result) EXPECT_NE(w, static_cast<uint32_t>(NOP));
}

// ---------------------------------------------------------------------------
// MOP expansion: epilogue instructions appended after last outer iteration
// ---------------------------------------------------------------------------

TEST_F(TensixFuncTest, BuildInsFromMop_EpilogueAfterLastOuter) {
    using SR = TensixSplReg::SplRegType;
    const int base     = 64 * 0;
    const int32_t NOP  = 0x2000000;
    const int32_t INSTR_A = 0x11111111;
    const int32_t END0    = 0x22222222;
    const int32_t END1    = 0x33333333;

    // 1×1 loop with epilogue END0 and END1
    spl_regs_->write_reg(base + 0, 1,      SR::MOP); // loop0_len
    spl_regs_->write_reg(base + 1, 1,      SR::MOP); // loop1_len
    spl_regs_->write_reg(base + 2, NOP,    SR::MOP); // loop_start_instr0
    spl_regs_->write_reg(base + 3, END0,   SR::MOP); // loop_end_instr0
    spl_regs_->write_reg(base + 4, END1,   SR::MOP); // loop_end_instr1
    spl_regs_->write_reg(base + 5, INSTR_A,SR::MOP); // loop_instr0
    spl_regs_->write_reg(base + 6, NOP,    SR::MOP); // loop_instr1
    spl_regs_->write_reg(base + 7, NOP,    SR::MOP); // loop0_last_instr
    spl_regs_->write_reg(base + 8, NOP,    SR::MOP); // loop1_last_instr
    spl_regs_->write_reg(base + 9, 0,      SR::MOP); // mop_sw_ctrl

    isa::Instruction ins = make_ins("MOP", {{"mop_type", 1}});
    ins.set_thread_id(0);
    const auto result = func_->build_ins_from_mop(ins);
    // Loop body: INSTR_A; Epilogue: END0, END1
    ASSERT_EQ(result.size(), 3u);
    EXPECT_EQ(result[0], static_cast<uint32_t>(INSTR_A));
    EXPECT_EQ(result[1], static_cast<uint32_t>(END0));
    EXPECT_EQ(result[2], static_cast<uint32_t>(END1));
}

// ---------------------------------------------------------------------------
// MOP expansion: 2×2 loop with last-iteration variants
// ---------------------------------------------------------------------------

TEST_F(TensixFuncTest, BuildInsFromMop_2x2Loop) {
    using SR = TensixSplReg::SplRegType;
    const int base     = 64 * 0;
    const int32_t NOP  = 0x2000000;
    const int32_t A    = 0xAAAAAAAA; // loop_instr0
    const int32_t B    = 0xBBBBBBBB; // loop_instr1
    const int32_t C    = 0xCCCCCCCC; // loop0_last_instr
    const int32_t D    = 0xDDDDDDDD; // loop1_last_instr

    spl_regs_->write_reg(base + 0, 2,   SR::MOP); // loop0_len=2
    spl_regs_->write_reg(base + 1, 2,   SR::MOP); // loop1_len=2
    spl_regs_->write_reg(base + 2, NOP, SR::MOP); // loop_start_instr0=NOP
    spl_regs_->write_reg(base + 3, NOP, SR::MOP); // loop_end_instr0=NOP
    spl_regs_->write_reg(base + 4, NOP, SR::MOP); // loop_end_instr1=NOP
    spl_regs_->write_reg(base + 5, A,   SR::MOP); // loop_instr0
    spl_regs_->write_reg(base + 6, B,   SR::MOP); // loop_instr1
    spl_regs_->write_reg(base + 7, C,   SR::MOP); // loop0_last_instr
    spl_regs_->write_reg(base + 8, D,   SR::MOP); // loop1_last_instr
    spl_regs_->write_reg(base + 9, 0,   SR::MOP); // mop_sw_ctrl

    isa::Instruction ins = make_ins("MOP", {{"mop_type", 1}});
    ins.set_thread_id(0);
    const auto result = func_->build_ins_from_mop(ins);

    // i=0, j=0 (normal):        [A, B]
    // i=0, j=1 (last_inner, !last_outer): instr0&&instr1&&loop1_last → [A, D]
    // i=1, j=0 (normal, last_outer):      [A, B]
    // i=1, j=1 (last_inner&&last_outer):  instr0&&instr1&&loop0_last → [A, C]
    // epilogue (i==last outer): end_instr0=NOP, end_instr1=NOP → nothing
    ASSERT_EQ(result.size(), 8u);
    const std::vector<uint32_t> expected = {
        (uint32_t)A, (uint32_t)B,  // i=0,j=0
        (uint32_t)A, (uint32_t)D,  // i=0,j=1 last_inner
        (uint32_t)A, (uint32_t)B,  // i=1,j=0
        (uint32_t)A, (uint32_t)C,  // i=1,j=1 last_inner&&last_outer
    };
    EXPECT_EQ(result, expected);
}

// ---------------------------------------------------------------------------
// max_autoloop_iterations: returns (max_count+1)*(max_loop+1) from cfg regs
// ---------------------------------------------------------------------------

TEST_F(TensixFuncTest, MaxAutoloopIterations_Product) {
    // Register two bit fields so get_cfg_reg_max_possible_value can compute.
    // 3-bit field → max = 7; 2-bit field → max = 3
    // (7+1)*(3+1) = 32
    spl_regs_->register_cfg_reg("THCON_UNPACKER2_REG0_INSTRN_COUNT",      0, 0, 0x7);
    spl_regs_->register_cfg_reg("THCON_UNPACKER2_REG0_INSTRN_LOOP_COUNT", 0, 4, 0x30);
    spl_regs_->register_cfg_reg("THCON_PACKER1_REG0_INSTRN_COUNT",        1, 0, 0x7);
    spl_regs_->register_cfg_reg("THCON_PACKER1_REG0_INSTRN_LOOP_COUNT",   1, 4, 0x30);

    EXPECT_EQ(func_->max_autoloop_iterations(), (7 + 1) * (3 + 1));
}
