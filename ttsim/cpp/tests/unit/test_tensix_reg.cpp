// SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include "neosim/units/tensix_reg.hpp"

using namespace neosim::units;

// ----------------------------------------------------------------
// Fixture: TensixReg with a simple 2-entry order scheme for reg 3 (dst).
// orderScheme[3] = {UNPACKER=0, MATH=1}
// All other registers have empty order schemes.
// ----------------------------------------------------------------
class TensixRegTest : public ::testing::Test {
protected:
    // orderScheme: outer vector has NUM_REGS=4 entries.
    // Reg 3 (dst) uses context order {0 (UNPACKER), 1 (MATH)}.
    std::vector<std::vector<int>> scheme_ = {
        {},        // srcA — no order
        {},        // srcB — no order
        {},        // srcS — no order
        {0, 1}     // dst  — UNPACKER then MATH
    };

    TensixReg reg_;

    void SetUp() override {
        reg_.init(0, scheme_);
    }
};

// ----------------------------------------------------------------
// Construction defaults
// ----------------------------------------------------------------

TEST(TensixRegBasic, DefaultConstruction) {
    TensixReg r;
    // Valids default to 0, banks to 0, in_use to false
    for (int reg = 0; reg < TensixReg::NUM_REGS; ++reg) {
        for (int ctx = 0; ctx < TensixReg::NUM_CONTEXTS; ++ctx) {
            EXPECT_EQ(r.peek_curr_bank(reg, ctx), 0);
            EXPECT_EQ(r.read_valid(reg, ctx), 0);
            EXPECT_EQ(r.read_curr_bank(reg, ctx), 0);
        }
        for (int bank = 0; bank < TensixReg::NUM_BANKS; ++bank) {
            EXPECT_FALSE(r.in_use(reg, bank));
            EXPECT_EQ(r.acc_thread_id(reg, bank), -1);
            EXPECT_EQ(r.acc_cnt_to_valid(reg, bank), -1);
        }
    }
}

// ----------------------------------------------------------------
// init() and cond_check_valid / cond_write_valid
// ----------------------------------------------------------------

TEST_F(TensixRegTest, InitCondCheckWrite_NoScheme) {
    // srcA, srcB, srcS have empty order → cond values = -1
    for (int reg = 0; reg < 3; ++reg) {
        for (int ctx = 0; ctx < TensixReg::NUM_CONTEXTS; ++ctx) {
            EXPECT_EQ(reg_.cond_check_valid(reg, ctx), -1);
            EXPECT_EQ(reg_.cond_write_valid(reg, ctx), -1);
        }
    }
}

TEST_F(TensixRegTest, InitCondCheckWrite_DstScheme) {
    // dst (reg=3), scheme = {0,1}
    // context 0 (UNPACKER): cond_check=0, cond_write=1
    EXPECT_EQ(reg_.cond_check_valid(3, 0), 0);
    EXPECT_EQ(reg_.cond_write_valid(3, 0), 1);
    // context 1 (MATH): cond_check=1, cond_write=0 (wrap)
    EXPECT_EQ(reg_.cond_check_valid(3, 1), 1);
    EXPECT_EQ(reg_.cond_write_valid(3, 1), 0);
    // contexts 2,3 not in order scheme → -1
    EXPECT_EQ(reg_.cond_check_valid(3, 2), -1);
    EXPECT_EQ(reg_.cond_check_valid(3, 3), -1);
}

// ----------------------------------------------------------------
// Bank rotation: peek_curr_bank / get_next_bank
// ----------------------------------------------------------------

TEST_F(TensixRegTest, BankRotation) {
    // Initial bank = 0
    EXPECT_EQ(reg_.peek_curr_bank(0, 0), 0);
    // get_next_bank returns OLD bank and advances
    int old = reg_.get_next_bank(0, 0);
    EXPECT_EQ(old, 0);
    EXPECT_EQ(reg_.peek_curr_bank(0, 0), 1);
    // Next rotation wraps
    old = reg_.get_next_bank(0, 0);
    EXPECT_EQ(old, 1);
    EXPECT_EQ(reg_.peek_curr_bank(0, 0), 0);
}

TEST_F(TensixRegTest, BankRotationIndependentPerContext) {
    reg_.get_next_bank(0, 0); // ctx0 → bank1
    EXPECT_EQ(reg_.peek_curr_bank(0, 0), 1);
    EXPECT_EQ(reg_.peek_curr_bank(0, 1), 0); // ctx1 unaffected
}

// ----------------------------------------------------------------
// write_valid — mode 0 (access-count only)
// ----------------------------------------------------------------

TEST_F(TensixRegTest, WriteValidMode0_AccTracking) {
    // mode 0, v_mask=true, b_mask=true
    // valid != val → real update path, mode 0 only updates acc_cnt
    reg_.write_valid(0, 0, 1, true, true, 0);
    // valid at bank 0 should NOT have changed
    EXPECT_EQ(reg_.read_valid(0, 0), 0);
    // bank should NOT have rotated
    EXPECT_EQ(reg_.peek_curr_bank(0, 0), 0);
    // acc_cnt incremented: starts at -1, +1 = 0 on first access
    EXPECT_EQ(reg_.acc_cnt_to_valid(0, 0), 0);
    EXPECT_EQ(reg_.acc_thread_id(0, 0), 0);
}

TEST_F(TensixRegTest, WriteValidMode0_NoMaskAccTracking) {
    // v_mask=false, b_mask=false → early-return no-op path
    reg_.write_valid(0, 0, 1, false, false, 0);
    EXPECT_EQ(reg_.read_valid(0, 0), 0);
    EXPECT_EQ(reg_.acc_cnt_to_valid(0, 0), 0); // acc_cnt += 1 from -1 = 0
}

// ----------------------------------------------------------------
// write_valid — mode 1 (update valid + rotate bank)
// ----------------------------------------------------------------

TEST_F(TensixRegTest, WriteValidMode1_RotatesBank) {
    // Write val=1 to reg 0, ctx 0, mode 1
    reg_.write_valid(0, 0, 1, true, true, 1);
    // bank should have rotated from 0 to 1
    EXPECT_EQ(reg_.peek_curr_bank(0, 0), 1);
    // valid at bank 0 (the old bank) should now be 1
    // After rotation, read_valid reads from bank_sel[0][0]=1 (NEW bank)
    // The old bank (0) should have been written
    EXPECT_EQ(reg_.acc_cnt_to_valid(0, 0), 0);
}

TEST_F(TensixRegTest, WriteValidMode1_ValidWrittenToOldBank) {
    // Bank starts at 0. Mode 1 writes to current bank (get_next_bank returns old=0)
    // then bank_sel advances to 1.
    reg_.write_valid(0, 0, 1, true, true, 1);
    // Bank is now 1. read_valid reads from bank 1 (still 0).
    EXPECT_EQ(reg_.read_valid(0, 0), 0);
    // But valids at bank 0 = 1 (verifiable through read_valid if we temporarily go back)
    // Rotate back to bank 0 manually to verify:
    reg_.get_next_bank(0, 0); // bank 1 → 0
    EXPECT_EQ(reg_.read_valid(0, 0), 1);
}

// ----------------------------------------------------------------
// write_valid — mode 2 (reset in_use, no valid update)
// ----------------------------------------------------------------

TEST_F(TensixRegTest, WriteValidMode2_ResetsInUse) {
    // Set in_use first via check_valid mode 2
    bool ok = reg_.check_valid(0, 0, 0, 2);
    EXPECT_TRUE(ok);
    EXPECT_TRUE(reg_.in_use(0, 0));

    // write_valid mode 2 resets in_use
    reg_.write_valid(0, 0, 0, true, true, 2);
    EXPECT_FALSE(reg_.in_use(0, 0));
    // bank should NOT have rotated
    EXPECT_EQ(reg_.peek_curr_bank(0, 0), 0);
}

// ----------------------------------------------------------------
// write_valid — mode 3 (update valid + rotate + reset in_use)
// ----------------------------------------------------------------

TEST_F(TensixRegTest, WriteValidMode3_UpdatesValidResetsInUse) {
    bool ok = reg_.check_valid(0, 0, 0, 2); // sets in_use
    EXPECT_TRUE(ok);

    reg_.write_valid(0, 0, 1, true, true, 3);
    EXPECT_FALSE(reg_.in_use(0, 0));   // in_use cleared
    EXPECT_EQ(reg_.peek_curr_bank(0, 0), 1); // bank rotated
    EXPECT_EQ(reg_.acc_cnt_to_valid(0, 0), 0);
}

// ----------------------------------------------------------------
// write_valid — early return when valid already matches
// ----------------------------------------------------------------

TEST_F(TensixRegTest, WriteValidEarlyReturn_ValidAlreadyMatches) {
    // valid at bank 0 = 0.  Write val=0 with mode 1 → should NOT rotate bank.
    reg_.write_valid(0, 0, 0, true, true, 1);
    EXPECT_EQ(reg_.peek_curr_bank(0, 0), 0); // bank did NOT rotate
    EXPECT_EQ(reg_.acc_cnt_to_valid(0, 0), 0); // reset (not incremented)
}

// ----------------------------------------------------------------
// check_valid — mode 0 (always true)
// ----------------------------------------------------------------

TEST_F(TensixRegTest, CheckValidMode0_AlwaysTrue) {
    EXPECT_TRUE(reg_.check_valid(0, 0, 0, 0));
    EXPECT_TRUE(reg_.check_valid(0, 0, 1, 0));
}

// ----------------------------------------------------------------
// check_valid — mode 1 (valid == val AND not in_use)
// ----------------------------------------------------------------

TEST_F(TensixRegTest, CheckValidMode1_Conditions) {
    // valid=0, in_use=false, check for val=0 → true
    EXPECT_TRUE(reg_.check_valid(0, 0, 0, 1));
    // check for val=1 → false (valid!=1)
    EXPECT_FALSE(reg_.check_valid(0, 0, 1, 1));

    // Set in_use; now check for val=0 should fail (in_use)
    reg_.check_valid(0, 0, 0, 2); // sets in_use
    EXPECT_FALSE(reg_.check_valid(0, 0, 0, 1));
}

// ----------------------------------------------------------------
// check_valid — mode 2 (set in_use if free)
// ----------------------------------------------------------------

TEST_F(TensixRegTest, CheckValidMode2_SetsInUse) {
    EXPECT_FALSE(reg_.in_use(0, 0));
    bool ok = reg_.check_valid(0, 0, 0, 2);
    EXPECT_TRUE(ok);
    EXPECT_TRUE(reg_.in_use(0, 0));

    // Second call: already in_use → false
    EXPECT_FALSE(reg_.check_valid(0, 0, 0, 2));
}

// ----------------------------------------------------------------
// check_valid — mode 3 (valid == val AND set in_use)
// ----------------------------------------------------------------

TEST_F(TensixRegTest, CheckValidMode3_ValidAndSetsInUse) {
    // valid=0, in_use=false, val=0 → true and sets in_use
    EXPECT_TRUE(reg_.check_valid(0, 0, 0, 3));
    EXPECT_TRUE(reg_.in_use(0, 0));

    // in_use now set → same check returns false
    EXPECT_FALSE(reg_.check_valid(0, 0, 0, 3));
}

TEST_F(TensixRegTest, CheckValidMode3_WrongVal) {
    // valid=0 but checking val=1 → false
    EXPECT_FALSE(reg_.check_valid(0, 0, 1, 3));
    EXPECT_FALSE(reg_.in_use(0, 0)); // in_use NOT set when condition not met
}

// ----------------------------------------------------------------
// write_cond_valid
// ----------------------------------------------------------------

TEST_F(TensixRegTest, WriteCondValid_UpdatesValues) {
    reg_.write_cond_valid(0, 0, 1, 2);
    EXPECT_EQ(reg_.cond_check_valid(0, 0), 1);
    EXPECT_EQ(reg_.cond_write_valid(0, 0), 2);
}

TEST_F(TensixRegTest, WriteCondValid_IgnoreSentinel) {
    // -2 (IGNORE) → no update
    int before_chk = reg_.cond_check_valid(0, 0);
    int before_wri = reg_.cond_write_valid(0, 0);
    reg_.write_cond_valid(0, 0, -2, 3);
    EXPECT_EQ(reg_.cond_check_valid(0, 0), before_chk);
    EXPECT_EQ(reg_.cond_write_valid(0, 0), before_wri);
}

// ----------------------------------------------------------------
// read_valid / read_curr_bank non-side-effect guarantees
// ----------------------------------------------------------------

TEST_F(TensixRegTest, ReadValidNoSideEffect) {
    reg_.get_next_bank(0, 0); // bank → 1
    int val = reg_.read_valid(0, 0);
    EXPECT_EQ(val, 0); // bank 1, valid still 0
    EXPECT_EQ(reg_.peek_curr_bank(0, 0), 1); // no rotation
}

// ----------------------------------------------------------------
// Init with empty order scheme (covers NULL-scheme branch)
// ----------------------------------------------------------------

TEST(TensixRegInit, EmptyOrderScheme) {
    TensixReg r;
    r.init(1, {});
    for (int reg = 0; reg < TensixReg::NUM_REGS; ++reg) {
        for (int ctx = 0; ctx < TensixReg::NUM_CONTEXTS; ++ctx) {
            EXPECT_EQ(r.cond_check_valid(reg, ctx), -1);
            EXPECT_EQ(r.cond_write_valid(reg, ctx), -1);
        }
    }
}
