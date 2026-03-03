// SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include "neosim/units/risc_reg.hpp"

using namespace neosim::units;

class RiscRegTest : public ::testing::Test {
protected:
    RiscReg reg_{2}; // 2 threads
};

// ----------------------------------------------------------------
// Construction
// ----------------------------------------------------------------

TEST_F(RiscRegTest, InitialState) {
    EXPECT_EQ(reg_.num_threads(), 2);
    for (int t = 0; t < 2; ++t) {
        for (int r = 0; r < RiscReg::NUM_REGS; ++r) {
            EXPECT_FALSE(reg_.check_in_use(t, r));
            EXPECT_EQ(reg_.check_valid(t, r), 0);
        }
    }
}

// ----------------------------------------------------------------
// set_in_use / check_in_use — toggle semantics
// ----------------------------------------------------------------

TEST_F(RiscRegTest, SetInUse_SetAndClear) {
    EXPECT_FALSE(reg_.check_in_use(0, 5));
    reg_.set_in_use(0, 5, true);
    EXPECT_TRUE(reg_.check_in_use(0, 5));
    reg_.set_in_use(0, 5, false);
    EXPECT_FALSE(reg_.check_in_use(0, 5));
}

TEST_F(RiscRegTest, SetInUse_ToggleAssert_SetTrue) {
#ifndef NDEBUG
    reg_.set_in_use(0, 5, true);
    // Setting to true again should assert — test via death test
    EXPECT_DEATH(reg_.set_in_use(0, 5, true), "");
#else
    GTEST_SKIP() << "Assertions disabled in Release build";
#endif
}

TEST_F(RiscRegTest, SetInUse_ToggleAssert_SetFalse) {
#ifndef NDEBUG
    // Already false; setting to false again should assert
    EXPECT_DEATH(reg_.set_in_use(0, 5, false), "");
#else
    GTEST_SKIP() << "Assertions disabled in Release build";
#endif
}

TEST_F(RiscRegTest, SetInUse_ThreadIsolation) {
    reg_.set_in_use(0, 3, true);
    EXPECT_TRUE(reg_.check_in_use(0, 3));
    EXPECT_FALSE(reg_.check_in_use(1, 3)); // other thread unaffected
}

// ----------------------------------------------------------------
// set_valid / check_valid / reset_valid
// ----------------------------------------------------------------

TEST_F(RiscRegTest, SetValid_RoundTrip) {
    reg_.set_valid(0, 10, 5);
    EXPECT_EQ(reg_.check_valid(0, 10), 5);
}

TEST_F(RiscRegTest, ResetValid_ClearsToZero) {
    reg_.set_valid(1, 7, 3);
    reg_.reset_valid(1, 7, 3);
    EXPECT_EQ(reg_.check_valid(1, 7), 0);
}

TEST_F(RiscRegTest, ResetValid_AssertWrongValue) {
#ifndef NDEBUG
    reg_.set_valid(0, 15, 2);
    // Resetting with wrong expected value should assert
    EXPECT_DEATH(reg_.reset_valid(0, 15, 99), "");
#else
    GTEST_SKIP() << "Assertions disabled in Release build";
#endif
}

TEST_F(RiscRegTest, ValidThreadIsolation) {
    reg_.set_valid(0, 20, 7);
    EXPECT_EQ(reg_.check_valid(0, 20), 7);
    EXPECT_EQ(reg_.check_valid(1, 20), 0); // other thread unaffected
}

// ----------------------------------------------------------------
// Multiple registers independent
// ----------------------------------------------------------------

TEST_F(RiscRegTest, AllRegistersIndependent) {
    for (int r = 0; r < RiscReg::NUM_REGS; ++r) {
        reg_.set_valid(0, r, r + 1);
    }
    for (int r = 0; r < RiscReg::NUM_REGS; ++r) {
        EXPECT_EQ(reg_.check_valid(0, r), r + 1);
    }
}

// ----------------------------------------------------------------
// Boundary checks
// ----------------------------------------------------------------

TEST_F(RiscRegTest, ConstructWithManyThreads) {
    RiscReg big(4);
    EXPECT_EQ(big.num_threads(), 4);
    big.set_in_use(3, 31, true);
    EXPECT_TRUE(big.check_in_use(3, 31));
}
