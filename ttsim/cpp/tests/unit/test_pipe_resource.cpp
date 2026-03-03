// SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include "neosim/units/pipe_resource.hpp"

using namespace neosim::units;

class PipeResourceTest : public ::testing::Test {
protected:
    PipeResource pr_{3, 4, true}; // 3 pipes, 4 threads, threadwise=true
};

// ----------------------------------------------------------------
// Construction
// ----------------------------------------------------------------

TEST_F(PipeResourceTest, InitialState) {
    EXPECT_EQ(pr_.num_pipes(), 3);
    EXPECT_EQ(pr_.num_threads(), 4);
    EXPECT_TRUE(pr_.threadwise());

    for (int p = 0; p < 3; ++p) {
        for (int t = 0; t < 4; ++t) {
            EXPECT_EQ(pr_.read_rsrc_state(p, t), 0);
        }
    }
}

// ----------------------------------------------------------------
// read_rsrc_state — no side effects
// ----------------------------------------------------------------

TEST_F(PipeResourceTest, ReadNoSideEffect) {
    pr_.read_rsrc_state(0, 0);
    EXPECT_EQ(pr_.read_rsrc_state(0, 0), 0);
}

// ----------------------------------------------------------------
// set_rsrc_state — threadwise mode
// ----------------------------------------------------------------

TEST_F(PipeResourceTest, SetRsrcState_SuccessWhenDifferent) {
    bool ok = pr_.set_rsrc_state(0, 0, 1);
    EXPECT_TRUE(ok);
    EXPECT_EQ(pr_.read_rsrc_state(0, 0), 1);
}

TEST_F(PipeResourceTest, SetRsrcState_FailWhenAlreadySame) {
    pr_.set_rsrc_state(0, 0, 1);
    bool ok = pr_.set_rsrc_state(0, 0, 1); // already 1 → false
    EXPECT_FALSE(ok);
    EXPECT_EQ(pr_.read_rsrc_state(0, 0), 1); // unchanged
}

TEST_F(PipeResourceTest, SetRsrcState_ThreadwiseIsolation) {
    pr_.set_rsrc_state(1, 2, 1);
    EXPECT_EQ(pr_.read_rsrc_state(1, 2), 1);
    // Other threads for same pipe unaffected
    EXPECT_EQ(pr_.read_rsrc_state(1, 0), 0);
    EXPECT_EQ(pr_.read_rsrc_state(1, 1), 0);
    EXPECT_EQ(pr_.read_rsrc_state(1, 3), 0);
}

TEST_F(PipeResourceTest, SetRsrcState_PipeIsolation) {
    pr_.set_rsrc_state(0, 0, 1);
    // Other pipes unaffected
    EXPECT_EQ(pr_.read_rsrc_state(1, 0), 0);
    EXPECT_EQ(pr_.read_rsrc_state(2, 0), 0);
}

TEST_F(PipeResourceTest, SetRsrcState_Toggle) {
    pr_.set_rsrc_state(0, 0, 1);
    EXPECT_TRUE(pr_.set_rsrc_state(0, 0, 0)); // back to 0
    EXPECT_EQ(pr_.read_rsrc_state(0, 0), 0);
}

// ----------------------------------------------------------------
// set_rsrc_state — global (non-threadwise) mode
// ----------------------------------------------------------------

TEST(PipeResourceGlobal, GlobalModeUpdatesAllThreads) {
    PipeResource pr(2, 3, false); // threadwise=false
    bool ok = pr.set_rsrc_state(0, 1, 1);
    EXPECT_TRUE(ok);
    // All threads for pipe 0 should be updated
    EXPECT_EQ(pr.read_rsrc_state(0, 0), 1);
    EXPECT_EQ(pr.read_rsrc_state(0, 1), 1);
    EXPECT_EQ(pr.read_rsrc_state(0, 2), 1);
    // Other pipe unaffected
    EXPECT_EQ(pr.read_rsrc_state(1, 0), 0);
}

// ----------------------------------------------------------------
// check_rsrc_state
// ----------------------------------------------------------------

TEST_F(PipeResourceTest, CheckRsrcState_ImmediateMatch) {
    // State=0, v=0, required=0 → done immediately
    auto res = pr_.check_rsrc_state(0, 0, 0, 0, 0);
    EXPECT_TRUE(res.done);
}

TEST_F(PipeResourceTest, CheckRsrcState_NoMatch) {
    // State=0, checking for v=1 → not done, consec reset to 0
    auto res = pr_.check_rsrc_state(0, 0, 1, 5, 3);
    EXPECT_FALSE(res.done);
    EXPECT_EQ(res.consec_count, 0);
}

TEST_F(PipeResourceTest, CheckRsrcState_ConsecutiveCount) {
    pr_.set_rsrc_state(0, 0, 1);
    // required=2 consecutive cycles of v=1
    // First call: prev_consec=0, 0 < 2 → not done, count=1
    auto r0 = pr_.check_rsrc_state(0, 0, 1, 0, 2);
    EXPECT_FALSE(r0.done);
    EXPECT_EQ(r0.consec_count, 1);

    // Second call: prev_consec=1, 1 < 2 → not done, count=2
    auto r1 = pr_.check_rsrc_state(0, 0, 1, r0.consec_count, 2);
    EXPECT_FALSE(r1.done);
    EXPECT_EQ(r1.consec_count, 2);

    // Third call: prev_consec=2, 2 >= 2 → done
    auto r2 = pr_.check_rsrc_state(0, 0, 1, r1.consec_count, 2);
    EXPECT_TRUE(r2.done);
}

TEST_F(PipeResourceTest, CheckRsrcState_ConsecResetOnMismatch) {
    pr_.set_rsrc_state(0, 0, 1);
    auto r0 = pr_.check_rsrc_state(0, 0, 1, 0, 3);
    EXPECT_EQ(r0.consec_count, 1);

    // State changes back to 0 — consecutive count resets
    pr_.set_rsrc_state(0, 0, 0);
    auto r1 = pr_.check_rsrc_state(0, 0, 1, r0.consec_count, 3);
    EXPECT_FALSE(r1.done);
    EXPECT_EQ(r1.consec_count, 0);
}

TEST_F(PipeResourceTest, CheckRsrcState_Required0_DefaultImmediateDone) {
    // Default required=0: any single matching cycle satisfies the condition
    auto res = pr_.check_rsrc_state(0, 0, 0);
    EXPECT_TRUE(res.done);
}
