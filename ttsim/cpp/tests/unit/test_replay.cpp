// SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "neosim/units/replay.hpp"
#include "neosim/isa/instruction.hpp"

#include <gtest/gtest.h>

using namespace neosim::units;
using namespace neosim::isa;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static InstrPtr make_ins(const std::string& op, uint32_t pc = 0x1000u)
{
    auto ins = std::make_shared<Instruction>();
    ins->set_op(op);
    ins->program_counter = pc;
    return ins;
}

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

TEST(ReplayStateTest, StartsInPassthrough)
{
    ReplayState rs;
    EXPECT_EQ(rs.mode(), ReplayState::Mode::PASSTHROUGH);
    EXPECT_EQ(rs.prev_mode(), ReplayState::Mode::PASSTHROUGH);
    EXPECT_TRUE(rs.in_passthrough());
}

// ---------------------------------------------------------------------------
// update_mode: LOAD
// ---------------------------------------------------------------------------

TEST(ReplayStateTest, UpdateMode_Load)
{
    ReplayState rs;
    rs.update_mode(/*load=*/1, /*exec_while_loading=*/0,
                   /*start_idx=*/0, /*len=*/3);
    EXPECT_EQ(rs.mode(), ReplayState::Mode::LOAD);
    EXPECT_EQ(rs.prev_mode(), ReplayState::Mode::PASSTHROUGH);
    EXPECT_EQ(rs.start_idx(), 0);
    EXPECT_EQ(rs.replay_len(), 3);
}

// ---------------------------------------------------------------------------
// update_mode: LOAD_EXECUTE
// ---------------------------------------------------------------------------

TEST(ReplayStateTest, UpdateMode_LoadExecute)
{
    ReplayState rs;
    rs.update_mode(/*load=*/1, /*exec_while_loading=*/1,
                   /*start_idx=*/2, /*len=*/4);
    EXPECT_EQ(rs.mode(), ReplayState::Mode::LOAD_EXECUTE);
    EXPECT_EQ(rs.start_idx(), 2);
    EXPECT_EQ(rs.replay_len(), 4);
}

// ---------------------------------------------------------------------------
// update_mode: EXECUTE
// ---------------------------------------------------------------------------

TEST(ReplayStateTest, UpdateMode_Execute)
{
    ReplayState rs;
    rs.update_mode(/*load=*/0, /*exec_while_loading=*/0,
                   /*start_idx=*/0, /*len=*/2);
    EXPECT_EQ(rs.mode(), ReplayState::Mode::EXECUTE);
}

// ---------------------------------------------------------------------------
// load_replay_list: stores instructions and resets on full window
// ---------------------------------------------------------------------------

TEST(ReplayStateTest, LoadReplayList_SingleEntry_ResetsToPassthrough)
{
    ReplayState rs;
    rs.update_mode(1, 0, /*start_idx=*/0, /*len=*/1);

    auto ins_a = make_ins("A");
    rs.load_replay_list(ins_a);

    // Window of 1 — should reset to passthrough after loading the single entry.
    EXPECT_TRUE(rs.in_passthrough());
}

TEST(ReplayStateTest, LoadReplayList_MultiEntry_ResetsAfterLastEntry)
{
    ReplayState rs;
    rs.update_mode(1, 0, /*start_idx=*/0, /*len=*/3);

    rs.load_replay_list(make_ins("A"));
    EXPECT_EQ(rs.mode(), ReplayState::Mode::LOAD);

    rs.load_replay_list(make_ins("B"));
    EXPECT_EQ(rs.mode(), ReplayState::Mode::LOAD);

    rs.load_replay_list(make_ins("C"));
    // After loading all 3, reset to passthrough
    EXPECT_TRUE(rs.in_passthrough());
}

TEST(ReplayStateTest, LoadReplayList_WithOffset)
{
    ReplayState rs;
    // Use start_idx=4, len=2 → slots 4 and 5
    rs.update_mode(1, 0, /*start_idx=*/4, /*len=*/2);

    rs.load_replay_list(make_ins("X"));
    EXPECT_EQ(rs.mode(), ReplayState::Mode::LOAD);

    rs.load_replay_list(make_ins("Y"));
    EXPECT_TRUE(rs.in_passthrough());
}

// ---------------------------------------------------------------------------
// exec_replay_list: returns instructions in order and resets
// ---------------------------------------------------------------------------

TEST(ReplayStateTest, ExecReplayList_ReturnsSingleEntry)
{
    ReplayState rs;
    // Load phase
    rs.update_mode(1, 0, /*start_idx=*/0, /*len=*/1);
    auto ins_a = make_ins("ALPHA", 0x2000);
    rs.load_replay_list(ins_a);
    EXPECT_TRUE(rs.in_passthrough());

    // Execute phase
    rs.update_mode(0, 0, /*start_idx=*/0, /*len=*/1);
    EXPECT_EQ(rs.mode(), ReplayState::Mode::EXECUTE);

    auto got = rs.exec_replay_list();
    ASSERT_NE(got, nullptr);
    EXPECT_EQ(got->get_op(), "ALPHA");

    // After consuming all, reset to passthrough
    EXPECT_TRUE(rs.in_passthrough());
}

TEST(ReplayStateTest, ExecReplayList_ReturnsMultipleInOrder)
{
    ReplayState rs;
    // Load 3 instructions
    rs.update_mode(1, 0, /*start_idx=*/0, /*len=*/3);
    rs.load_replay_list(make_ins("OP0", 0x1000));
    rs.load_replay_list(make_ins("OP1", 0x1004));
    rs.load_replay_list(make_ins("OP2", 0x1008));

    // Execute
    rs.update_mode(0, 0, /*start_idx=*/0, /*len=*/3);

    auto g0 = rs.exec_replay_list();
    ASSERT_NE(g0, nullptr);
    EXPECT_EQ(g0->get_op(), "OP0");

    auto g1 = rs.exec_replay_list();
    ASSERT_NE(g1, nullptr);
    EXPECT_EQ(g1->get_op(), "OP1");

    auto g2 = rs.exec_replay_list();
    ASSERT_NE(g2, nullptr);
    EXPECT_EQ(g2->get_op(), "OP2");

    EXPECT_TRUE(rs.in_passthrough());
}

TEST(ReplayStateTest, ExecReplayList_ReturnsNullptrWhenNotInExecuteMode)
{
    ReplayState rs;  // starts in PASSTHROUGH
    EXPECT_EQ(rs.exec_replay_list(), nullptr);

    rs.update_mode(1, 0, 0, 2);  // LOAD mode
    EXPECT_EQ(rs.exec_replay_list(), nullptr);
}

// ---------------------------------------------------------------------------
// LOAD_EXECUTE: both loading and executing from the same call site
// ---------------------------------------------------------------------------

TEST(ReplayStateTest, LoadExecuteMode)
{
    ReplayState rs;
    rs.update_mode(1, 1, /*start_idx=*/0, /*len=*/2);
    EXPECT_EQ(rs.mode(), ReplayState::Mode::LOAD_EXECUTE);

    // Caller loads two instructions (in LOAD_EXECUTE mode the instruction is
    // both stored and executed; after loading all len_ entries, resets to PASS).
    rs.load_replay_list(make_ins("X0"));
    EXPECT_EQ(rs.mode(), ReplayState::Mode::LOAD_EXECUTE);

    rs.load_replay_list(make_ins("X1"));
    EXPECT_TRUE(rs.in_passthrough());
}

// ---------------------------------------------------------------------------
// prev_mode tracking
// ---------------------------------------------------------------------------

TEST(ReplayStateTest, PrevModeUpdatedOnTransition)
{
    ReplayState rs;
    rs.update_mode(1, 0, 0, 1);
    EXPECT_EQ(rs.prev_mode(), ReplayState::Mode::PASSTHROUGH);
    EXPECT_EQ(rs.mode(),      ReplayState::Mode::LOAD);

    rs.update_mode(0, 0, 0, 1);
    EXPECT_EQ(rs.prev_mode(), ReplayState::Mode::LOAD);
    EXPECT_EQ(rs.mode(),      ReplayState::Mode::EXECUTE);
}

// ---------------------------------------------------------------------------
// Replay with non-zero start_idx + exec
// ---------------------------------------------------------------------------

TEST(ReplayStateTest, ExecReplayList_NonZeroStartIdx)
{
    ReplayState rs;
    rs.update_mode(1, 0, /*start_idx=*/5, /*len=*/2);
    rs.load_replay_list(make_ins("M", 0xABC0));
    rs.load_replay_list(make_ins("N", 0xABC4));

    rs.update_mode(0, 0, /*start_idx=*/5, /*len=*/2);

    auto g0 = rs.exec_replay_list();
    ASSERT_NE(g0, nullptr);
    EXPECT_EQ(g0->get_op(), "M");

    auto g1 = rs.exec_replay_list();
    ASSERT_NE(g1, nullptr);
    EXPECT_EQ(g1->get_op(), "N");

    EXPECT_TRUE(rs.in_passthrough());
}
