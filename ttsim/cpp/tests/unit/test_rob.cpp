// SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "neosim/units/rob.hpp"
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

TEST(RobTest, StartsEmpty)
{
    Rob rob;
    EXPECT_TRUE(rob.empty());
    EXPECT_EQ(rob.size(), 0);
    EXPECT_EQ(rob.next_id(), 0u);
}

// ---------------------------------------------------------------------------
// append / next_id
// ---------------------------------------------------------------------------

TEST(RobTest, AppendReturnsSequentialIds)
{
    Rob rob;
    auto id0 = rob.append(make_ins("NOP", 0x1000));
    auto id1 = rob.append(make_ins("ADD", 0x1004));
    auto id2 = rob.append(make_ins("SUB", 0x1008));

    EXPECT_EQ(id0, 0u);
    EXPECT_EQ(id1, 1u);
    EXPECT_EQ(id2, 2u);
    EXPECT_EQ(rob.size(), 3);
    EXPECT_EQ(rob.next_id(), 3u);
}

TEST(RobTest, AppendSetsHeadId)
{
    Rob rob;
    auto id = rob.append(make_ins("NOP"));
    EXPECT_EQ(rob.head_id(), id);
}

// ---------------------------------------------------------------------------
// is_head
// ---------------------------------------------------------------------------

TEST(RobTest, IsHead_FirstEntry)
{
    Rob rob;
    auto id0 = rob.append(make_ins("A"));
    auto id1 = rob.append(make_ins("B"));

    EXPECT_TRUE(rob.is_head(id0));
    EXPECT_FALSE(rob.is_head(id1));
}

TEST(RobTest, IsHead_FalseWhenEmpty)
{
    Rob rob;
    EXPECT_FALSE(rob.is_head(0u));
    EXPECT_FALSE(rob.is_head(99u));
}

// ---------------------------------------------------------------------------
// pop_head
// ---------------------------------------------------------------------------

TEST(RobTest, PopHead_SucceedsWhenAtHead)
{
    Rob rob;
    auto id0 = rob.append(make_ins("A"));
    auto id1 = rob.append(make_ins("B"));

    EXPECT_TRUE(rob.pop_head(id0));
    EXPECT_EQ(rob.size(), 1);
    EXPECT_TRUE(rob.is_head(id1));
}

TEST(RobTest, PopHead_FailsWhenNotAtHead)
{
    Rob rob;
    rob.append(make_ins("A"));
    auto id1 = rob.append(make_ins("B"));

    // id1 is second, not head
    EXPECT_FALSE(rob.pop_head(id1));
    EXPECT_EQ(rob.size(), 2);
}

TEST(RobTest, PopHead_FailsWhenEmpty)
{
    Rob rob;
    EXPECT_FALSE(rob.pop_head(0u));
}

TEST(RobTest, PopHead_BecomeEmptyAfterLast)
{
    Rob rob;
    auto id = rob.append(make_ins("X"));
    EXPECT_TRUE(rob.pop_head(id));
    EXPECT_TRUE(rob.empty());
}

// ---------------------------------------------------------------------------
// remove (non-blocking removal from any position)
// ---------------------------------------------------------------------------

TEST(RobTest, Remove_FromHead)
{
    Rob rob;
    auto id0 = rob.append(make_ins("A"));
    auto id1 = rob.append(make_ins("B"));
    rob.remove(id0);
    EXPECT_EQ(rob.size(), 1);
    EXPECT_TRUE(rob.is_head(id1));
}

TEST(RobTest, Remove_FromMiddle)
{
    Rob rob;
    auto id0 = rob.append(make_ins("A"));
    auto id1 = rob.append(make_ins("B"));
    auto id2 = rob.append(make_ins("C"));
    rob.remove(id1);
    EXPECT_EQ(rob.size(), 2);
    EXPECT_TRUE(rob.is_head(id0));
    EXPECT_NE(rob.find(id2), nullptr);
}

TEST(RobTest, Remove_FromTail)
{
    Rob rob;
    auto id0 = rob.append(make_ins("A"));
    auto id1 = rob.append(make_ins("B"));
    rob.remove(id1);
    EXPECT_EQ(rob.size(), 1);
    EXPECT_TRUE(rob.is_head(id0));
}

TEST(RobTest, Remove_NoOpWhenNotFound)
{
    Rob rob;
    rob.append(make_ins("A"));
    rob.remove(99u);  // doesn't exist — silent no-op
    EXPECT_EQ(rob.size(), 1);
}

TEST(RobTest, Remove_NoOpOnEmpty)
{
    Rob rob;
    rob.remove(0u);  // should not crash
    EXPECT_TRUE(rob.empty());
}

// ---------------------------------------------------------------------------
// find
// ---------------------------------------------------------------------------

TEST(RobTest, Find_ReturnsCorrectInstruction)
{
    Rob rob;
    auto ins_a = make_ins("A", 0x1000);
    auto ins_b = make_ins("B", 0x2000);
    auto id0 = rob.append(ins_a);
    auto id1 = rob.append(ins_b);

    auto found_a = rob.find(id0);
    ASSERT_NE(found_a, nullptr);
    EXPECT_EQ(found_a->get_op(), "A");

    auto found_b = rob.find(id1);
    ASSERT_NE(found_b, nullptr);
    EXPECT_EQ(found_b->get_op(), "B");
}

TEST(RobTest, Find_ReturnsNullptrWhenMissing)
{
    Rob rob;
    rob.append(make_ins("A"));
    EXPECT_EQ(rob.find(99u), nullptr);
}

TEST(RobTest, Find_ReturnsNullptrOnEmpty)
{
    Rob rob;
    EXPECT_EQ(rob.find(0u), nullptr);
}

// ---------------------------------------------------------------------------
// Sequential pop_head drains the ROB in order
// ---------------------------------------------------------------------------

TEST(RobTest, SequentialPopHead_DrainsFifoOrder)
{
    Rob rob;
    std::vector<uint32_t> ids;
    for (int i = 0; i < 5; ++i) {
        ids.push_back(rob.append(make_ins("I")));
    }
    EXPECT_EQ(rob.size(), 5);

    for (int i = 0; i < 5; ++i) {
        EXPECT_TRUE(rob.is_head(ids[static_cast<std::size_t>(i)]));
        EXPECT_TRUE(rob.pop_head(ids[static_cast<std::size_t>(i)]));
    }
    EXPECT_TRUE(rob.empty());
}

// ---------------------------------------------------------------------------
// IdCounter continues after pop/remove
// ---------------------------------------------------------------------------

TEST(RobTest, IdCounterMonotonicallyIncreasing)
{
    Rob rob;
    auto id0 = rob.append(make_ins("A"));
    rob.pop_head(id0);
    auto id1 = rob.append(make_ins("B"));
    EXPECT_GT(id1, id0);
}
