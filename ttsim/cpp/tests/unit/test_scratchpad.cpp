// SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include "neosim/units/scratchpad.hpp"

using namespace neosim::units;

// Reset the global request-ID counter before each test so that req_ids are
// predictable within a test case.
class MemReqTest : public ::testing::Test {
protected:
    void SetUp() override { MemReq::reset_counter(0); }
};

class ScratchpadTest : public ::testing::Test {
protected:
    void SetUp() override { MemReq::reset_counter(0); }

    Scratchpad::Config make_cfg(int lat_rd = 10, int num_cores = 1, int num_engines = 4) {
        return {lat_rd, 1, num_cores, num_engines, false};
    }
};

// ================================================================
// MemReq tests
// ================================================================

// ----------------------------------------------------------------
// Construction and req_id auto-increment
// ----------------------------------------------------------------

TEST_F(MemReqTest, ReqId_AutoIncrement) {
    MemReq r0(MemOp::READ,  0x1000, 128);
    MemReq r1(MemOp::WRITE, 0x2000, 64);
    EXPECT_EQ(r0.req_id(), 0u);
    EXPECT_EQ(r1.req_id(), 1u);
}

TEST_F(MemReqTest, ReqId_ResetCounter) {
    MemReq r0(MemOp::READ, 0, 0);
    EXPECT_EQ(r0.req_id(), 0u);
    MemReq::reset_counter(100);
    MemReq r1(MemOp::READ, 0, 0);
    EXPECT_EQ(r1.req_id(), 100u);
}

TEST_F(MemReqTest, ConstructorFields) {
    MemReq r(MemOp::WRITE, 0xDEAD, 256);
    EXPECT_EQ(r.op(),    MemOp::WRITE);
    EXPECT_EQ(r.addr(),  0xDEADu);
    EXPECT_EQ(r.bytes(), 256u);
}

// ----------------------------------------------------------------
// Default sentinel values
// ----------------------------------------------------------------

TEST_F(MemReqTest, DefaultThreadId_IsMinusOne) {
    MemReq r(MemOp::READ, 0, 0);
    EXPECT_EQ(r.thread_id(), -1);
}

TEST_F(MemReqTest, DefaultCoreId_IsMinusOne) {
    MemReq r(MemOp::READ, 0, 0);
    EXPECT_EQ(r.core_id(), -1);
}

TEST_F(MemReqTest, DefaultPipeId_IsMinusOne) {
    MemReq r(MemOp::READ, 0, 0);
    EXPECT_EQ(r.pipe_id(), -1);
}

TEST_F(MemReqTest, OptionalFields_DefaultUnset) {
    MemReq r(MemOp::READ, 0, 0);
    EXPECT_FALSE(r.ins_id().has_value());
    EXPECT_FALSE(r.src().has_value());
    EXPECT_FALSE(r.target().has_value());
    EXPECT_FALSE(r.src_pipe().has_value());
}

TEST_F(MemReqTest, ParentChildIds_DefaultEmpty) {
    MemReq r(MemOp::READ, 0, 0);
    EXPECT_TRUE(r.parent_req_ids().empty());
    EXPECT_TRUE(r.child_req_ids().empty());
}

// ----------------------------------------------------------------
// Mutators round-trip
// ----------------------------------------------------------------

TEST_F(MemReqTest, SetGetOp) {
    MemReq r(MemOp::READ, 0, 0);
    r.set_op(MemOp::WRITE);
    EXPECT_EQ(r.op(), MemOp::WRITE);
}

TEST_F(MemReqTest, SetGetAddr) {
    MemReq r(MemOp::READ, 0, 0);
    r.set_addr(0xABCD);
    EXPECT_EQ(r.addr(), 0xABCDu);
}

TEST_F(MemReqTest, SetGetBytes) {
    MemReq r(MemOp::READ, 0, 0);
    r.set_bytes(512);
    EXPECT_EQ(r.bytes(), 512u);
}

TEST_F(MemReqTest, SetGetThreadId) {
    MemReq r(MemOp::READ, 0, 0);
    r.set_thread_id(2);
    EXPECT_EQ(r.thread_id(), 2);
}

TEST_F(MemReqTest, SetGetCoreId) {
    MemReq r(MemOp::READ, 0, 0);
    r.set_core_id(1);
    EXPECT_EQ(r.core_id(), 1);
}

TEST_F(MemReqTest, SetGetPipeId) {
    MemReq r(MemOp::READ, 0, 0);
    r.set_pipe_id(3);
    EXPECT_EQ(r.pipe_id(), 3);
}

TEST_F(MemReqTest, SetGetInsId) {
    MemReq r(MemOp::READ, 0, 0);
    r.set_ins_id(42);
    ASSERT_TRUE(r.ins_id().has_value());
    EXPECT_EQ(*r.ins_id(), 42u);
}

TEST_F(MemReqTest, SetGetSrc) {
    MemReq r(MemOp::READ, 0, 0);
    r.set_src("UNPACKER");
    ASSERT_TRUE(r.src().has_value());
    EXPECT_EQ(*r.src(), "UNPACKER");
}

TEST_F(MemReqTest, SetGetTarget) {
    MemReq r(MemOp::READ, 0, 0);
    r.set_target("L1");
    ASSERT_TRUE(r.target().has_value());
    EXPECT_EQ(*r.target(), "L1");
}

TEST_F(MemReqTest, SetGetSrcPipe) {
    MemReq r(MemOp::READ, 0, 0);
    r.set_src_pipe("COMPUTE_0");
    ASSERT_TRUE(r.src_pipe().has_value());
    EXPECT_EQ(*r.src_pipe(), "COMPUTE_0");
}

TEST_F(MemReqTest, SetParentReqIds) {
    MemReq r(MemOp::READ, 0, 0);
    r.set_parent_req_ids({1, 2, 3});
    ASSERT_EQ(r.parent_req_ids().size(), 3u);
    EXPECT_EQ(r.parent_req_ids()[0], 1u);
    EXPECT_EQ(r.parent_req_ids()[2], 3u);
}

TEST_F(MemReqTest, SetChildReqIds) {
    MemReq r(MemOp::READ, 0, 0);
    r.set_child_req_ids({10, 20});
    ASSERT_EQ(r.child_req_ids().size(), 2u);
    EXPECT_EQ(r.child_req_ids()[1], 20u);
}

TEST_F(MemReqTest, AddParentAndChildIds) {
    MemReq r(MemOp::READ, 0, 0);
    r.add_parent_req_id(5);
    r.add_child_req_id(6);
    r.add_child_req_id(7);
    EXPECT_EQ(r.parent_req_ids().size(), 1u);
    EXPECT_EQ(r.child_req_ids().size(),  2u);
}

// ================================================================
// Scratchpad tests
// ================================================================

// ----------------------------------------------------------------
// Construction
// ----------------------------------------------------------------

TEST_F(ScratchpadTest, Construction_NumPorts) {
    Scratchpad sp({10, 1, 2, 4, false}); // 2 cores × 4 engines = 8 ports
    EXPECT_EQ(sp.num_ports(), 8);
}

TEST_F(ScratchpadTest, Construction_Latencies) {
    Scratchpad sp({10, 1, 1, 1, false});
    EXPECT_EQ(sp.latency_rd(), 10);
    EXPECT_EQ(sp.latency_wr(), 1);
}

TEST_F(ScratchpadTest, Construction_NoInFlight) {
    Scratchpad sp(make_cfg());
    EXPECT_EQ(sp.num_in_flight(), 0);
}

// ----------------------------------------------------------------
// submit — returns correct latency
// ----------------------------------------------------------------

TEST_F(ScratchpadTest, Submit_ReadReturnsLatencyRd) {
    Scratchpad sp(make_cfg(10));
    MemReq req(MemOp::READ, 0x1000, 128);
    int lat = sp.submit(req);
    EXPECT_EQ(lat, 10);
}

TEST_F(ScratchpadTest, Submit_WriteReturnsLatencyWr) {
    Scratchpad sp(make_cfg());
    MemReq req(MemOp::WRITE, 0x2000, 64);
    int lat = sp.submit(req);
    EXPECT_EQ(lat, 1); // hardcoded write latency
}

TEST_F(ScratchpadTest, Submit_CustomReadLatency) {
    Scratchpad sp(make_cfg(15));
    MemReq req(MemOp::READ, 0, 0);
    EXPECT_EQ(sp.submit(req), 15);
}

// ----------------------------------------------------------------
// submit — request is now in-flight
// ----------------------------------------------------------------

TEST_F(ScratchpadTest, Submit_RequestIsInFlight) {
    Scratchpad sp(make_cfg());
    MemReq req(MemOp::READ, 0x1000, 64);
    uint32_t id = req.req_id();
    sp.submit(req);
    EXPECT_TRUE(sp.is_in_flight(id));
    EXPECT_EQ(sp.num_in_flight(), 1);
}

TEST_F(ScratchpadTest, Submit_MultipleRequestsTracked) {
    Scratchpad sp(make_cfg());
    MemReq r0(MemOp::READ,  0x1000, 64);
    MemReq r1(MemOp::WRITE, 0x2000, 32);
    MemReq r2(MemOp::READ,  0x3000, 128);
    sp.submit(r0);
    sp.submit(r1);
    sp.submit(r2);
    EXPECT_EQ(sp.num_in_flight(), 3);
    EXPECT_TRUE(sp.is_in_flight(r0.req_id()));
    EXPECT_TRUE(sp.is_in_flight(r1.req_id()));
    EXPECT_TRUE(sp.is_in_flight(r2.req_id()));
}

TEST_F(ScratchpadTest, NotInFlight_BeforeSubmit) {
    Scratchpad sp(make_cfg());
    EXPECT_FALSE(sp.is_in_flight(999u));
}

// ----------------------------------------------------------------
// complete — removes from tracker, returns request
// ----------------------------------------------------------------

TEST_F(ScratchpadTest, Complete_RemovesFromTracker) {
    Scratchpad sp(make_cfg());
    MemReq req(MemOp::READ, 0x1000, 64);
    uint32_t id = req.req_id();
    sp.submit(req);
    sp.complete(id);
    EXPECT_FALSE(sp.is_in_flight(id));
    EXPECT_EQ(sp.num_in_flight(), 0);
}

TEST_F(ScratchpadTest, Complete_ReturnsCorrectRequest) {
    Scratchpad sp(make_cfg());
    MemReq req(MemOp::READ, 0xABCD, 256);
    req.set_thread_id(2);
    req.set_core_id(1);
    uint32_t id = req.req_id();
    sp.submit(req);
    MemReq done = sp.complete(id);
    EXPECT_EQ(done.req_id(),   id);
    EXPECT_EQ(done.addr(),     0xABCDu);
    EXPECT_EQ(done.bytes(),    256u);
    EXPECT_EQ(done.thread_id(), 2);
    EXPECT_EQ(done.core_id(),   1);
}

TEST_F(ScratchpadTest, Complete_PartiallyRemoves) {
    Scratchpad sp(make_cfg());
    MemReq r0(MemOp::READ,  0x1000, 64);
    MemReq r1(MemOp::WRITE, 0x2000, 32);
    sp.submit(r0);
    sp.submit(r1);
    sp.complete(r0.req_id());
    EXPECT_FALSE(sp.is_in_flight(r0.req_id()));
    EXPECT_TRUE(sp.is_in_flight(r1.req_id()));
    EXPECT_EQ(sp.num_in_flight(), 1);
}

#ifndef NDEBUG
TEST_F(ScratchpadTest, Complete_UnknownIdAsserts) {
    Scratchpad sp(make_cfg());
    EXPECT_DEATH(sp.complete(999u), "");
}

TEST_F(ScratchpadTest, Submit_DuplicateIdAsserts) {
    Scratchpad sp(make_cfg());
    MemReq req(MemOp::READ, 0x1000, 64);
    sp.submit(req);
    // Submitting same req object again has the same req_id → should assert
    EXPECT_DEATH(sp.submit(req), "");
}
#endif

// ----------------------------------------------------------------
// Full lifecycle: submit → complete
// ----------------------------------------------------------------

TEST_F(ScratchpadTest, FullLifecycle_ReadRequest) {
    Scratchpad sp(make_cfg(10));

    MemReq req(MemOp::READ, 0x5000, 128);
    req.set_thread_id(1);
    req.set_core_id(0);
    req.set_src_pipe("UNPACKER_0");
    req.set_target("L1");

    uint32_t id  = req.req_id();
    int      lat = sp.submit(req);

    EXPECT_EQ(lat, 10);
    EXPECT_TRUE(sp.is_in_flight(id));

    // Simulate: after `lat` cycles the response event fires
    MemReq done = sp.complete(id);

    EXPECT_EQ(done.req_id(), id);
    EXPECT_FALSE(sp.is_in_flight(id));
    EXPECT_EQ(*done.src_pipe(), "UNPACKER_0");
}

TEST_F(ScratchpadTest, FullLifecycle_WriteRequest) {
    Scratchpad sp(make_cfg());

    MemReq req(MemOp::WRITE, 0x8000, 64);
    req.set_thread_id(0);

    int lat = sp.submit(req);
    EXPECT_EQ(lat, 1); // write latency is always 1

    MemReq done = sp.complete(req.req_id());
    EXPECT_EQ(done.op(), MemOp::WRITE);
}

// ----------------------------------------------------------------
// num_ports reflects config
// ----------------------------------------------------------------

TEST_F(ScratchpadTest, NumPorts_SingleCore) {
    Scratchpad sp({10, 1, 1, 6, false});
    EXPECT_EQ(sp.num_ports(), 6); // 1 × 6
}

TEST_F(ScratchpadTest, NumPorts_MultiCore) {
    Scratchpad sp({10, 1, 4, 6, false});
    EXPECT_EQ(sp.num_ports(), 24); // 4 × 6
}
