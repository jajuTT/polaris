#include "neosim/isa/instruction.hpp"

#include <gtest/gtest.h>

using namespace neosim::isa;

// ---------------------------------------------------------------------------
// get_num_bytes_from_data_format
// ---------------------------------------------------------------------------

TEST(GetNumBytesFromDataFormat, Float32) {
    EXPECT_EQ(get_num_bytes_from_data_format(0), 4);
}

TEST(GetNumBytesFromDataFormat, Tf32) {
    EXPECT_EQ(get_num_bytes_from_data_format(4), 2);
}

TEST(GetNumBytesFromDataFormat, Float16) {
    EXPECT_EQ(get_num_bytes_from_data_format(1), 2);
}

TEST(GetNumBytesFromDataFormat, Int32) {
    EXPECT_EQ(get_num_bytes_from_data_format(8), 4);
}

TEST(GetNumBytesFromDataFormat, Int8) {
    EXPECT_EQ(get_num_bytes_from_data_format(14), 1);
}

TEST(GetNumBytesFromDataFormat, Fp8R) {
    EXPECT_EQ(get_num_bytes_from_data_format(10), 1);
}

TEST(GetNumBytesFromDataFormat, Uint16) {
    EXPECT_EQ(get_num_bytes_from_data_format(130), 2);
}

TEST(GetNumBytesFromDataFormat, UnknownFormat255) {
    // 255 is the UNPACR0_STRIDE sentinel — should return 2 (Float16 default)
    EXPECT_EQ(get_num_bytes_from_data_format(255), 2);
}

TEST(GetNumBytesFromDataFormat, InvalidFormatThrows) {
    EXPECT_THROW(get_num_bytes_from_data_format(99), std::runtime_error);
}

// ---------------------------------------------------------------------------
// Default construction & field initialisation
// ---------------------------------------------------------------------------

TEST(InstructionDefaultConstruct, FieldsInitialisedToSentinels) {
    Instruction ins;
    EXPECT_EQ(ins.get_ins_id(),    0u);
    EXPECT_EQ(ins.get_core_id(),   0u);
    EXPECT_EQ(ins.get_thread_id(), 0u);
    EXPECT_EQ(ins.get_pipe_delay(), -1);
    EXPECT_TRUE(ins.get_ex_pipe().empty());
    EXPECT_EQ(ins.get_num_datums(), 0);
    EXPECT_FALSE(ins.get_src_format().has_value());
    EXPECT_FALSE(ins.get_dst_format().has_value());
    EXPECT_FALSE(ins.get_pipes_thread_id().has_value());
    EXPECT_TRUE(ins.get_src_pipes().empty());
    EXPECT_TRUE(ins.get_dst_pipes().empty());
    EXPECT_TRUE(ins.get_src_int().empty());
    EXPECT_TRUE(ins.get_dst_int().empty());
    EXPECT_TRUE(ins.get_imm().empty());
    EXPECT_TRUE(ins.get_attr().empty());
}

TEST(InstructionDefaultConstruct, PredicatesFalse) {
    Instruction ins;
    EXPECT_FALSE(ins.is_tt());
    EXPECT_FALSE(ins.is_mop());
    EXPECT_FALSE(ins.is_replay());
}

// ---------------------------------------------------------------------------
// Converting constructor from decoded_instruction
// ---------------------------------------------------------------------------

TEST(InstructionConvert, CopiesBaseFields) {
    ttdecode::decode::decoded_instruction di;
    di.word             = 0xDEADBEEF;
    di.program_counter  = 0x1000;
    di.mnemonic         = "ADDI";

    Instruction ins(di);
    EXPECT_EQ(ins.word, 0xDEADBEEFu);
    EXPECT_EQ(ins.get_addr(), 0x1000u);
    EXPECT_EQ(ins.get_op(), "ADDI");
}

// ---------------------------------------------------------------------------
// Setter / getter round-trips
// ---------------------------------------------------------------------------

TEST(InstructionSetGet, IdentityFields) {
    Instruction ins;
    ins.set_ins_id(42u);
    ins.set_core_id(1u);
    ins.set_thread_id(3u);
    EXPECT_EQ(ins.get_ins_id(),    42u);
    EXPECT_EQ(ins.get_core_id(),   1u);
    EXPECT_EQ(ins.get_thread_id(), 3u);
}

TEST(InstructionSetGet, OpAndExPipe) {
    Instruction ins;
    ins.set_op("MOP");
    ins.set_ex_pipe("MATH");
    EXPECT_EQ(ins.get_op(), "MOP");
    EXPECT_EQ(ins.get_ex_pipe(), "MATH");
}

TEST(InstructionSetGet, PipeDelay) {
    Instruction ins;
    ins.set_pipe_delay(5);
    EXPECT_EQ(ins.get_pipe_delay(), 5);
}

TEST(InstructionSetGet, SrcDstInt) {
    Instruction ins;
    ins.set_src_int({1, 2, 3});
    ins.set_dst_int({7});
    EXPECT_EQ(ins.get_src_int(), (std::vector<int>{1, 2, 3}));
    EXPECT_EQ(ins.get_dst_int(), (std::vector<int>{7}));
}

TEST(InstructionSetGet, Immediates) {
    Instruction ins;
    ins.set_imm({0xFF, 0x10});
    EXPECT_EQ(ins.get_imm(), (std::vector<int>{0xFF, 0x10}));
}

TEST(InstructionSetGet, Attributes) {
    Instruction ins;
    ins.set_attr({{"dest_pulse_last", 3}, {"foo", 1}});
    auto attrs = ins.get_attr();
    EXPECT_EQ(attrs.at("dest_pulse_last"), 3);
    EXPECT_EQ(attrs.at("foo"), 1);
}

TEST(InstructionSetGet, DataFormat) {
    Instruction ins;
    ins.set_src_format(1);  // Float16
    ins.set_dst_format(0);  // Float32
    ins.set_num_datums(16);
    EXPECT_EQ(ins.get_src_size(), 2 * 16);  // 32 bytes
    EXPECT_EQ(ins.get_dst_size(), 4 * 16);  // 64 bytes
}

TEST(InstructionSetGet, Pipes) {
    Instruction ins;
    ins.set_src_pipes({"UNPACKER0", "UNPACKER1"});
    ins.set_dst_pipes({"MATH"});
    ins.set_pipes_thread_id(2u);
    EXPECT_EQ(ins.get_src_pipes(), (std::vector<std::string>{"UNPACKER0", "UNPACKER1"}));
    EXPECT_EQ(ins.get_dst_pipes(), (std::vector<std::string>{"MATH"}));
    EXPECT_EQ(ins.get_pipes_thread_id(), 2u);
}

TEST(InstructionSetGet, MemInfo) {
    Instruction ins;
    ins.set_mem_info("read_bytes", 128);
    EXPECT_EQ(ins.get_mem_info("read_bytes"), 128);
    ins.incr_mem_info("read_bytes", 64);
    EXPECT_EQ(ins.get_mem_info("read_bytes"), 192);
}

TEST(InstructionSetGet, EventInfo) {
    Instruction ins;
    ins.set_event_info("tag", "start");
    EXPECT_EQ(ins.get_event_info("tag"), "start");
}

// ---------------------------------------------------------------------------
// Sync mask boundary conditions
// ---------------------------------------------------------------------------

TEST(InstructionMasks, VldUpdMaskHasGet) {
    Instruction ins;
    EXPECT_FALSE(ins.has_vld_upd_mask(0));
    ins.set_vld_upd_mask({{0, 0b1010}, {1, 0b0101}});
    EXPECT_TRUE(ins.has_vld_upd_mask(0));
    EXPECT_TRUE(ins.has_vld_upd_mask(1));
    EXPECT_FALSE(ins.has_vld_upd_mask(2));
    EXPECT_EQ(ins.get_vld_upd_mask(0), 0b1010);
    EXPECT_EQ(ins.get_vld_upd_mask(1), 0b0101);
}

TEST(InstructionMasks, BankUpdMaskHasGet) {
    Instruction ins;
    EXPECT_FALSE(ins.has_bank_upd_mask(3));
    ins.set_bank_upd_mask({{3, 7}});
    EXPECT_TRUE(ins.has_bank_upd_mask(3));
    EXPECT_EQ(ins.get_bank_upd_mask(3), 7);
}

TEST(InstructionMasks, CondChkVldUpdBoundary) {
    Instruction ins;
    // Empty map → has returns false.
    EXPECT_FALSE(ins.has_cond_chk_vld_upd(0, 0));

    // Populate with IGNORE sentinel — has should still return false.
    const int IGNORE = static_cast<int>(ValueStatus::IGNORE);
    ins.set_cond_chk_vld_upd({{0, {{0, IGNORE}, {1, 1}}}});
    EXPECT_FALSE(ins.has_cond_chk_vld_upd(0, 0));  // IGNORE → false
    EXPECT_TRUE(ins.has_cond_chk_vld_upd(0, 1));   // non-IGNORE → true
    EXPECT_EQ(ins.get_cond_chk_vld_upd(0, 1), 1);
}

TEST(InstructionMasks, CondWriVldUpdBoundary) {
    Instruction ins;
    EXPECT_FALSE(ins.has_cond_wri_vld_upd(1, 2));

    ins.set_cond_wri_vld_upd({{1, {{2, 5}}}});
    EXPECT_TRUE(ins.has_cond_wri_vld_upd(1, 2));
    EXPECT_EQ(ins.get_cond_wri_vld_upd(1, 2), 5);
}

TEST(InstructionMasks, PipeBankCtrl) {
    Instruction ins;
    EXPECT_FALSE(ins.has_pipe_bank_ctrl(0));
    ins.set_pipe_bank_ctrl({{0, 3}, {2, 1}});
    EXPECT_TRUE(ins.has_pipe_bank_ctrl(0));
    EXPECT_EQ(ins.get_pipe_bank_ctrl(0), 3);
    EXPECT_EQ(ins.get_pipe_bank_ctrl(2), 1);
}

// ---------------------------------------------------------------------------
// Predicates: is_mop, is_replay, is_tt
// ---------------------------------------------------------------------------

TEST(InstructionPredicates, IsMop) {
    Instruction ins;
    ins.set_op("MOP");
    EXPECT_TRUE(ins.is_mop());
    ins.set_op("ADDI");
    EXPECT_FALSE(ins.is_mop());
}

TEST(InstructionPredicates, IsReplay) {
    Instruction ins;
    ins.set_op("REPLAY");
    EXPECT_TRUE(ins.is_replay());
    ins.set_op("MOP");
    EXPECT_FALSE(ins.is_replay());
}

TEST(InstructionPredicates, IsTT) {
    Instruction ins;
    EXPECT_FALSE(ins.is_tt());
    ins.kind = ttdecode::isa::instruction_kind::ttqs;
    EXPECT_TRUE(ins.is_tt());
    ins.kind = ttdecode::isa::instruction_kind::ttwh;
    EXPECT_TRUE(ins.is_tt());
    ins.kind = ttdecode::isa::instruction_kind::ttbh;
    EXPECT_TRUE(ins.is_tt());
    ins.kind = ttdecode::isa::instruction_kind::rv32;
    EXPECT_FALSE(ins.is_tt());
}

// ---------------------------------------------------------------------------
// get_context
// ---------------------------------------------------------------------------

TEST(InstructionGetContext, SETDVALID_UsesThreadId) {
    Instruction ins;
    ins.set_op("SETDVALID");
    ins.set_thread_id(2u);  // SFPU_THREAD
    EXPECT_EQ(ins.get_context(), static_cast<int>(ThreadMap::SFPU_THREAD));
}

TEST(InstructionGetContext, CLEARDVALID_UnpackerMask) {
    Instruction ins;
    ins.set_op("CLEARDVALID");
    ins.set_attr({{"dest_pulse_last", CLEAR_DVALID_UNPACKER}});
    EXPECT_EQ(ins.get_context(), static_cast<int>(ThreadMap::UNPACKER_THREAD));
}

TEST(InstructionGetContext, CLEARDVALID_MathMask) {
    Instruction ins;
    ins.set_op("CLEARDVALID");
    ins.set_attr({{"dest_pulse_last", CLEAR_DVALID_MATH}});
    EXPECT_EQ(ins.get_context(), static_cast<int>(ThreadMap::MATH_THREAD));
}

TEST(InstructionGetContext, CLEARDVALID_SfpuMask) {
    Instruction ins;
    ins.set_op("CLEARDVALID");
    ins.set_attr({{"dest_pulse_last", CLEAR_DVALID_SFPU}});
    EXPECT_EQ(ins.get_context(), static_cast<int>(ThreadMap::SFPU_THREAD));
}

TEST(InstructionGetContext, CLEARDVALID_PackerMask) {
    Instruction ins;
    ins.set_op("CLEARDVALID");
    ins.set_attr({{"dest_pulse_last", CLEAR_DVALID_PACKER}});
    EXPECT_EQ(ins.get_context(), static_cast<int>(ThreadMap::PACKER_THREAD));
}

TEST(InstructionGetContext, CLEARDVALID_ZeroMaskFallsBackToThread) {
    Instruction ins;
    ins.set_op("CLEARDVALID");
    ins.set_thread_id(1u);  // MATH_THREAD
    ins.set_attr({{"dest_pulse_last", 0}});
    EXPECT_EQ(ins.get_context(), static_cast<int>(ThreadMap::MATH_THREAD));
}

TEST(InstructionGetContext, ExPipeUnpacker) {
    Instruction ins;
    ins.set_op("SOMEOP");
    ins.set_thread_id(0u);
    ins.set_ex_pipe("UNPACKER1");
    EXPECT_EQ(ins.get_context(), static_cast<int>(ThreadMap::UNPACKER_THREAD));
}

TEST(InstructionGetContext, ExPipeMath) {
    Instruction ins;
    ins.set_op("SOMEOP");
    ins.set_thread_id(0u);
    ins.set_ex_pipe("INSTRISSUE");
    EXPECT_EQ(ins.get_context(), static_cast<int>(ThreadMap::MATH_THREAD));
}

TEST(InstructionGetContext, ExPipeSfpu) {
    Instruction ins;
    ins.set_op("SOMEOP");
    ins.set_thread_id(0u);
    ins.set_ex_pipe("SFPU");
    EXPECT_EQ(ins.get_context(), static_cast<int>(ThreadMap::SFPU_THREAD));
}

TEST(InstructionGetContext, ExPipePacker) {
    Instruction ins;
    ins.set_op("SOMEOP");
    ins.set_thread_id(0u);
    ins.set_ex_pipe("PACKER0");
    EXPECT_EQ(ins.get_context(), static_cast<int>(ThreadMap::PACKER_THREAD));
}

TEST(InstructionGetContext, FallbackToThreadId) {
    Instruction ins;
    ins.set_op("SOMEOP");
    ins.set_thread_id(3u);  // PACKER_THREAD
    // No ex_pipe set, not SETDVALID/CLEARDVALID
    EXPECT_EQ(ins.get_context(), static_cast<int>(ThreadMap::PACKER_THREAD));
}

// ---------------------------------------------------------------------------
// set_state copies correct subset of fields
// ---------------------------------------------------------------------------

TEST(InstructionSetState, CopiesPipeSyncFields) {
    Instruction src;
    src.set_op("BASE");
    src.set_bank_upd_mask({{0, 1}});
    src.set_cond_chk_vld_upd({{0, {{0, 5}}}});
    src.set_cond_wri_vld_upd({{1, {{2, 9}}}});
    src.set_dst_int({4, 5});
    src.set_dst_pipes({"PACKER0"});
    src.set_ex_pipe("PACKER0");
    src.set_imm({0xAB});
    src.set_src_int({1, 2, 3});
    src.set_src_pipes({"UNPACKER0"});
    src.set_vld_upd_mask({{2, 0xFF}});
    src.set_pipes_thread_id(1u);

    Instruction dst;
    dst.set_op("DST_OP");  // Should not be overwritten
    dst.set_ins_id(99u);   // Should not be overwritten
    dst.set_state(src);

    // Fields that must be copied.
    EXPECT_TRUE(dst.has_bank_upd_mask(0));
    EXPECT_EQ(dst.get_bank_upd_mask(0), 1);
    EXPECT_TRUE(dst.has_cond_chk_vld_upd(0, 0));
    EXPECT_EQ(dst.get_cond_chk_vld_upd(0, 0), 5);
    EXPECT_TRUE(dst.has_cond_wri_vld_upd(1, 2));
    EXPECT_EQ(dst.get_cond_wri_vld_upd(1, 2), 9);
    EXPECT_EQ(dst.get_dst_int(), (std::vector<int>{4, 5}));
    EXPECT_EQ(dst.get_dst_pipes(), (std::vector<std::string>{"PACKER0"}));
    EXPECT_EQ(dst.get_ex_pipe(), "PACKER0");
    EXPECT_EQ(dst.get_imm(), (std::vector<int>{0xAB}));
    EXPECT_EQ(dst.get_src_int(), (std::vector<int>{1, 2, 3}));
    EXPECT_EQ(dst.get_src_pipes(), (std::vector<std::string>{"UNPACKER0"}));
    EXPECT_TRUE(dst.has_vld_upd_mask(2));
    EXPECT_EQ(dst.get_vld_upd_mask(2), 0xFF);
    EXPECT_EQ(dst.get_pipes_thread_id(), 1u);

    // Fields that must NOT be overwritten.
    EXPECT_EQ(dst.get_op(), "DST_OP");
    EXPECT_EQ(dst.get_ins_id(), 99u);
}

// ---------------------------------------------------------------------------
// get_t3sim_pipes_from_stall_res
// ---------------------------------------------------------------------------

TEST(GetT3SimPipes, BasicMapping) {
    std::map<std::string, std::vector<std::string>> pipe_grps = {
        {"TDMA",     {"UNPACKER0", "UNPACKER1"}},
        {"MATH",     {"MATH"}},
        {"SFPU",     {"SFPU"}},
        {"PACKER",   {"PACKER0", "PACKER1"}},
    };
    std::vector<std::string> pipes = {"UNPACKER0", "UNPACKER1", "MATH", "SFPU", "PACKER0", "PACKER1"};

    // "compute/tdma" should map to "TDMA" engine group.
    auto result = get_t3sim_pipes_from_stall_res({"compute/tdma"}, pipe_grps, pipes);
    EXPECT_EQ(result, (std::vector<int>{0, 1}));
}

TEST(GetT3SimPipes, DeduplicatesAndSorts) {
    std::map<std::string, std::vector<std::string>> pipe_grps = {
        {"MATH", {"MATH", "SFPU"}},
        {"SFPU", {"SFPU"}},
    };
    std::vector<std::string> pipes = {"MATH", "SFPU"};

    auto result = get_t3sim_pipes_from_stall_res({"math", "sfpu"}, pipe_grps, pipes);
    EXPECT_EQ(result, (std::vector<int>{0, 1}));
}

TEST(GetT3SimPipes, EmptyStallResThrows) {
    std::map<std::string, std::vector<std::string>> pipe_grps;
    std::vector<std::string> pipes;
    EXPECT_THROW(get_t3sim_pipes_from_stall_res({}, pipe_grps, pipes), std::runtime_error);
}

// ---------------------------------------------------------------------------
// InstrPtr shared_ptr alias
// ---------------------------------------------------------------------------

TEST(InstrPtr, BasicSharedOwnership) {
    InstrPtr p = std::make_shared<Instruction>();
    p->set_ins_id(7u);
    InstrPtr q = p;
    EXPECT_EQ(q->get_ins_id(), 7u);
    EXPECT_EQ(p.use_count(), 2);
}
