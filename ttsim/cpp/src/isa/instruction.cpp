#include "neosim/isa/instruction.hpp"

#include <cassert>
#include <sstream>
#include <stdexcept>

namespace neosim::isa {

// ---------------------------------------------------------------------------
// Free functions
// ---------------------------------------------------------------------------

int get_num_bytes_from_data_format(int format) {
    switch (format) {
        case 0:   return 4; // Float32 - E8M23
        case 4:   return 2; // Tf32 - E8M10 - stored in 32-bit container
        case 1:   return 2; // Float16 - E5M10
        case 5:   return 2; // Float16_b - E8M7
        case 10:  return 1; // Fp8R - E5M2
        case 16:  return 1; // Fp8P - E4M3
        case 18:  return 1; // MxFp8R - E5M2 with block exp
        case 20:  return 1; // MxFp8P - E4M3 with block exp
        case 19:  return 1; // MxFp6R - E3M2 with block exp
        case 21:  return 1; // MxFp6P - E2M3 with block exp
        case 22:  return 1; // MxFp4 - E2M1 with block exp
        case 2:   return 1; // MxInt8 - E0M7 with block exp
        case 3:   return 1; // MxInt4 - E0M3 with block exp
        case 11:  return 1; // MxInt2 - E0M1 with block exp
        case 8:   return 4; // Int32
        case 14:  return 1; // Int8
        case 9:   return 2; // Int16
        case 17:  return 1; // Uint8
        case 130: return 2; // Uint16
        case 27:  return 1; // MxFp4_2x_A
        case 24:  return 1; // MxFp4_2x_B
        case 23:  return 1; // Int4
        case 25:  return 1; // Uint4
        case 26:  return 1; // Int8_2x
        case 255: return 2; // Unknown (UNPACR0_STRIDE), defaults to Float16
        default:
            throw std::runtime_error("get_num_bytes_from_data_format: unhandled format " +
                                     std::to_string(format));
    }
}

std::vector<int> get_t3sim_pipes_from_stall_res(
    const std::vector<std::string>&                        stall_res,
    const std::map<std::string, std::vector<std::string>>& pipe_grps,
    const std::vector<std::string>&                        pipes)
{
    if (stall_res.empty()) {
        throw std::runtime_error(
            "get_t3sim_pipes_from_stall_res: stall_res must be a non-empty list");
    }

    std::vector<std::string> pipe_list;
    for (const auto& res : stall_res) {
        const std::string engine_grp = (res == "compute/tdma") ? "TDMA" : [&]{
            std::string up = res;
            for (char& c : up) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
            return up;
        }();
        const auto it = pipe_grps.find(engine_grp);
        if (it == pipe_grps.end()) {
            throw std::runtime_error(
                "get_t3sim_pipes_from_stall_res: unknown engine group '" + engine_grp + "'");
        }
        for (const auto& p : it->second) {
            pipe_list.push_back(p);
        }
    }

    // Deduplicate and sort pipe indices.
    std::vector<int> result;
    for (const auto& p : pipe_list) {
        const auto it = std::find(pipes.begin(), pipes.end(), p);
        if (it == pipes.end()) {
            throw std::runtime_error(
                "get_t3sim_pipes_from_stall_res: pipe '" + p + "' not found in pipes list");
        }
        int idx = static_cast<int>(std::distance(pipes.begin(), it));
        if (std::find(result.begin(), result.end(), idx) == result.end()) {
            result.push_back(idx);
        }
    }
    std::sort(result.begin(), result.end());
    return result;
}

// ---------------------------------------------------------------------------
// Instruction constructors
// ---------------------------------------------------------------------------

Instruction::Instruction(const ttdecode::decode::decoded_instruction& di) {
    // Copy all base-class fields.
    word               = di.word;
    program_counter    = di.program_counter;
    kind               = di.kind;
    opcode             = di.opcode;
    mnemonic           = di.mnemonic;
    operands           = di.operands;
    compressed_quadrant = di.compressed_quadrant;
    compressed_funct3  = di.compressed_funct3;
}

// ---------------------------------------------------------------------------
// Address / identity
// ---------------------------------------------------------------------------

uint32_t Instruction::get_addr() const {
    return program_counter.value_or(0);
}

// ---------------------------------------------------------------------------
// Opcode / mnemonic
// ---------------------------------------------------------------------------

const std::string& Instruction::get_op() const {
    static const std::string empty;
    return mnemonic.has_value() ? *mnemonic : empty;
}

void Instruction::set_op(const std::string& op) {
    mnemonic = op;
}

// ---------------------------------------------------------------------------
// Instruction-class predicates
// ---------------------------------------------------------------------------

bool Instruction::is_tt() const {
    if (!kind.has_value()) return false;
    return kind == ttdecode::isa::instruction_kind::ttqs ||
           kind == ttdecode::isa::instruction_kind::ttwh ||
           kind == ttdecode::isa::instruction_kind::ttbh;
}

bool Instruction::is_mop() const {
    return mnemonic.has_value() && *mnemonic == "MOP";
}

bool Instruction::is_replay() const {
    return mnemonic.has_value() && *mnemonic == "REPLAY";
}

// ---------------------------------------------------------------------------
// Thread / pipe context
// ---------------------------------------------------------------------------

int Instruction::get_context() const {
    const std::string& op = get_op();
    int context = -1;

    if (op == "SETDVALID") {
        context = static_cast<int>(thread_id_);
    } else if (op == "CLEARDVALID") {
        const auto attrs = get_attr();
        const auto it = attrs.find("dest_pulse_last");
        if (it != attrs.end()) {
            const int mask = it->second;
            if      (mask & CLEAR_DVALID_UNPACKER) context = static_cast<int>(ThreadMap::UNPACKER_THREAD);
            else if (mask & CLEAR_DVALID_MATH)     context = static_cast<int>(ThreadMap::MATH_THREAD);
            else if (mask & CLEAR_DVALID_SFPU)     context = static_cast<int>(ThreadMap::SFPU_THREAD);
            else if (mask & CLEAR_DVALID_PACKER)   context = static_cast<int>(ThreadMap::PACKER_THREAD);
            else                                   context = static_cast<int>(thread_id_);
        } else {
            context = static_cast<int>(thread_id_);
        }
    } else if (ex_pipe_ == "UNPACKER0" || ex_pipe_ == "UNPACKER1" || ex_pipe_ == "UNPACKER2") {
        context = static_cast<int>(ThreadMap::UNPACKER_THREAD);
    } else if (ex_pipe_ == "MATH" || ex_pipe_ == "INSTRISSUE") {
        context = static_cast<int>(ThreadMap::MATH_THREAD);
    } else if (ex_pipe_ == "SFPU") {
        context = static_cast<int>(ThreadMap::SFPU_THREAD);
    } else if (ex_pipe_ == "PACKER0" || ex_pipe_ == "PACKER1") {
        context = static_cast<int>(ThreadMap::PACKER_THREAD);
    } else {
        context = static_cast<int>(thread_id_);
    }

    assert(context >= 0 && "get_context: invalid context (< 0)");
    assert(context < static_cast<int>(ThreadMap::NUM_CONTEXTS) &&
           "get_context: context out of bounds");
    return context;
}

// ---------------------------------------------------------------------------
// Operand accessors
// ---------------------------------------------------------------------------

std::vector<int> Instruction::get_src_int() const {
    if (!operands.has_value()) return {};
    return operands->sources.integers;
}

void Instruction::set_src_int(const std::vector<int>& v) {
    if (v.empty()) return;
    set_integer_sources(v);
}

std::vector<int> Instruction::get_dst_int() const {
    if (!operands.has_value()) return {};
    return operands->destinations.integers;
}

void Instruction::set_dst_int(const std::vector<int>& v) {
    if (v.empty()) return;
    set_integer_destinations(v);
}

std::vector<int> Instruction::get_imm() const {
    if (!operands.has_value()) return {};
    return operands->immediates;
}

void Instruction::set_imm(const std::vector<int>& v) {
    if (v.empty()) return;
    set_immediates(v);
}

std::map<std::string, int> Instruction::get_attr() const {
    if (!operands.has_value()) return {};
    return operands->attributes;
}

void Instruction::set_attr(const std::map<std::string, int>& attrs) {
    if (attrs.empty()) return;
    set_attributes(attrs);
}

// ---------------------------------------------------------------------------
// Data format / size
// ---------------------------------------------------------------------------

int Instruction::get_src_size() const {
    if (!src_format_.has_value()) return 0;
    return get_num_bytes_from_data_format(*src_format_) * num_datums_;
}

int Instruction::get_dst_size() const {
    if (!dst_format_.has_value()) return 0;
    return get_num_bytes_from_data_format(*dst_format_) * num_datums_;
}

// ---------------------------------------------------------------------------
// Synchronisation masks
// ---------------------------------------------------------------------------

bool Instruction::has_vld_upd_mask(int r) const {
    return vld_upd_mask_.count(r) > 0;
}

int Instruction::get_vld_upd_mask(int r) const {
    return vld_upd_mask_.at(r);
}

void Instruction::set_vld_upd_mask(const std::map<int, int>& m) {
    for (const auto& [k, v] : m) {
        vld_upd_mask_[k] = v;
    }
}

bool Instruction::has_bank_upd_mask(int r) const {
    return bank_upd_mask_.count(r) > 0;
}

int Instruction::get_bank_upd_mask(int r) const {
    return bank_upd_mask_.at(r);
}

void Instruction::set_bank_upd_mask(const std::map<int, int>& m) {
    for (const auto& [k, v] : m) {
        bank_upd_mask_[k] = v;
    }
}

bool Instruction::has_cond_chk_vld_upd(int r, int t) const {
    const auto outer = cond_chk_vld_upd_val_.find(r);
    if (outer == cond_chk_vld_upd_val_.end()) return false;
    const auto inner = outer->second.find(t);
    if (inner == outer->second.end()) return false;
    return inner->second != static_cast<int>(ValueStatus::IGNORE);
}

int Instruction::get_cond_chk_vld_upd(int r, int t) const {
    return cond_chk_vld_upd_val_.at(r).at(t);
}

void Instruction::set_cond_chk_vld_upd(const std::map<int, std::map<int, int>>& v) {
    cond_chk_vld_upd_val_ = v;
}

bool Instruction::has_cond_wri_vld_upd(int r, int t) const {
    const auto outer = cond_wri_vld_upd_val_.find(r);
    if (outer == cond_wri_vld_upd_val_.end()) return false;
    const auto inner = outer->second.find(t);
    if (inner == outer->second.end()) return false;
    return inner->second != static_cast<int>(ValueStatus::IGNORE);
}

int Instruction::get_cond_wri_vld_upd(int r, int t) const {
    return cond_wri_vld_upd_val_.at(r).at(t);
}

void Instruction::set_cond_wri_vld_upd(const std::map<int, std::map<int, int>>& v) {
    cond_wri_vld_upd_val_ = v;
}

bool Instruction::has_pipe_bank_ctrl(int r) const {
    return pipe_bank_ctrl_.count(r) > 0;
}

int Instruction::get_pipe_bank_ctrl(int r) const {
    return pipe_bank_ctrl_.at(r);
}

void Instruction::set_pipe_bank_ctrl(const std::map<int, int>& m) {
    for (const auto& [k, v] : m) {
        pipe_bank_ctrl_[k] = v;
    }
}

// ---------------------------------------------------------------------------
// Memory / event info
// ---------------------------------------------------------------------------

void Instruction::set_mem_info(const std::string& key, int64_t value) {
    mem_info_[key] = value;
}

int64_t Instruction::get_mem_info(const std::string& key) const {
    const auto it = mem_info_.find(key);
    if (it == mem_info_.end()) {
        throw std::out_of_range("get_mem_info: key '" + key + "' not found");
    }
    return it->second;
}

void Instruction::incr_mem_info(const std::string& key, int64_t delta) {
    mem_info_[key] += delta;
}

void Instruction::set_event_info(const std::string& key, const std::string& value) {
    event_info_[key] = value;
}

std::string Instruction::get_event_info(const std::string& key) const {
    const auto it = event_info_.find(key);
    if (it == event_info_.end()) {
        throw std::out_of_range("get_event_info: key '" + key + "' not found");
    }
    return it->second;
}

// ---------------------------------------------------------------------------
// State copy (ReplayUnit)
// ---------------------------------------------------------------------------

void Instruction::set_state(const Instruction& other) {
    set_bank_upd_mask(other.bank_upd_mask_);
    set_cond_chk_vld_upd(other.cond_chk_vld_upd_val_);
    set_cond_wri_vld_upd(other.cond_wri_vld_upd_val_);
    set_dst_int(other.get_dst_int());
    set_dst_pipes(other.dst_pipes_);
    set_ex_pipe(other.ex_pipe_);
    set_imm(other.get_imm());
    set_src_int(other.get_src_int());
    set_src_pipes(other.src_pipes_);
    set_vld_upd_mask(other.vld_upd_mask_);
    if (other.pipes_thread_id_.has_value()) {
        pipes_thread_id_ = other.pipes_thread_id_;
    }
}

// ---------------------------------------------------------------------------
// Diagnostics
// ---------------------------------------------------------------------------

std::string Instruction::to_string() const {
    std::ostringstream oss;
    oss << "Instruction{"
        << "id=" << ins_id_
        << " thread=" << thread_id_
        << " addr=0x" << std::hex << get_addr() << std::dec
        << " op=" << get_op();

    const auto srcs = get_src_int();
    oss << " src=[";
    for (std::size_t i = 0; i < srcs.size(); ++i) {
        if (i) oss << ',';
        oss << srcs[i];
    }
    oss << ']';

    const auto dsts = get_dst_int();
    oss << " dst=[";
    for (std::size_t i = 0; i < dsts.size(); ++i) {
        if (i) oss << ',';
        oss << dsts[i];
    }
    oss << ']';

    oss << " ex_pipe=" << ex_pipe_
        << '}';
    return oss.str();
}

} // namespace neosim::isa
