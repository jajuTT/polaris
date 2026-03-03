#pragma once

#include "decode/decoded_instruction.hpp"
#include "isa/isa.hpp"

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace neosim::isa {

// ---------------------------------------------------------------------------
// Enums — direct translations of Python IntEnums
// ---------------------------------------------------------------------------

enum class RegIndex : int {
    SRC_A = 0,
    SRC_B = 1,
    SRC_S = 2,
    DST   = 3,
    COUNT = 4,
};

enum class ThreadMap : int {
    UNPACKER_THREAD = 0,
    MATH_THREAD     = 1,
    SFPU_THREAD     = 2,
    PACKER_THREAD   = 3,
    NUM_CONTEXTS    = 4,
};

/// Bit flags used in CLEARDVALID dest_pulse_last field.
enum ClearDvalidMask : int {
    CLEAR_DVALID_NONE     = 0x0,
    CLEAR_DVALID_UNPACKER = 0x1,
    CLEAR_DVALID_MATH     = 0x2,
    CLEAR_DVALID_SFPU     = 0x4,
    CLEAR_DVALID_PACKER   = 0x8,
};

/// Sentinel values stored in conditional valid maps.
enum class ValueStatus : int {
    IGNORE = -2,
    UNSET  = -1,
};

// ---------------------------------------------------------------------------
// Free functions
// ---------------------------------------------------------------------------

/// Returns the number of bytes per datum for the given ttdecode data-format code.
/// Mirrors Python getNumBytesFromDataFormat().
int get_num_bytes_from_data_format(int format);

/// Translates stall-resource strings into pipe indices.
/// Mirrors Python getT3SimPipesFromStallRes().
std::vector<int> get_t3sim_pipes_from_stall_res(
    const std::vector<std::string>&                            stall_res,
    const std::map<std::string, std::vector<std::string>>&     pipe_grps,
    const std::vector<std::string>&                            pipes);

// ---------------------------------------------------------------------------
// Instruction
// ---------------------------------------------------------------------------

/// Central instruction token that flows through the NeoSim pipeline.
///
/// Inherits ttdecode::decode::decoded_instruction (IS-A relationship, matching
/// the Python `instr(decoded_instruction, operands)` hierarchy).  All
/// NeoSim-specific simulation fields are added here with trailing underscores
/// per project conventions.
class Instruction : public ttdecode::decode::decoded_instruction {
public:
    // ------------------------------------------------------------------
    // Constructors
    // ------------------------------------------------------------------

    Instruction() = default;

    /// Wrap a freshly decoded instruction.  Copies all base-class fields.
    explicit Instruction(const ttdecode::decode::decoded_instruction& di);

    // ------------------------------------------------------------------
    // Address / identity
    // ------------------------------------------------------------------

    /// Returns program_counter, or 0 if not set.
    uint32_t get_addr() const;

    uint32_t get_ins_id() const  { return ins_id_; }
    void     set_ins_id(uint32_t id) { ins_id_ = id; }

    uint32_t get_core_id() const  { return core_id_; }
    void     set_core_id(uint32_t id) { core_id_ = id; }

    uint32_t get_thread_id() const  { return thread_id_; }
    void     set_thread_id(uint32_t id) { thread_id_ = id; }

    // ------------------------------------------------------------------
    // Opcode / mnemonic
    // ------------------------------------------------------------------

    const std::string& get_op() const;
    void               set_op(const std::string& op);

    // ------------------------------------------------------------------
    // Instruction-class predicates
    // ------------------------------------------------------------------

    /// True when kind is ttqs, ttwh, or ttbh.
    bool is_tt() const;

    /// True when mnemonic == "MOP".
    bool is_mop() const;

    /// True when mnemonic == "REPLAY".
    bool is_replay() const;

    // ------------------------------------------------------------------
    // Thread / pipe context
    // ------------------------------------------------------------------

    /// Derive the ThreadMap context from mnemonic, CLEARDVALID attrs, or ex_pipe_.
    /// Returns a ThreadMap int (0–3).  Asserts on invalid values.
    int get_context() const;

    const std::string& get_ex_pipe() const    { return ex_pipe_; }
    void               set_ex_pipe(const std::string& p) { if (!p.empty()) ex_pipe_ = p; }

    // ------------------------------------------------------------------
    // Operand accessors (delegate to operands field)
    // ------------------------------------------------------------------

    /// Sources as integer register indices.  Returns empty vector if not set.
    std::vector<int> get_src_int() const;
    void             set_src_int(const std::vector<int>& v);

    /// Destinations as integer register indices.  Returns empty vector if not set.
    std::vector<int> get_dst_int() const;
    void             set_dst_int(const std::vector<int>& v);

    /// Immediates list.  Returns empty vector if not set.
    std::vector<int> get_imm() const;
    void             set_imm(const std::vector<int>& v);

    /// Attributes map.  Returns empty map if not set.
    std::map<std::string, int> get_attr() const;
    void                       set_attr(const std::map<std::string, int>& attrs);

    // ------------------------------------------------------------------
    // Data format / size
    // ------------------------------------------------------------------

    std::optional<int> get_src_format() const { return src_format_; }
    void               set_src_format(int f)  { src_format_ = f; }

    std::optional<int> get_dst_format() const { return dst_format_; }
    void               set_dst_format(int f)  { dst_format_ = f; }

    int  get_num_datums() const       { return num_datums_; }
    void set_num_datums(int n)        { num_datums_ = n; }

    /// src_format bytes-per-datum × num_datums.
    int get_src_size() const;

    /// dst_format bytes-per-datum × num_datums.
    int get_dst_size() const;

    // ------------------------------------------------------------------
    // Pipe routing
    // ------------------------------------------------------------------

    const std::vector<std::string>& get_src_pipes() const { return src_pipes_; }
    void set_src_pipes(const std::vector<std::string>& v) { src_pipes_ = v; }

    const std::vector<std::string>& get_dst_pipes() const { return dst_pipes_; }
    void set_dst_pipes(const std::vector<std::string>& v) { dst_pipes_ = v; }

    std::optional<uint32_t> get_pipes_thread_id() const   { return pipes_thread_id_; }
    void set_pipes_thread_id(uint32_t tid)                 { pipes_thread_id_ = tid; }

    int32_t get_pipe_delay() const   { return pipe_delay_; }
    void    set_pipe_delay(int32_t d) { pipe_delay_ = d; }

    // ------------------------------------------------------------------
    // Synchronisation masks (per-register, per-register×thread)
    // ------------------------------------------------------------------

    bool has_vld_upd_mask(int r) const;
    int  get_vld_upd_mask(int r) const;
    void set_vld_upd_mask(const std::map<int, int>& m);

    bool has_bank_upd_mask(int r) const;
    int  get_bank_upd_mask(int r) const;
    void set_bank_upd_mask(const std::map<int, int>& m);

    /// Returns true when condChkVldUpdVal[r][t] exists and is not IGNORE.
    bool has_cond_chk_vld_upd(int r, int t) const;
    int  get_cond_chk_vld_upd(int r, int t) const;
    void set_cond_chk_vld_upd(const std::map<int, std::map<int, int>>& v);

    /// Returns true when condWriVldUpdVal[r][t] exists and is not IGNORE.
    bool has_cond_wri_vld_upd(int r, int t) const;
    int  get_cond_wri_vld_upd(int r, int t) const;
    void set_cond_wri_vld_upd(const std::map<int, std::map<int, int>>& v);

    bool has_pipe_bank_ctrl(int r) const;
    int  get_pipe_bank_ctrl(int r) const;
    void set_pipe_bank_ctrl(const std::map<int, int>& m);

    // ------------------------------------------------------------------
    // Memory / event info
    // ------------------------------------------------------------------

    void    set_mem_info(const std::string& key, int64_t value);
    int64_t get_mem_info(const std::string& key) const;
    void    incr_mem_info(const std::string& key, int64_t delta);

    void        set_event_info(const std::string& key, const std::string& value);
    std::string get_event_info(const std::string& key) const;

    // ------------------------------------------------------------------
    // State copy (used by ReplayUnit)
    // ------------------------------------------------------------------

    /// Copies pipe/sync fields from another Instruction (mirrors Python setState()).
    void set_state(const Instruction& other);

    // ------------------------------------------------------------------
    // Diagnostics
    // ------------------------------------------------------------------

    std::string to_string() const;

private:
    // Identity
    uint32_t ins_id_    = 0;
    uint32_t core_id_   = 0;
    uint32_t thread_id_ = 0;

    // Pipe routing
    int32_t     pipe_delay_ = -1;
    std::string ex_pipe_;

    // Synchronisation masks
    std::map<int, int>                 vld_upd_mask_;
    std::map<int, int>                 bank_upd_mask_;
    std::map<int, std::map<int, int>>  cond_chk_vld_upd_val_;
    std::map<int, std::map<int, int>>  cond_wri_vld_upd_val_;
    std::map<int, int>                 pipe_bank_ctrl_;

    // Pipe lists
    std::vector<std::string>  src_pipes_;
    std::vector<std::string>  dst_pipes_;
    std::optional<uint32_t>   pipes_thread_id_;

    // Data format
    std::optional<int> src_format_;
    std::optional<int> dst_format_;
    int                num_datums_ = 0;

    // Auxiliary info
    std::map<std::string, int64_t>     mem_info_;
    std::map<std::string, std::string> event_info_;
};

/// Shared ownership handle — instructions are created once and shared across
/// pipeline stages (ROB, pipe buffers, replay list).
using InstrPtr = std::shared_ptr<Instruction>;

} // namespace neosim::isa
