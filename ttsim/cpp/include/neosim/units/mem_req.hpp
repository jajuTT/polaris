// SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <atomic>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace neosim::units {

/// Memory operation type.
enum class MemOp { READ, WRITE };

/// L1 scratchpad memory request token — replaces Python class `memReq` (scratchpad.py).
///
/// A MemReq is created by a pipe unit, submitted to the Scratchpad, and returned
/// (with the same object) on completion.  The unique req_id (auto-assigned from a
/// class-level counter) lets multiple in-flight requests be tracked simultaneously.
///
/// Fields that are not yet set at construction are represented as std::optional
/// (ins_id, src, target, src_pipe) or use a sentinel (-1 for integer IDs).
///
/// Thread-safety: the global req_id counter uses std::atomic so that concurrent
/// unit tests in different threads (e.g. under `ctest -j`) do not alias IDs.
class MemReq {
public:
    // ----------------------------------------------------------------
    // Global request ID counter (matches Python class-level itertools.count)
    // ----------------------------------------------------------------

    /// Allocate and return the next unique request ID.
    static uint32_t alloc_req_id() { return counter_.fetch_add(1); }

    /// Reset the counter — for unit-test reproducibility only.
    static void reset_counter(uint32_t val = 0) { counter_.store(val); }

    // ----------------------------------------------------------------
    // Construction
    // ----------------------------------------------------------------

    /// Primary constructor (matches Python memReq.__init__).
    /// req_id is auto-assigned; all other optional fields start unset.
    MemReq(MemOp op, uint32_t addr, uint32_t bytes)
        : req_id_(alloc_req_id())
        , op_(op)
        , addr_(addr)
        , bytes_(bytes)
    {}

    // ----------------------------------------------------------------
    // Read accessors
    // ----------------------------------------------------------------

    uint32_t req_id()   const { return req_id_; }
    MemOp    op()       const { return op_; }
    uint32_t addr()     const { return addr_; }
    uint32_t bytes()    const { return bytes_; }
    int      thread_id() const { return thread_id_; }
    int      core_id()   const { return core_id_; }
    int      pipe_id()   const { return pipe_id_; }

    std::optional<uint32_t>    ins_id()   const { return ins_id_; }
    std::optional<std::string> src()      const { return src_; }
    std::optional<std::string> target()   const { return target_; }
    std::optional<std::string> src_pipe() const { return src_pipe_; }

    const std::vector<uint32_t>& parent_req_ids() const { return parent_req_ids_; }
    const std::vector<uint32_t>& child_req_ids()  const { return child_req_ids_; }

    // ----------------------------------------------------------------
    // Write accessors (mutators)
    // ----------------------------------------------------------------

    void set_op(MemOp v)          { op_ = v; }
    void set_addr(uint32_t v)     { addr_ = v; }
    void set_bytes(uint32_t v)    { bytes_ = v; }
    void set_thread_id(int v)     { thread_id_ = v; }
    void set_core_id(int v)       { core_id_ = v; }
    void set_pipe_id(int v)       { pipe_id_ = v; }
    void set_ins_id(uint32_t v)   { ins_id_ = v; }
    void set_src(std::string v)   { src_ = std::move(v); }
    void set_target(std::string v){ target_ = std::move(v); }
    void set_src_pipe(std::string v){ src_pipe_ = std::move(v); }

    void set_parent_req_ids(std::vector<uint32_t> ids) {
        parent_req_ids_ = std::move(ids);
    }
    void set_child_req_ids(std::vector<uint32_t> ids) {
        child_req_ids_ = std::move(ids);
    }

    void add_parent_req_id(uint32_t id) { parent_req_ids_.push_back(id); }
    void add_child_req_id(uint32_t id)  { child_req_ids_.push_back(id); }

    // ----------------------------------------------------------------
    // Diagnostic
    // ----------------------------------------------------------------

    /// Print a one-line summary (mirrors Python __printReq__).
    void print() const;

    /// Print the L1_TRACE line if target == "L1" (mirrors Python __traceReq__).
    void trace(int cycle) const;

private:
    static std::atomic<uint32_t> counter_;

    uint32_t req_id_;
    MemOp    op_;
    uint32_t addr_;
    uint32_t bytes_;

    // Optional fields: not all are set at construction time.
    std::optional<uint32_t>    ins_id_;
    std::optional<std::string> src_;
    std::optional<std::string> target_;
    std::optional<std::string> src_pipe_;

    // Integer IDs: -1 = unset (matches Python sentinel -1 for threadId).
    int thread_id_ = -1;
    int core_id_   = -1;
    int pipe_id_   = -1;

    std::vector<uint32_t> parent_req_ids_;
    std::vector<uint32_t> child_req_ids_;
};

} // namespace neosim::units
