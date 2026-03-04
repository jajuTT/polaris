// SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "neosim/isa/instruction.hpp"
#include "neosim/risc/trisc_func.hpp"
#include "neosim/risc/trisc_regs.hpp"
#include "neosim/units/pipe_resource.hpp"
#include "neosim/units/replay.hpp"
#include "neosim/units/risc_reg.hpp"
#include "neosim/units/rob.hpp"
#include "neosim/units/tensix_func.hpp"
#include "neosim/units/tensix_reg.hpp"
#include "neosim/units/tensix_spl_reg.hpp"

#include <deque>
#include <map>
#include <string>
#include <vector>

namespace neosim::units {

/// Thread execution engine — replaces Python class `thread` (t3sim.py).
///
/// Models the per-thread instruction pipeline as six cooperating pipeline stages:
///
///   [fetch] → input_buf → [decode]
///                              ├─ TT/MOP → mop_buf → [mop_decode] → instr_buf → [arbiter]
///                              └─ RISC: exec, then → risc_check_buf → [scoreboard_check]
///                                                         └─ risc_exec_trk_buf → [scoreboard_reset]
///
/// In Python these are six SimPy coroutines.  In C++ each stage is a method
/// called once per simulated cycle from `step()`.  Blocking conditions are
/// modelled as "stall" returns: the stage does nothing and the caller retries
/// next cycle, mirroring the Sparta `event_.schedule(1)` re-scheduling pattern.
///
/// No Sparta dependency — pure C++20 data model testable on macOS ARM64.
class ThreadUnit {
public:
    // ----------------------------------------------------------------
    // Configuration
    // ----------------------------------------------------------------
    struct Config {
        int         core_id   = 0;
        int         thread_id = 0;
        int         risc_pipe_depth        = 3;   ///< riscPipeDepth
        int         branch_mispredict_lat  = 3;   ///< branchMisPredictLat
        bool        enable_scoreboard      = true; ///< enableScoreboardCheckforRegs
        bool        enable_forwarding      = false; ///< enableForwardingforRegs
        bool        enable_in_order_issue  = true;  ///< in-order TT issue via ROB
        bool        enable_pipe_stall      = true;  ///< enablePipeStall
        bool        enable_sync            = true;  ///< enableSync (valid checks)
        std::string arch;
        std::vector<std::string>                  pipes;
        TensixFunc::PipeGroups                    pipe_grps;
    };

    // ----------------------------------------------------------------
    // Kernel / instruction table
    // ----------------------------------------------------------------

    /// One decoded kernel: a contiguous address range and its instructions.
    struct KernelRange {
        std::string name;
        uint32_t    start_addr = 0;
        uint32_t    end_addr   = 0;
    };

    /// Address → decoded Instruction map (one per kernel).
    using InstrMap = std::map<uint32_t, isa::InstrPtr>;

    // ----------------------------------------------------------------
    // Result of one arbiter cycle
    // ----------------------------------------------------------------

    /// What the arbiter produced this cycle (returned by step()).
    struct RouteResult {
        isa::InstrPtr ins;         ///< instruction ready for pipe (nullptr if none)
        std::string   target_pipe; ///< pipe name (empty if ins is nullptr or NONE pipe)
    };

    // ----------------------------------------------------------------
    // Construction
    // ----------------------------------------------------------------

    explicit ThreadUnit(const Config&       cfg,
                        TensixFunc&         tensix_func,
                        risc::TriscFunc&    trisc_func,
                        TensixReg&          tensix_reg,
                        RiscReg&            risc_reg,
                        PipeResource&       pipe_resource,
                        TensixSplReg&       spl_regs,
                        risc::TriscRegs&    trisc_regs);

    // ----------------------------------------------------------------
    // Kernel loading / sequencing
    // ----------------------------------------------------------------

    /// Load a kernel's decoded instruction map and address range.
    void load_kernel(const std::string& name,
                     const InstrMap&    instrs,
                     uint32_t           start_addr,
                     uint32_t           end_addr);

    /// Queue a kernel name for execution.  The thread runs kernels FIFO.
    void enqueue_kernel(const std::string& name);

    // ----------------------------------------------------------------
    // Main simulation step
    // ----------------------------------------------------------------

    /// Advance the thread by one cycle.
    ///
    /// Processes all six pipeline stages in order.  Returns the routing
    /// result from the arbiter: either a (instruction, pipe_name) pair to
    /// send to the target pipe, or {nullptr, ""} if no instruction was
    /// ready this cycle.
    RouteResult step(int cycle);

    // ----------------------------------------------------------------
    // Per-stage step methods (public for unit testing)
    // ----------------------------------------------------------------

    /// T6.1: Fetch — walk PC, find instruction, push to input_buf_.
    /// Returns false if stalled (input_buf_ full or no kernel queued).
    bool step_fetch(int cycle);

    /// T6.2: Decode — pop from input_buf_, classify and execute.
    /// RISC instructions: execute via trisc_func, push to risc_check_buf_.
    /// TT / MOP: push to mop_buf_.
    /// Returns false if stalled (queues empty or full).
    bool step_decode(int cycle);

    /// T6.3a: Scoreboard check — pop from risc_check_buf_, apply branch
    /// penalty, check/set GPR scoreboard.  Pushes to risc_exec_trk_buf_.
    /// Returns false if stalled (waiting for scoreboard or penalty).
    bool step_scoreboard_check(int cycle);

    /// T6.3b: Scoreboard reset — pop from risc_exec_trk_buf_ after the
    /// configured pipeline latency, clear GPR scoreboard.
    /// Returns false if nothing to process.
    bool step_scoreboard_reset(int cycle);

    /// T6.4: MOP decode — pop from mop_buf_, expand MOP via tensix_func,
    /// emit one sub-instruction per cycle into instr_buf_.
    /// Returns false if stalled (nothing in mop_buf_ or instr_buf_ full).
    bool step_mop_decode(int cycle);

    /// T6.5: Arbiter — pop instruction (from instr_buf_ or replay buffer),
    /// handle replay state machine, execute TT instruction via tensix_func,
    /// perform pipe-stall / valid checks, insert into ROB, route to pipe.
    /// Returns the RouteResult (nullptr if stalled at any check).
    RouteResult step_arbiter(int cycle);

    // ----------------------------------------------------------------
    // Accessors (diagnostics and testing)
    // ----------------------------------------------------------------

    int  thread_id()    const { return cfg_.thread_id; }
    int  core_id()      const { return cfg_.core_id; }
    uint32_t pc()       const { return pc_; }

    bool fetch_done()   const { return fetch_done_; }
    bool all_done()     const;  ///< true when all stages are idle and queues empty

    int  input_buf_size()     const { return static_cast<int>(input_buf_.size()); }
    int  mop_buf_size()       const { return static_cast<int>(mop_buf_.size()); }
    int  instr_buf_size()     const { return static_cast<int>(instr_buf_.size()); }
    int  risc_check_buf_size() const { return static_cast<int>(risc_check_buf_.size()); }

    const Rob&         rob()          const { return rob_; }
    Rob&               mutable_rob()        { return rob_; } ///< for PipeUnit wiring in TensixCore
    const ReplayState& replay_state() const { return replay_; }

private:
    // ----------------------------------------------------------------
    // Internal helpers
    // ----------------------------------------------------------------

    /// Look up the instruction at pc in the active kernel's InstrMap.
    /// Returns nullptr if not found or no kernel is active.
    isa::InstrPtr find_ins(uint32_t addr) const;

    /// True when op is a memory-load instruction.
    static bool is_load_op(const std::string& op);

    /// True when op is a branch instruction (for misprediction penalty).
    static bool is_branch_op(const std::string& op);

    /// Attempt to acquire exe_pipe: if already free, set it busy and return
    /// true; else return false (caller stalls).
    bool try_acquire_exe_pipe(const std::string& pipe_name, int thread_id);

    /// Release exe_pipe (set idle) — called by the pipe completion callback
    /// (or directly in tests).  This mirrors Python's freeing of exe_pipe
    /// after the instruction leaves the arbiter.
    void release_exe_pipe(const std::string& pipe_name, int thread_id);

    // ----------------------------------------------------------------
    // Scoreboard check sub-state (T6.3a)
    // ----------------------------------------------------------------
    struct ScoreboardCheckState {
        isa::InstrPtr ins;
        int  branch_penalty_remaining = 0;
        bool src_checked  = false;
        bool dst_set      = false;
        bool ready        = false;
    };

    // ----------------------------------------------------------------
    // Arbiter sub-state (T6.5)
    // ----------------------------------------------------------------
    enum class ArbiterPhase : int {
        IDLE,                ///< waiting for instruction from instr_buf / replay
        STALL_DST_PIPES,     ///< setting dst pipes busy
        CHECK_SRC_PIPES,     ///< checking src pipes are idle
        WAIT_EXE_PIPE,       ///< acquiring exe pipe
        CHECK_VALIDS,        ///< checking TensixReg valid bits
        WAIT_ROB_HEAD,       ///< waiting for head-of-ROB
        ROUTING,             ///< ready to route to pipe this cycle
    };

    struct ArbiterState {
        ArbiterPhase  phase      = ArbiterPhase::IDLE;
        isa::InstrPtr ins;
        uint32_t      rob_id     = 0;
        int           src_consec = 0;  ///< consecutive-idle count for src pipe check
        int           exe_consec = 0;  ///< consecutive-idle count for exe pipe check
    };

    // ----------------------------------------------------------------
    // MOP decode sub-state (T6.4)
    // ----------------------------------------------------------------
    struct MopDecodeState {
        std::vector<uint32_t> words;  ///< expanded instruction words
        int                   idx = 0;
        bool                  active = false;
    };

    // ----------------------------------------------------------------
    // Data members
    // ----------------------------------------------------------------
    Config            cfg_;
    TensixFunc&       tensix_func_;
    risc::TriscFunc&  trisc_func_;
    TensixReg&        tensix_reg_;
    RiscReg&          risc_reg_;
    PipeResource&     pipe_resource_;
    TensixSplReg&     spl_regs_;
    risc::TriscRegs&  trisc_regs_;

    // Kernel table
    std::map<std::string, InstrMap>    kernel_instrs_;  ///< name → InstrMap
    std::map<std::string, KernelRange> kernel_ranges_;  ///< name → KernelRange
    std::deque<std::string>            kernel_queue_;   ///< pending kernel names

    // Execution state
    uint32_t    pc_          = 0;
    uint32_t    prev_pc_     = static_cast<uint32_t>(-1);
    std::string active_kernel_;
    bool        fetch_done_  = false;  ///< true once the active kernel finishes (PC→0)

    // Inter-stage queues (capacities match Python Store capacities)
    static constexpr int INPUT_BUF_CAP  = 2;
    static constexpr int MOP_BUF_CAP    = 1;
    static constexpr int INSTR_BUF_CAP  = 2;
    static constexpr int RCHECK_BUF_CAP = 1;

    std::deque<isa::InstrPtr> input_buf_;        ///< fetch → decode
    std::deque<isa::InstrPtr> mop_buf_;          ///< decode → mop_decode
    std::deque<isa::InstrPtr> instr_buf_;        ///< mop_decode → arbiter
    std::deque<isa::InstrPtr> risc_check_buf_;   ///< decode → scoreboard_check

    // Scoreboard pipeline
    ScoreboardCheckState          sboard_check_;
    std::deque<isa::InstrPtr>     risc_exec_trk_buf_;  ///< cap = risc_pipe_depth
    int                           exec_trk_delay_ = 0; ///< cycles until exec completes

    // MOP decode state
    MopDecodeState mop_state_;

    // Arbiter state
    ArbiterState arbiter_;
    Rob          rob_;
    ReplayState  replay_;
};

} // namespace neosim::units
