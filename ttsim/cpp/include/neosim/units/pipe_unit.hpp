// SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "neosim/isa/instruction.hpp"
#include "neosim/units/pipe_resource.hpp"
#include "neosim/units/rob.hpp"
#include "neosim/units/scratchpad.hpp"
#include "neosim/units/tensix_reg.hpp"

#include <deque>
#include <string>
#include <vector>

namespace neosim::units {

/// Tensix pipe execution engine — replaces Python coroutines `tensixPipe`,
/// `unpacker`, and `packer` (t3sim.py).
///
/// One PipeUnit models a single execution pipe with a configurable Flavor
/// (COMPUTE, UNPACK, PACK).  Instructions are pushed by the arbiter (ThreadUnit)
/// and consumed one per simulated cycle via step().
///
/// The internal phase state machine mirrors the Python SimPy yield patterns:
///
///   COMPUTE:  IDLE → [CHECK_SRC_PIPES] → [CHECK_VALIDS] → EXECUTE → CLEANUP
///   UNPACK:   IDLE → [CHECK_SRC_PIPES] → L1_READ → [CHECK_VALIDS] → REG_WRITE → CLEANUP
///             (no memory: skip L1_READ/REG_WRITE)
///   PACK:     IDLE → [CHECK_SRC_PIPES] → FE_DELAY → [CHECK_VALIDS] → REG_READ → L1_WRITE → CLEANUP
///             (no memory: skip REG_READ/L1_WRITE)
///
/// Phases in [] are skipped when enable_pipe_stall / enable_sync are false.
class PipeUnit {
public:
    // ----------------------------------------------------------------
    // Flavor
    // ----------------------------------------------------------------
    enum class Flavor { COMPUTE, UNPACK, PACK };

    // ----------------------------------------------------------------
    // Constants
    // ----------------------------------------------------------------
    static constexpr int PACKERFE_DELAY_CYCLES = 3;
    static constexpr int PIPE_BUF_CAP          = 8;

    // ----------------------------------------------------------------
    // Configuration
    // ----------------------------------------------------------------
    struct Config {
        int         pipe_id   = 0;
        std::string pipe_name;
        Flavor      flavor    = Flavor::COMPUTE;
        int         core_id   = 0;
        int         thread_id = 0;       ///< default when pipes_thread_id not set
        int         l1_port_width  = 128;
        int         reg_port_width = 256;
        bool        enable_pipe_stall = true;
        bool        enable_sync       = true;
        std::vector<std::string> pipes; ///< all pipe names for resource lookup
    };

    // ----------------------------------------------------------------
    // Step result
    // ----------------------------------------------------------------
    struct StepResult {
        isa::InstrPtr completed_ins;    ///< non-null when instruction finishes
    };

    // ----------------------------------------------------------------
    // Construction
    // ----------------------------------------------------------------
    explicit PipeUnit(const Config& cfg, TensixReg& tensix_reg,
                      PipeResource& pipe_resource, Rob& rob,
                      Scratchpad& scratchpad);

    // ----------------------------------------------------------------
    // Main interface
    // ----------------------------------------------------------------

    /// Push an instruction into the pipe buffer (called by arbiter).
    void       push(isa::InstrPtr ins);

    /// Advance one simulated cycle.  Returns completed_ins when an
    /// instruction finishes all phases this cycle; otherwise nullptr.
    StepResult step(int cycle);

    // ----------------------------------------------------------------
    // Accessors
    // ----------------------------------------------------------------
    int                pipe_id()   const { return cfg_.pipe_id; }
    const std::string& pipe_name() const { return cfg_.pipe_name; }

    /// True when no instruction is active and the buffer is empty.
    bool is_idle()   const;

    /// Number of instructions waiting in the buffer (not counting active).
    int  buf_size()  const;

private:
    // ----------------------------------------------------------------
    // Internal phase enum
    // ----------------------------------------------------------------
    enum class Phase {
        IDLE,
        CHECK_SRC_PIPES,
        FE_DELAY,
        CHECK_VALIDS,
        EXECUTE,
        L1_READ,
        REG_WRITE,
        REG_READ,
        L1_WRITE,
        CLEANUP,
    };

    // ----------------------------------------------------------------
    // Phase transition helpers
    // ----------------------------------------------------------------

    /// True when the active instruction accesses L1/REG memory
    /// (not POP_TILES for UNPACK, not PUSH_TILES for PACK).
    bool has_memory() const;

    /// Set up cycles_remaining_ and any scratchpad side-effects for a new phase.
    void enter_phase(Phase p);

    /// Next phase after CHECK_SRC_PIPES completes.
    Phase next_after_src_pipes() const;

    /// Next phase after FE_DELAY completes (PACK only).
    Phase next_after_fe_delay()  const;

    /// Next phase after CHECK_VALIDS completes.
    Phase next_after_valids()    const;

    // ----------------------------------------------------------------
    // Phase execution helpers
    // ----------------------------------------------------------------

    /// Execute CHECK_SRC_PIPES: check all src_pipes are idle.
    /// Returns true when all conditions satisfied (phase may advance).
    bool do_check_src_pipes();

    /// Execute CHECK_VALIDS: check TensixReg valid bits for active_ins_.
    /// Returns true when all conditions satisfied (phase may advance).
    bool do_check_valids();

    /// Apply pipe cleanup: update valid bits, release pipe resources,
    /// remove from ROB.  Mirrors Python _common_pipe_cleanup.
    void apply_cleanup();

    // ----------------------------------------------------------------
    // Utilities
    // ----------------------------------------------------------------

    /// Find the index of a pipe by name in cfg_.pipes.  Returns -1 if not found.
    int find_pipe_id(const std::string& name) const;

    /// Return the thread ID to use for this instruction's pipe resource checks.
    int thread_of_ins() const;

    /// Round bytes up to the nearest multiple of port_width.
    static uint32_t align_bytes(uint32_t bytes, uint32_t port_width);

    // ----------------------------------------------------------------
    // Data members
    // ----------------------------------------------------------------
    Config        cfg_;
    TensixReg&    tensix_reg_;
    PipeResource& pipe_resource_;
    Rob&          rob_;
    Scratchpad&   scratchpad_;

    std::deque<isa::InstrPtr> pipe_buf_;         ///< waiting instructions
    Phase         phase_            = Phase::IDLE;
    isa::InstrPtr active_ins_;                   ///< instruction being processed
    int           cycles_remaining_ = 0;         ///< countdown for timed phases
    uint32_t      pending_req_id_   = 0;         ///< in-flight scratchpad req
    int           src_consec_       = 0;         ///< consecutive-idle count for src pipes
};

} // namespace neosim::units
