// SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "neosim/config/sim_config.hpp"
#include "neosim/isa/elf_loader.hpp"
#include "neosim/risc/trisc_func.hpp"
#include "neosim/risc/trisc_mem_func.hpp"
#include "neosim/risc/trisc_regs.hpp"
#include "neosim/units/pipe_resource.hpp"
#include "neosim/units/pipe_unit.hpp"
#include "neosim/units/risc_reg.hpp"
#include "neosim/units/scratchpad.hpp"
#include "neosim/units/tensix_func.hpp"
#include "neosim/units/tensix_reg.hpp"
#include "neosim/units/tensix_spl_reg.hpp"
#include "neosim/units/thread_unit.hpp"

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace neosim::core {

/// Step-based Tensix core driver — Track 8 functional integration.
///
/// Owns all unit objects for one Tensix Neo core:
///   - Shared across threads: TensixSplReg, TensixReg, RiscReg, PipeResource,
///                            TensixFunc, Scratchpad.
///   - Per-thread: TriscMemFunc, TriscRegs, TriscFunc, ThreadUnit.
///   - Per-thread per-pipe: PipeUnit (one PipeUnit per thread per engine, mirroring
///     the Python model where each thread owns its own set of pipe coroutines).
///
/// The step-ordering per cycle mirrors the Python SimPy event ordering:
///   1. Pipe units step first  → completed instructions free resources.
///   2. Thread units step next → arbiters see freed resources and can issue.
///   3. Route: push each thread's RouteResult to the target PipeUnit.
///
/// No Sparta dependency — pure C++20, buildable on macOS ARM64 for development.
class TensixCore {
public:
    explicit TensixCore(const config::SimConfig& cfg, int core_id = 0);

    // ── ELF loading ────────────────────────────────────────────────────────

    /// Load and decode an ELF for a given thread.
    ///
    /// Finds the start_function in the ELF, builds the InstrMap, and enqueues
    /// the kernel for execution.  Threads with an empty elf_path are skipped
    /// (they stay idle and contribute to all_done() immediately).
    void load_thread_elf(int               thread_id,
                         const std::string& elf_dir,
                         const std::string& elf_name,
                         const std::string& start_fn);

    // ── Simulation ─────────────────────────────────────────────────────────

    /// Advance the core by one simulated cycle.
    ///
    /// Returns the number of Tensix instructions that completed this cycle.
    int step(int cycle);

    /// True when all threads are fully done (fetch_done and all queues empty)
    /// and all pipes are idle.
    bool all_done() const;

    // ── Statistics ─────────────────────────────────────────────────────────

    int                              total_instructions()  const { return total_insns_; }
    const std::map<std::string, int>& insns_per_pipe()     const { return insns_per_pipe_; }
    int                              core_id()             const { return core_id_; }

private:
    // ── Config helpers (static to avoid depending on object state) ─────────

    static units::TensixSplReg::Config make_spl_reg_cfg(
        const config::SimConfig& cfg, int core_id);

    static units::TensixFunc::Config make_tensix_func_cfg(
        const config::SimConfig& cfg, int core_id);

    static units::Scratchpad::Config make_scratchpad_cfg(
        const config::SimConfig& cfg);

    static units::ThreadUnit::Config make_thread_cfg(
        const config::SimConfig& cfg, int core_id, int thread_id);

    static units::PipeUnit::Config make_pipe_cfg(
        const config::SimConfig& cfg, int core_id, int thread_id, int pipe_idx);

    static units::PipeUnit::Flavor engine_flavor(const std::string& grp);

    // ── Shared units (initialization order matches constructor order) ───────
    int core_id_;
    config::SimConfig cfg_;

    units::TensixSplReg  spl_regs_;
    units::TensixReg     tensix_reg_;
    units::RiscReg       risc_reg_;
    units::PipeResource  pipe_resource_;
    units::TensixFunc    tensix_func_;
    units::Scratchpad    scratchpad_;

    // ── Per-thread state ────────────────────────────────────────────────────

    /// All objects needed for one thread.
    /// Members must be declared in construction order (dependencies first).
    struct PerThread {
        risc::TriscMemFunc  mem;
        risc::TriscRegs     regs;
        risc::TriscFunc     trisc_func;
        units::ThreadUnit   thread;

        /// Per-pipe execution units for this thread (index = pipe_id).
        std::vector<std::unique_ptr<units::PipeUnit>> pipes;

        PerThread(const config::SimConfig& cfg,
                  int                      core_id,
                  int                      thread_id,
                  units::TensixFunc&       tensix_func,
                  units::TensixReg&        tensix_reg,
                  units::RiscReg&          risc_reg,
                  units::PipeResource&     pipe_resource,
                  units::TensixSplReg&     spl_regs,
                  units::Scratchpad&       scratchpad);
    };

    std::vector<std::unique_ptr<PerThread>> threads_;

    // ── Statistics ──────────────────────────────────────────────────────────
    std::map<std::string, int> insns_per_pipe_;
    int                        total_insns_ = 0;
};

} // namespace neosim::core
