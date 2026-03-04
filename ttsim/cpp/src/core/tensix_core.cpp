// SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "neosim/core/tensix_core.hpp"
#include "neosim/isa/elf_loader.hpp"

#include <filesystem>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;

namespace neosim::core {

// ---------------------------------------------------------------------------
// Static config builder helpers
// ---------------------------------------------------------------------------

units::TensixSplReg::Config TensixCore::make_spl_reg_cfg(
    const config::SimConfig& cfg, int core_id)
{
    units::TensixSplReg::Config c;
    c.core_id               = core_id;
    c.cfg_start             = cfg.cfg_start;
    c.cfg_end               = cfg.cfg_end_exclusive;
    c.cfg_bytes_per_reg     = 4;
    c.instr_buf_start         = cfg.instr_buf_start;
    c.instr_buf_bytes_per_reg = 4;
    c.mop_start             = cfg.mop_start;
    c.pcbuf_start           = cfg.pcbuf_start;
    c.tile_cnt_start        = cfg.tile_cnt_start;
    c.tile_cnt_end          = cfg.tile_cnt_end_inclusive;
    c.tile_cnt_entry_bytes  = cfg.tile_cnt_entry_bytes;
    return c;
}

units::TensixFunc::Config TensixCore::make_tensix_func_cfg(
    const config::SimConfig& cfg, int core_id)
{
    units::TensixFunc::Config c;
    c.core_id   = core_id;
    c.arch      = cfg.arch;
    c.llk_group = 0; // TODO: map from llk_tag (feb20 → group 0)
    c.pipe_grps = cfg.pipe_grps;
    c.pipes     = cfg.pipe_names;
    return c;
}

units::Scratchpad::Config TensixCore::make_scratchpad_cfg(
    const config::SimConfig& cfg)
{
    units::Scratchpad::Config c;
    c.latency_rd  = static_cast<int>(cfg.latency_l1);
    c.latency_wr  = 1; // hardcoded per Python scratchpadRam
    c.num_cores   = 1;
    c.num_engines = static_cast<int>(cfg.engines.size());
    return c;
}

units::ThreadUnit::Config TensixCore::make_thread_cfg(
    const config::SimConfig& cfg, int core_id, int thread_id)
{
    units::ThreadUnit::Config c;
    c.core_id               = core_id;
    c.thread_id             = thread_id;
    c.risc_pipe_depth       = cfg.risc_pipe_depth;
    c.branch_mispredict_lat = cfg.branch_mispredict_lat;
    c.enable_scoreboard     = cfg.enable_scoreboard;
    c.enable_forwarding     = cfg.enable_forwarding;
    c.enable_in_order_issue = true;
    c.enable_pipe_stall     = true;
    c.enable_sync           = cfg.enable_sync;
    c.arch                  = cfg.arch;
    c.pipes                 = cfg.pipe_names;
    c.pipe_grps             = cfg.pipe_grps;
    return c;
}

units::PipeUnit::Config TensixCore::make_pipe_cfg(
    const config::SimConfig& cfg, int core_id, int thread_id, int pipe_idx)
{
    units::PipeUnit::Config c;
    c.pipe_id          = pipe_idx;
    c.pipe_name        = cfg.pipe_names[static_cast<std::size_t>(pipe_idx)];
    c.flavor           = engine_flavor(
        cfg.engines[static_cast<std::size_t>(pipe_idx)].grp);
    c.core_id          = core_id;
    c.thread_id        = thread_id;
    c.l1_port_width    = 128;  // DEFAULT_L1_PORT_WIDTH from plan
    c.reg_port_width   = 256;  // DEFAULT_REG_PORT_WIDTH from plan
    c.enable_pipe_stall = true;
    c.enable_sync       = cfg.enable_sync;
    c.pipes             = cfg.pipe_names;
    return c;
}

units::PipeUnit::Flavor TensixCore::engine_flavor(const std::string& grp)
{
    if (grp == "UNPACK") return units::PipeUnit::Flavor::UNPACK;
    if (grp == "PACK")   return units::PipeUnit::Flavor::PACK;
    return units::PipeUnit::Flavor::COMPUTE;
}

// ---------------------------------------------------------------------------
// PerThread constructor
// ---------------------------------------------------------------------------

TensixCore::PerThread::PerThread(
    const config::SimConfig& cfg,
    int                      core_id,
    int                      thread_id,
    units::TensixFunc&       tensix_func,
    units::TensixReg&        tensix_reg,
    units::RiscReg&          risc_reg,
    units::PipeResource&     pipe_resource,
    units::TensixSplReg&     spl_regs,
    units::Scratchpad&       scratchpad)
    : mem({cfg.arch})
    , regs({core_id,
            thread_id,
            static_cast<std::size_t>(thread_id) < cfg.stack_tops.size()
                ? cfg.stack_tops[static_cast<std::size_t>(thread_id)] : 0u,
            cfg.global_pointer})
    , trisc_func({core_id, thread_id}, mem, spl_regs, regs)
    , thread(TensixCore::make_thread_cfg(cfg, core_id, thread_id),
             tensix_func, trisc_func, tensix_reg, risc_reg,
             pipe_resource, spl_regs, regs)
{
    const int num_pipes = static_cast<int>(cfg.pipe_names.size());
    for (int p = 0; p < num_pipes; ++p) {
        pipes.push_back(std::make_unique<units::PipeUnit>(
            TensixCore::make_pipe_cfg(cfg, core_id, thread_id, p),
            tensix_reg,
            pipe_resource,
            thread.mutable_rob(),
            scratchpad));
    }
}

// ---------------------------------------------------------------------------
// TensixCore constructor
// ---------------------------------------------------------------------------

TensixCore::TensixCore(const config::SimConfig& cfg, int core_id)
    : core_id_(core_id)
    , cfg_(cfg)
    , spl_regs_(make_spl_reg_cfg(cfg, core_id))
    , tensix_reg_()
    , risc_reg_(cfg.num_threads)
    , pipe_resource_(static_cast<int>(cfg.pipe_names.size()), cfg.num_threads)
    , tensix_func_(make_tensix_func_cfg(cfg, core_id), spl_regs_)
    , scratchpad_(make_scratchpad_cfg(cfg))
{
    for (int t = 0; t < cfg.num_threads; ++t) {
        threads_.push_back(std::make_unique<PerThread>(
            cfg, core_id, t,
            tensix_func_, tensix_reg_, risc_reg_,
            pipe_resource_, spl_regs_, scratchpad_));
    }
}

// ---------------------------------------------------------------------------
// ELF loading
// ---------------------------------------------------------------------------

void TensixCore::load_thread_elf(int               thread_id,
                                  const std::string& elf_dir,
                                  const std::string& elf_name,
                                  const std::string& start_fn)
{
    if (elf_name.empty() || elf_dir.empty()) return;  // thread has no kernel

    const std::string elf_path = (fs::path(elf_dir) / elf_name).string();

    isa::ElfLoader loader(elf_path);
    if (!loader.is_valid()) {
        throw std::runtime_error(
            "TensixCore: failed to open ELF: " + elf_path);
    }

    // Decode the start function (all instructions up to the end of the function)
    std::vector<isa::Instruction> insns = loader.decode_function(start_fn);
    if (insns.empty()) {
        throw std::runtime_error(
            "TensixCore: function '" + start_fn + "' not found or empty in " + elf_path);
    }

    // Build InstrMap: address → InstrPtr, and annotate thread/core IDs
    units::ThreadUnit::InstrMap instr_map;
    uint32_t start_addr = insns.front().get_addr();
    uint32_t end_addr   = insns.back().get_addr() + 4u; // exclusive upper bound

    for (const auto& ins : insns) {
        auto ptr = std::make_shared<isa::Instruction>(ins);
        ptr->set_thread_id(static_cast<uint32_t>(thread_id));
        ptr->set_core_id(static_cast<uint32_t>(core_id_));
        instr_map[ptr->get_addr()] = ptr;
    }

    const std::string kernel_name = start_fn + "_t" + std::to_string(thread_id);

    auto& tu = threads_[static_cast<std::size_t>(thread_id)]->thread;
    tu.load_kernel(kernel_name, instr_map, start_addr, end_addr);
    tu.enqueue_kernel(kernel_name);
}

// ---------------------------------------------------------------------------
// Simulation step
// ---------------------------------------------------------------------------

int TensixCore::step(int cycle)
{
    int completed = 0;

    // 1. Step all pipes for all threads (pipes advance before threads see
    //    freed resources — mirrors Python SimPy event ordering).
    for (auto& pt : threads_) {
        for (auto& pipe : pt->pipes) {
            auto res = pipe->step(cycle);
            if (res.completed_ins) {
                ++completed;
                ++total_insns_;
                ++insns_per_pipe_[pipe->pipe_name()];
            }
        }
    }

    // 2. Step all thread units; collect routing results.
    for (std::size_t t = 0; t < threads_.size(); ++t) {
        auto result = threads_[t]->thread.step(cycle);
        if (!result.ins || result.target_pipe.empty()) continue;

        // 3. Route: push the instruction to the matching pipe for this thread.
        const auto& pnames = cfg_.pipe_names;
        for (std::size_t p = 0; p < pnames.size(); ++p) {
            if (pnames[p] == result.target_pipe) {
                threads_[t]->pipes[p]->push(result.ins);
                break;
            }
        }
    }

    return completed;
}

// ---------------------------------------------------------------------------
// Completion predicate
// ---------------------------------------------------------------------------

bool TensixCore::all_done() const
{
    for (const auto& pt : threads_) {
        if (!pt->thread.all_done()) return false;
        for (const auto& pipe : pt->pipes) {
            if (!pipe->is_idle()) return false;
        }
    }
    return true;
}

} // namespace neosim::core
