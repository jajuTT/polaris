// SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "neosim/core/neo_sim.hpp"

#include <stdexcept>
#include <string>

namespace neosim::core {

NeoSim::NeoSim(const config::SimConfig& cfg) : cfg_(cfg)
{
    for (int c = 0; c < cfg_.num_cores; ++c) {
        cores_.push_back(std::make_unique<TensixCore>(cfg_, c));
    }
}

NeoSim::Result NeoSim::run()
{
    // Load ELFs for all cores and threads.
    // Each core gets the same ELF set (single-core config repeats; multi-core
    // configs have per-core th*Path entries — for now all cores share tc0).
    for (int c = 0; c < static_cast<int>(cores_.size()); ++c) {
        for (int t = 0; t < cfg_.num_threads; ++t) {
            const auto& ti = cfg_.threads[static_cast<std::size_t>(t)];
            if (ti.elf_name.empty()) continue;
            cores_[static_cast<std::size_t>(c)]->load_thread_elf(
                t, ti.elf_dir, ti.elf_name, cfg_.start_function);
        }
    }

    // Cycle loop: run until all cores are done or timeout fires.
    for (int cycle = 0; cycle < MAXTIMER_HIGH; ++cycle) {
        for (auto& core : cores_) {
            core->step(cycle);
        }

        bool all_done = true;
        for (const auto& core : cores_) {
            if (!core->all_done()) { all_done = false; break; }
        }

        if (all_done) {
            Result r;
            r.total_cycles = cycle + 1;
            for (const auto& core : cores_) {
                CoreResult cr;
                cr.total_insns    = core->total_instructions();
                cr.insns_per_pipe = core->insns_per_pipe();
                r.cores.push_back(cr);
            }
            return r;
        }
    }

    throw std::runtime_error(
        "NeoSim: simulation timed out after " +
        std::to_string(MAXTIMER_HIGH) + " cycles");
}

} // namespace neosim::core
