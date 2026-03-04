// SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "neosim/config/sim_config.hpp"
#include "neosim/core/tensix_core.hpp"

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace neosim::core {

/// Top-level NeoSim runner — Track 8 step-based integration.
///
/// Loads ELFs from a SimConfig, runs the cycle loop until all cores are done
/// (or a timeout fires), and returns cycle count and per-pipe statistics.
///
/// Example usage:
/// @code
///   auto cfg  = neosim::config::SimConfig::load(inputcfg_path, polaris_dir);
///   auto sim  = neosim::core::NeoSim(cfg);
///   auto res  = sim.run();
///   std::cout << "Cycles = " << res.total_cycles << "\n";
/// @endcode
class NeoSim {
public:
    /// Per-core simulation results.
    struct CoreResult {
        int                        total_insns = 0;
        std::map<std::string, int> insns_per_pipe;
    };

    /// Aggregate simulation results.
    struct Result {
        int                     total_cycles = 0;
        std::vector<CoreResult> cores;
    };

    /// Maximum simulated cycles before aborting (mirrors Python MAXTIMER_HIGH).
    static constexpr int MAXTIMER_HIGH = 25000;

    explicit NeoSim(const config::SimConfig& cfg);

    /// Run the simulation to completion.
    ///
    /// @throws std::runtime_error if simulation exceeds MAXTIMER_HIGH cycles.
    Result run();

private:
    config::SimConfig                    cfg_;
    std::vector<std::unique_ptr<TensixCore>> cores_;
};

} // namespace neosim::core
