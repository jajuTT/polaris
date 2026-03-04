// SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Stand-alone NeoSim example — mirrors the integration test T8.4–T8.6
// but without GoogleTest, so print statements can be freely added for debugging.
//
// Usage:
//   ./neo_sim_example [polaris_dir]
//   POLARIS_DIR=/path/to/polaris ./neo_sim_example
//   ./neo_sim_example  # uses POLARIS_DIR compile-time define (set by CMake)
//
// Test: t6-quas-n1-ttx-elwadd-broadcast-scalar-fp16-llk
//   Python baseline: 1771 cycles  (±5% tolerance: 1682–1860)

#include "neosim/config/sim_config.hpp"
#include "neosim/core/neo_sim.hpp"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>

namespace fs = std::filesystem;

#ifndef POLARIS_DIR
#  define POLARIS_DIR ""
#endif

static constexpr const char* kTestName =
    "t6-quas-n1-ttx-elwadd-broadcast-scalar-fp16-llk";
static constexpr const char* kLlkTag = "feb20";
static constexpr int kExpectedCycles = 1771;
static constexpr double kTolerance   = 0.05;

int main(int argc, char* argv[])
{
    // ── Resolve POLARIS_DIR ───────────────────────────────────────────────
    std::string polaris_dir;
    if (argc >= 2) {
        polaris_dir = argv[1];
    } else if (const char* env = std::getenv("POLARIS_DIR")) {
        polaris_dir = env;
    } else {
        polaris_dir = std::string(POLARIS_DIR);
    }

    if (polaris_dir.empty()) {
        std::cerr << "Error: POLARIS_DIR not set.\n"
                  << "  Usage: " << argv[0] << " <polaris_dir>\n"
                  << "  or:    export POLARIS_DIR=...\n";
        return 1;
    }

    const std::string inputcfg_path =
        polaris_dir + "/__config_files/baseline/" +
        kLlkTag + "/inputcfg_" + kTestName + ".json";

    std::cout << "POLARIS_DIR : " << polaris_dir << "\n";
    std::cout << "inputcfg    : " << inputcfg_path << "\n\n";

    // ── Load config ───────────────────────────────────────────────────────
    neosim::config::SimConfig cfg;
    try {
        cfg = neosim::config::SimConfig::load(inputcfg_path, polaris_dir);
    } catch (const std::exception& e) {
        std::cerr << "Config load failed: " << e.what() << "\n";
        return 1;
    }

    std::cout << "Config:\n"
              << "  arch        = " << cfg.arch << "\n"
              << "  llk_tag     = " << cfg.llk_tag << "\n"
              << "  num_cores   = " << cfg.num_cores << "\n"
              << "  num_threads = " << cfg.num_threads << "\n"
              << "  enable_sync = " << cfg.enable_sync << "\n"
              << "  risc_pipe_depth = " << cfg.risc_pipe_depth << "\n"
              << "  latency_l1  = " << cfg.latency_l1 << "\n";

    std::cout << "\nEngines (" << cfg.engines.size() << "):\n";
    for (const auto& e : cfg.engines) {
        std::cout << "  [" << e.name << "]  grp=" << e.grp
                  << "  delay=" << e.delay << "\n";
    }

    std::cout << "\nThreads:\n";
    for (int t = 0; t < cfg.num_threads; ++t) {
        const auto& ti = cfg.threads[static_cast<std::size_t>(t)];
        std::cout << "  thread " << t << ": "
                  << (ti.elf_name.empty() ? "(no ELF)" : ti.elf_name)
                  << "\n";
    }
    std::cout << "\n";

    // ── Run simulation ────────────────────────────────────────────────────
    std::cout << "Running simulation...\n";

    neosim::core::NeoSim sim(cfg);
    neosim::core::NeoSim::Result result;
    try {
        result = sim.run();
    } catch (const std::exception& e) {
        std::cerr << "Simulation failed: " << e.what() << "\n";
        return 1;
    }

    // ── Print results ─────────────────────────────────────────────────────
    std::cout << "\n=== Results ===\n";
    std::cout << "Total cycles : " << result.total_cycles << "\n";
    std::cout << "Expected     : " << kExpectedCycles
              << "  (±" << static_cast<int>(kTolerance * 100) << "% → "
              << static_cast<int>(kExpectedCycles * (1.0 - kTolerance)) << "–"
              << static_cast<int>(kExpectedCycles * (1.0 + kTolerance)) << ")\n";

    const double ratio = static_cast<double>(result.total_cycles) /
                         static_cast<double>(kExpectedCycles);
    if (ratio < 1.0 - kTolerance) {
        std::cout << "  WARN: cycle count is more than 5% BELOW baseline\n";
    } else if (ratio > 1.0 + kTolerance) {
        std::cout << "  WARN: cycle count is more than 5% ABOVE baseline\n";
    } else {
        std::cout << "  OK: within ±5% tolerance\n";
    }

    for (int c = 0; c < static_cast<int>(result.cores.size()); ++c) {
        const auto& cr = result.cores[c];
        std::cout << "\nCore " << c << "  total_insns=" << cr.total_insns << "\n";
        std::cout << "  insns_per_pipe:\n";
        bool has_math = false, has_unpack = false, has_pack = false;
        for (const auto& [pipe, count] : cr.insns_per_pipe) {
            if (count == 0) continue;
            std::cout << "    " << pipe << ": " << count << "\n";
            if (pipe == "MATH" || pipe == "INSTISSUE") has_math = true;
            if (pipe.find("UNPACKER") != std::string::npos) has_unpack = true;
            if (pipe.find("PACKER")   != std::string::npos) has_pack   = true;
        }
        std::cout << "  Pipe presence check: MATH/INSTISSUE="
                  << (has_math   ? "OK" : "MISSING") << "  UNPACKER="
                  << (has_unpack ? "OK" : "MISSING") << "  PACKER="
                  << (has_pack   ? "OK" : "MISSING") << "\n";
    }

    return 0;
}
