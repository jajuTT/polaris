// SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Integration test: end-to-end NeoSim run against feb20 LLK ELFs.
//
// Requires ELF files under POLARIS_DIR (defined by CMake at build time).
// Test: t6-quas-n1-ttx-elwadd-broadcast-scalar-fp16-llk
//   Python baseline: 1771 cycles  (±5% tolerance: 1682 – 1860)

#include "neosim/config/sim_config.hpp"
#include "neosim/core/neo_sim.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <string>

namespace fs = std::filesystem;

#ifndef POLARIS_DIR
#  error "POLARIS_DIR must be defined by CMake (-DPOLARIS_DIR=...)"
#endif

// Helper: convert POLARIS_DIR macro to std::string.
static const std::string kPolarisDirStr{POLARIS_DIR};

// ---------------------------------------------------------------------------
// Test fixture — verifies that required files exist before running tests.
// ---------------------------------------------------------------------------

class NeoSimIntegrationTest : public ::testing::Test {
protected:
    static constexpr const char* kLlkTag     = "feb20";
    static constexpr const char* kTestName   =
        "t6-quas-n1-ttx-elwadd-broadcast-scalar-fp16-llk";
    static constexpr int         kExpectedCycles = 1771; // Python baseline
    static constexpr double      kTolerance      = 0.05; // ±5 %

    std::string inputcfg_path() const {
        return kPolarisDirStr +
               "/__config_files/baseline/" + kLlkTag + "/inputcfg_" +
               kTestName + ".json";
    }

    bool files_available() const {
        // Check that at least the inputcfg and one ELF exist.
        if (!fs::exists(inputcfg_path())) return false;

        // Quick-check thread 0 ELF directory exists.
        const std::string elf_base =
            kPolarisDirStr +
            "/__ext/rtl_test_data_set/" + kLlkTag +
            "/rsim/debug/" + kTestName +
            "_0/ttx/kernels/core_00_00/neo_0/thread_0/out";
        return fs::exists(elf_base);
    }
};

// ---------------------------------------------------------------------------
// T8.4: Simulation completes without timeout.
// ---------------------------------------------------------------------------

TEST_F(NeoSimIntegrationTest, ElwaddCompletes)
{
    if (!files_available()) {
        GTEST_SKIP() << "LLK ELF files not found under " << kPolarisDirStr
                     << " (run the RTL data download first)";
    }

    neosim::config::SimConfig cfg;
    ASSERT_NO_THROW(cfg = neosim::config::SimConfig::load(
        inputcfg_path(),
        kPolarisDirStr   // base_dir for relative paths inside inputcfg
    ));

    neosim::core::NeoSim sim(cfg);
    neosim::core::NeoSim::Result result;
    ASSERT_NO_THROW(result = sim.run());

    EXPECT_GT(result.total_cycles, 0);
    ASSERT_EQ(static_cast<int>(result.cores.size()), cfg.num_cores);
}

// ---------------------------------------------------------------------------
// T8.6: Baseline cycle-count comparison (±5 %).
// ---------------------------------------------------------------------------

TEST_F(NeoSimIntegrationTest, ElwaddCycleCountWithinTolerance)
{
    if (!files_available()) {
        GTEST_SKIP() << "LLK ELF files not found under " << kPolarisDirStr;
    }

    neosim::config::SimConfig cfg;
    cfg = neosim::config::SimConfig::load(inputcfg_path(), kPolarisDirStr);

    neosim::core::NeoSim sim(cfg);
    const auto result = sim.run();

    const double ratio =
        static_cast<double>(result.total_cycles) /
        static_cast<double>(kExpectedCycles);

    EXPECT_GE(ratio, 1.0 - kTolerance)
        << "Cycle count " << result.total_cycles
        << " is more than 5% below Python baseline " << kExpectedCycles;

    EXPECT_LE(ratio, 1.0 + kTolerance)
        << "Cycle count " << result.total_cycles
        << " is more than 5% above Python baseline " << kExpectedCycles;
}

// ---------------------------------------------------------------------------
// T8.5: Per-pipe instruction counts sanity check.
// ---------------------------------------------------------------------------

TEST_F(NeoSimIntegrationTest, ElwaddHasInstructionsOnExpectedPipes)
{
    if (!files_available()) {
        GTEST_SKIP() << "LLK ELF files not found under " << kPolarisDirStr;
    }

    neosim::config::SimConfig cfg;
    cfg = neosim::config::SimConfig::load(inputcfg_path(), kPolarisDirStr);

    neosim::core::NeoSim sim(cfg);
    const auto result = sim.run();

    ASSERT_FALSE(result.cores.empty());
    const auto& core0 = result.cores[0];

    // The elwadd kernel must have executed instructions on MATH and UNPACK pipes.
    // Packer pipes should also have run.
    bool has_math_or_instissue = false;
    bool has_unpack            = false;
    bool has_pack              = false;

    for (const auto& [pipe, count] : core0.insns_per_pipe) {
        if (count == 0) continue;
        const std::string& n = pipe;
        if (n == "MATH" || n == "INSTISSUE") has_math_or_instissue = true;
        if (n.find("UNPACKER") != std::string::npos) has_unpack = true;
        if (n.find("PACKER")   != std::string::npos) has_pack   = true;
    }

    EXPECT_TRUE(has_math_or_instissue)
        << "Expected MATH/INSTISSUE instructions but got none";
    EXPECT_TRUE(has_unpack)
        << "Expected UNPACKER instructions but got none";
    EXPECT_TRUE(has_pack)
        << "Expected PACKER instructions but got none";
}
