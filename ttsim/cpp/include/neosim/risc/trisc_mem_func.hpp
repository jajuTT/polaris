// SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>

namespace neosim::risc {

/// Non-L1 memory model for TRISC threads — replaces Python class `triscMemFunc`.
///
/// Flat address-value store (32-bit word granularity) pre-seeded with
/// hardware-default values used by LLK kernels.  Serves two purposes:
///   1. Generic heap / stack for RISC-V instructions that miss all MMR ranges.
///   2. Seed values for well-known addresses that kernels read before writing
///      (e.g. tile-counter region, WH eltwise-binary registers).
class TriscMemFunc {
public:
    struct Config {
        std::string arch; ///< "ttqs" — controls tile-counter pre-seed range
    };

    explicit TriscMemFunc(const Config& cfg);

    /// Write a 32-bit value to @p addr.
    void write_mem(uint32_t addr, int32_t val);

    /// Read a 32-bit value from @p addr.
    /// If the address is not initialised, auto-seeds it with 0xFF and returns 0xFF.
    int32_t read_mem(uint32_t addr);

    /// Returns true when @p addr has been written at least once.
    bool is_initialized(uint32_t addr) const;

private:
    void seed_defaults(const std::string& arch);

    std::unordered_map<uint32_t, int32_t> mem_;
};

} // namespace neosim::risc
