// SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "neosim/risc/trisc_mem_func.hpp"

#include <cassert>

namespace neosim::risc {

TriscMemFunc::TriscMemFunc(const Config& cfg)
{
    seed_defaults(cfg.arch);
}

void TriscMemFunc::seed_defaults(const std::string& arch)
{
    // Addresses read by LLK kernels before being written — pre-seeded to
    // avoid uninitialised-memory assertions.  Values mirror the Python
    // triscMemFunc.__init__ defaults.
    mem_[0xFFE80034u] = 0;

    // Double-indirection pair: mem[0xFFB00010] = 111; mem[111] = 1
    mem_[0xFFB00010u] = 111;
    mem_[111u]        = 1;

    mem_[0xFFB006CCu] = 100;
    mem_[0x80F084u]   = 0;

    // WH eltwise-binary seed registers
    mem_[0xFFB48010u] = static_cast<int32_t>(0xFFFFFFFF);
    mem_[0xFFB49010u] = static_cast<int32_t>(0xFFFFFFFF);
    mem_[0xFFB48028u] = static_cast<int32_t>(0xFFFFFFFF);
    mem_[0xFFB49028u] = static_cast<int32_t>(0xFFFFFFFF);
    mem_[0xFFB4A028u] = static_cast<int32_t>(0xFFFFFFFF);

    mem_[0x2A3u]      = 0x80;
    mem_[0xFFE80024u] = 0x0;

    if (arch == "ttqs") {
        // Pre-zero the tile-counter memory region so LLK while-loops terminate.
        // TILE_COUNTERS_START = 0x0080b000, END = 0x0080c000, step = 4 bytes.
        for (uint32_t a = 0x0080b000u; a < 0x0080c000u; a += 4u) {
            mem_[a] = 0;
        }
    }
}

void TriscMemFunc::write_mem(uint32_t addr, int32_t val)
{
    mem_[addr] = val;
}

int32_t TriscMemFunc::read_mem(uint32_t addr)
{
    auto it = mem_.find(addr);
    if (it == mem_.end()) {
        // Lazy-init: mirrors Python "Initializing memory which should have been available"
        mem_[addr] = 0xFF;
        return 0xFF;
    }
    return it->second;
}

bool TriscMemFunc::is_initialized(uint32_t addr) const
{
    return mem_.count(addr) != 0;
}

} // namespace neosim::risc
