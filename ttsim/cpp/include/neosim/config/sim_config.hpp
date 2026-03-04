// SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "neosim/units/tensix_func.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace neosim::config {

/// Configuration for a single Tensix pipe engine (from engines[].{engineName,engineGrp,delay}).
struct EngineConfig {
    std::string name;    ///< engineName (e.g. "MATH", "UNPACKER0", "PACKER0")
    std::string grp;     ///< engineGrp  (e.g. "MATH", "UNPACK", "PACK", "CFG", …)
    int         delay = 10; ///< instruction execution latency in cycles
};

/// Simulation configuration aggregated from three JSON files:
///   1. inputcfg_*.json         — ELF paths, thread count, LLK version tag
///   2. ttqs_neo4_*.json        — pipeline flags, engine latencies, RISC config
///   3. ttqs_memory_map_*.json  — MMR address ranges for TensixSplReg
///
/// Mirrors the Python `args_dict` built by `tneoSim.py`.
struct SimConfig {
    // ── From inputcfg ─────────────────────────────────────────────────────
    std::string arch           = "ttqs"; ///< architecture tag (auto-detected from ELF)
    std::string llk_tag;                  ///< llkVersionTag (e.g. "feb20")
    int         num_cores   = 1;          ///< numTCores
    int         num_threads = 0;          ///< input.tc0.numThreads
    std::string start_function;           ///< input.tc0.startFunction (e.g. "_start")

    /// Per-thread ELF location.
    struct ThreadInput {
        std::string elf_name; ///< th{N}Elf  (filename, e.g. "thread_0.elf")
        std::string elf_dir;  ///< th{N}Path (directory containing the ELF)
    };
    std::vector<ThreadInput> threads; ///< [0 .. num_threads-1]

    // ── From engine config (ttqs_neo4_*.json) ─────────────────────────────
    std::vector<EngineConfig> engines;           ///< one entry per pipe engine (in order)
    bool    enable_sync             = true;      ///< enableSync
    int     risc_pipe_depth         = 4;         ///< riscPipeDepth
    int     branch_mispredict_lat   = 3;         ///< branchMisPredictPenalty
    bool    enable_scoreboard       = true;      ///< enableScoreboardCheckforRegs
    bool    enable_forwarding       = true;      ///< enableForwardingforRegs
    float   latency_l1              = 10.0f;     ///< latency_l1
    bool    enable_shared_l1        = true;      ///< enableSharedL1
    int     max_threads_per_core    = 4;         ///< maxNumThreadsperNeoCore
    std::vector<uint32_t> stack_tops;            ///< initial SP per thread (stack[N][0])
    uint32_t global_pointer         = 0u;        ///< globalPointer

    // ── From memory map (ttqs_memory_map_*.json) ──────────────────────────
    uint32_t cfg_start              = 0x820000u; ///< trisc_map.cfg_regs.START
    uint32_t cfg_end_exclusive      = 0x820de0u; ///< trisc_map.cfg_regs.END + 1
    uint32_t instr_buf_start        = 0x80f000u; ///< trisc_map.ibuffer.START
    uint32_t mop_start              = 0x80e000u; ///< trisc_map.mop_cfg.START
    uint32_t pcbuf_start            = 0x810000u; ///< trisc_map.pcbuffer.START
    uint32_t tile_cnt_start         = 0x80c000u; ///< trisc_map.tile_counters.START
    uint32_t tile_cnt_end_inclusive = 0x80c3ffu; ///< trisc_map.tile_counters.END (inclusive)
    uint32_t tile_cnt_entry_bytes   = 32u;        ///< bytes per tile-counter entry

    // ── Derived from engines ───────────────────────────────────────────────
    std::vector<std::string>      pipe_names; ///< ordered pipe names (index = pipe ID)
    units::TensixFunc::PipeGroups pipe_grps;  ///< grp_name -> [pipe_names]

    // ── Factory ───────────────────────────────────────────────────────────

    /// Load and merge configuration from three JSON files.
    ///
    /// @param inputcfg_path  Path to the inputcfg_*.json file.
    /// @param base_dir       Directory used to resolve relative paths found inside
    ///                       inputcfg (th*Path, engine cfg, memory map, etc.).
    ///                       When empty, defaults to the parent of inputcfg_path.
    /// @param engine_cfg     Explicit path to the engine config JSON.
    ///                       When empty, derived from llkVersionTag.
    /// @param mem_map        Explicit path to the memory map JSON.
    ///                       When empty, derived from llkVersionTag.
    ///
    /// @throws std::runtime_error if any required file is missing or malformed.
    static SimConfig load(const std::string& inputcfg_path,
                          const std::string& base_dir   = {},
                          const std::string& engine_cfg = {},
                          const std::string& mem_map    = {});
};

} // namespace neosim::config
