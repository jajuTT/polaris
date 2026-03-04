// SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "neosim/config/sim_config.hpp"

#include <nlohmann/json.hpp>

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;
using json   = nlohmann::json;

namespace {

/// Read and parse a JSON file; throw on error.
json read_json(const std::string& path)
{
    std::ifstream f(path);
    if (!f.is_open()) {
        throw std::runtime_error("SimConfig: cannot open file: " + path);
    }
    try {
        return json::parse(f);
    } catch (const json::parse_error& e) {
        throw std::runtime_error("SimConfig: JSON parse error in '" + path + "': " + e.what());
    }
}

/// Return the directory part of a path (equivalent to Python os.path.dirname).
std::string dir_of(const std::string& path)
{
    return fs::path(path).parent_path().string();
}

/// Resolve a path that may be relative to base_dir.
std::string resolve(const std::string& p, const std::string& base_dir)
{
    if (p.empty()) return p;
    fs::path fp(p);
    if (fp.is_absolute()) return p;
    return (fs::path(base_dir) / fp).string();
}

/// Parse the inputcfg JSON → populate SimConfig fields for inputcfg section.
void parse_inputcfg(const json& j, const std::string& base_dir, neosim::config::SimConfig& cfg)
{
    // llkVersionTag
    if (j.contains("llkVersionTag") && j["llkVersionTag"].is_string()) {
        cfg.llk_tag = j["llkVersionTag"].get<std::string>();
    }

    // numTCores
    if (j.contains("numTCores")) {
        cfg.num_cores = j["numTCores"].get<int>();
    }

    // input.tc0
    if (!j.contains("input") || !j["input"].contains("tc0")) {
        throw std::runtime_error("SimConfig: inputcfg missing 'input.tc0'");
    }
    const auto& tc0 = j["input"]["tc0"];

    cfg.num_threads    = tc0.value("numThreads", 0);
    cfg.start_function = tc0.value("startFunction", std::string("_start"));

    // Per-thread ELF info: th0..th3
    cfg.threads.clear();
    for (int t = 0; t < cfg.num_threads; ++t) {
        std::string elf_key  = "th" + std::to_string(t) + "Elf";
        std::string path_key = "th" + std::to_string(t) + "Path";
        std::string elf_name = tc0.value(elf_key,  std::string{});
        std::string elf_dir  = tc0.value(path_key, std::string{});

        // Resolve relative ELF dir against base_dir
        if (!elf_dir.empty()) {
            elf_dir = resolve(elf_dir, base_dir);
        }

        cfg.threads.push_back({elf_name, elf_dir});
    }
}

/// Parse the engine config JSON → populate SimConfig fields for engine section.
void parse_engine_cfg(const json& j, neosim::config::SimConfig& cfg)
{
    cfg.enable_sync           = j.value("enableSync",                 1) != 0;
    cfg.risc_pipe_depth       = j.value("riscPipeDepth",              4);
    cfg.branch_mispredict_lat = j.value("branchMisPredictPenalty",    3);
    cfg.enable_scoreboard     = j.value("enableScoreboardCheckforRegs", 1) != 0;
    cfg.enable_forwarding     = j.value("enableForwardingforRegs",    1) != 0;
    cfg.latency_l1            = j.value("latency_l1",                 10.0f);
    cfg.enable_shared_l1      = j.value("enableSharedL1",             1) != 0;
    cfg.max_threads_per_core  = j.value("maxNumThreadsperNeoCore",    4);

    // globalPointer: hex string
    if (j.contains("globalPointer") && j["globalPointer"].is_string()) {
        cfg.global_pointer = static_cast<uint32_t>(
            std::stoul(j["globalPointer"].get<std::string>(), nullptr, 0));
    }

    // stack: object keyed by thread index, value = [top, bottom]
    cfg.stack_tops.assign(static_cast<std::size_t>(cfg.num_threads), 0u);
    if (j.contains("stack") && j["stack"].is_object()) {
        for (const auto& [k, v] : j["stack"].items()) {
            int t = std::stoi(k);
            if (t >= 0 && t < cfg.num_threads && v.is_array() && v.size() >= 1) {
                std::string top_str = v[0].get<std::string>();
                cfg.stack_tops[static_cast<std::size_t>(t)] =
                    static_cast<uint32_t>(std::stoul(top_str, nullptr, 0));
            }
        }
    }

    // engines → pipe_names, pipe_grps, engines vector
    cfg.engines.clear();
    cfg.pipe_names.clear();
    cfg.pipe_grps.clear();

    if (j.contains("engines") && j["engines"].is_array()) {
        for (const auto& e : j["engines"]) {
            neosim::config::EngineConfig ec;
            ec.name  = e.value("engineName", std::string{});
            ec.grp   = e.value("engineGrp",  std::string{});
            ec.delay = e.value("delay",       10);
            cfg.engines.push_back(ec);

            cfg.pipe_names.push_back(ec.name);
            cfg.pipe_grps[ec.grp].push_back(ec.name);
        }
    }
}

/// Parse the memory map JSON → populate SimConfig fields for MMR addresses.
void parse_memory_map(const json& j, neosim::config::SimConfig& cfg)
{
    if (!j.contains("trisc_map")) {
        throw std::runtime_error("SimConfig: memory map missing 'trisc_map'");
    }
    const auto& tm = j["trisc_map"];

    auto hex_addr = [](const json& region, const std::string& key) -> uint32_t {
        if (region.contains(key)) {
            const auto& v = region[key];
            if (v.is_string()) {
                return static_cast<uint32_t>(std::stoul(v.get<std::string>(), nullptr, 0));
            }
            if (v.is_number_unsigned()) {
                return v.get<uint32_t>();
            }
            if (v.is_number_integer()) {
                return static_cast<uint32_t>(v.get<int64_t>());
            }
        }
        return 0u;
    };

    if (tm.contains("cfg_regs")) {
        const auto& r = tm["cfg_regs"];
        cfg.cfg_start         = hex_addr(r, "START");
        cfg.cfg_end_exclusive = hex_addr(r, "END") + 1u;  // END is inclusive in JSON
    }
    if (tm.contains("ibuffer")) {
        cfg.instr_buf_start = hex_addr(tm["ibuffer"], "START");
    }
    if (tm.contains("mop_cfg")) {
        cfg.mop_start = hex_addr(tm["mop_cfg"], "START");
    }
    if (tm.contains("pcbuffer")) {
        cfg.pcbuf_start = hex_addr(tm["pcbuffer"], "START");
    }
    if (tm.contains("tile_counters")) {
        const auto& tc = tm["tile_counters"];
        cfg.tile_cnt_start         = hex_addr(tc, "START");
        cfg.tile_cnt_end_inclusive = hex_addr(tc, "END");
    }
    // Compute tile_cnt_entry_bytes from tile_counters.counters[0] span
    if (tm.contains("tile_counters.counters[0]")) {
        const auto& c0 = tm["tile_counters.counters[0]"];
        uint32_t start = hex_addr(c0, "START");
        uint32_t end   = hex_addr(c0, "END");
        if (end >= start) {
            cfg.tile_cnt_entry_bytes = end - start + 1u;
        }
    }
}

} // anonymous namespace

namespace neosim::config {

SimConfig SimConfig::load(const std::string& inputcfg_path,
                          const std::string& base_dir_arg,
                          const std::string& engine_cfg_arg,
                          const std::string& mem_map_arg)
{
    SimConfig cfg;

    // ── 1. Parse inputcfg ────────────────────────────────────────────────
    const json inputcfg = read_json(inputcfg_path);

    // Resolve base_dir: use explicit arg, else directory containing inputcfg
    const std::string base_dir =
        base_dir_arg.empty() ? dir_of(inputcfg_path) : base_dir_arg;

    parse_inputcfg(inputcfg, base_dir, cfg);

    // ── 2. Resolve engine config path ────────────────────────────────────
    std::string engine_cfg_path = engine_cfg_arg;
    if (engine_cfg_path.empty()) {
        if (cfg.llk_tag.empty()) {
            throw std::runtime_error(
                "SimConfig: 'llkVersionTag' not found in inputcfg and no "
                "engine_cfg path provided");
        }
        engine_cfg_path = resolve(
            "config/tensix_neo/ttqs_neo4_" + cfg.llk_tag + ".json", base_dir);
    }

    // ── 3. Parse engine config ───────────────────────────────────────────
    const json eng_json = read_json(engine_cfg_path);
    parse_engine_cfg(eng_json, cfg);

    // Resize stack_tops to num_threads if not already done (parse_engine_cfg
    // uses num_threads, which was set by parse_inputcfg above).
    if (cfg.stack_tops.size() < static_cast<std::size_t>(cfg.num_threads)) {
        cfg.stack_tops.resize(static_cast<std::size_t>(cfg.num_threads), 0u);
    }

    // ── 4. Resolve memory map path ───────────────────────────────────────
    std::string mem_map_path = mem_map_arg;
    if (mem_map_path.empty()) {
        mem_map_path = resolve(
            "config/tensix_neo/ttqs_memory_map_" + cfg.llk_tag + ".json",
            base_dir);
    }

    // ── 5. Parse memory map ──────────────────────────────────────────────
    const json mm_json = read_json(mem_map_path);
    parse_memory_map(mm_json, cfg);

    return cfg;
}

} // namespace neosim::config
