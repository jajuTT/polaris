// SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <cassert>
#include <cstdint>
#include <string>
#include <vector>

namespace neosim::units {

/// Tensix special register file — replaces Python class `ttSplRegs` (tensixFunc.py).
///
/// Models the memory-mapped register banks used by TRISC threads and Tensix
/// execution pipes.  Each register type has a base address and size that maps
/// it into the RISC-V memory address space.
///
/// Register types (in Python terminology):
///   cfg         — configuration registers (read/write from cfg address range)
///   instrBuffer — instruction-buffer register (1 entry)
///   mop         — MOP configuration registers (64 × MAX_THREADS entries)
///   mopSync     — MOP synchronisation registers (1 per thread)
///   idleSync    — idle synchronisation registers (1 per thread)
///   semaphores  — Tensix semaphores (NUM_FIELDS × NUM_SEM_PER_BANK × NUM_BANKS)
///   tileCounters — tile counter structs (one per counter, 8 named sub-fields each)
///
/// `pipeReg` is explicitly NOT an MMR (skipped in is_mmr) and is represented
/// only by its physical dimensions.
///
/// Non-MMR callers access registers via read_reg / write_reg with a SplRegType tag.
/// MMR callers first resolve an address with is_mmr() then use the returned offset.
class TensixSplReg {
public:
    // ----------------------------------------------------------------
    // Constants
    // ----------------------------------------------------------------
    static constexpr int MAX_THREADS           = 4;
    static constexpr int MOP_PER_THREAD        = 64;  ///< mop entries per thread
    static constexpr int NUM_SEM_PER_BANK      = 8;
    static constexpr int NUM_SEM_FIELDS        = 8;   ///< tt_semaphore_idx count
    static constexpr int NUM_SEM_BANKS         = 32;
    static constexpr int SEM_ENTRIES_PER_BANK  = NUM_SEM_PER_BANK * NUM_SEM_FIELDS;
    static constexpr int TILE_CNT_FIELDS       = 8;

    // PCBuffer-relative offsets for mopSync / idleSync / semaphores
    static constexpr uint32_t PCBUF_IDLE_SYNC_OFFSET  = 0x04;
    static constexpr uint32_t PCBUF_MOP_SYNC_OFFSET   = 0x08;
    static constexpr uint32_t PCBUF_SEM_OFFSET        = 0x80;

    // ----------------------------------------------------------------
    // Register type tag (returned by is_mmr)
    // ----------------------------------------------------------------
    enum class SplRegType {
        NONE,
        CFG,
        INSTR_BUF,
        MOP,
        MOP_SYNC,
        IDLE_SYNC,
        SEMAPHORES,
        TILE_COUNTERS,
    };

    /// Result of an is_mmr() lookup.
    struct MmrInfo {
        SplRegType type   = SplRegType::NONE;
        int        offset = -1; ///< word offset from the region base; -1 if NONE
    };

    // ----------------------------------------------------------------
    // Tile counter sub-field indices (matches Python tt_semaphore_idx layout
    // and tileCounters dict order)
    // ----------------------------------------------------------------
    enum class TileCounterField : int {
        RESERVED0              = 0,
        RESET                  = 1,
        TILES_AVAILABLE        = 2,
        SPACE_AVAILABLE        = 3,
        BUFFER_CAPACITY        = 4,
        RESERVED1              = 5,
        TILES_AVAIL_IRQ_THRESH = 6,
        SPACE_AVAIL_IRQ_THRESH = 7,
    };

    // ----------------------------------------------------------------
    // Semaphore sub-field indices (matches Python tt_semaphore_idx)
    // ----------------------------------------------------------------
    enum class SemField : int {
        ID                = 0,
        BANK              = 1,
        INIT_VALUE        = 2,
        MAX_VALUE         = 3,
        CURRENT_VALUE     = 4,
        WAIT_SEM_COND     = 5,
        PIPES_TO_STALL    = 6,
        THREAD_ID_OF_PIPES = 7,
    };

    // ----------------------------------------------------------------
    // Configuration (populated from memory-map JSON at construction)
    // ----------------------------------------------------------------
    struct Config {
        int      core_id = 0;

        // cfg registers
        uint32_t cfg_start         = 0; ///< byte address of first cfg register
        uint32_t cfg_end           = 0; ///< exclusive upper byte address
        uint32_t cfg_bytes_per_reg = 4; ///< bytes per cfg register word

        // instruction buffer (single register)
        uint32_t instr_buf_start         = 0;
        uint32_t instr_buf_bytes_per_reg = 4;

        // mop (MOP_PER_THREAD × MAX_THREADS entries)
        uint32_t mop_start = 0;

        // PCBuffer base (mopSync / idleSync / semaphores are at fixed offsets)
        uint32_t pcbuf_start = 0;

        // tile counters
        uint32_t tile_cnt_start      = 0; ///< byte address of first tile counter
        uint32_t tile_cnt_end        = 0; ///< inclusive byte address of last byte
        uint32_t tile_cnt_entry_bytes = 32; ///< bytes per tile counter entry (8 fields × 4 B)
    };

    explicit TensixSplReg(const Config& cfg);

    // ----------------------------------------------------------------
    // MMR address decoding  (mirrors Python __ismmr__)
    // ----------------------------------------------------------------

    /// Resolve a byte address to a register type and word offset.
    /// Returns {NONE, -1} if the address does not map to any known MMR region.
    /// Note: pipeReg regions are NOT checked (same as Python).
    MmrInfo is_mmr(uint32_t addr) const;

    // ----------------------------------------------------------------
    // Flat register read/write (use after resolving via is_mmr)
    // ----------------------------------------------------------------

    /// Read a scalar register.  For TILE_COUNTERS use the tile_counter_* methods.
    int32_t read_reg(int offset, SplRegType type) const;

    /// Write a scalar register.  For TILE_COUNTERS use the tile_counter_* methods.
    void write_reg(int offset, int32_t value, SplRegType type);

    // ----------------------------------------------------------------
    // Tile counter sub-field accessors
    // ----------------------------------------------------------------

    int32_t read_tile_counter(int idx, TileCounterField field) const;
    void    write_tile_counter(int idx, TileCounterField field, int32_t value);

    // ----------------------------------------------------------------
    // Semaphore flat-array accessors
    // Indexing: flat_idx = bank * SEM_ENTRIES_PER_BANK + sem_idx * NUM_SEM_FIELDS + field
    // ----------------------------------------------------------------

    int32_t read_semaphore(int bank, int sem_idx, SemField field) const;
    void    write_semaphore(int bank, int sem_idx, SemField field, int32_t value);

    // ----------------------------------------------------------------
    // Diagnostic accessors
    // ----------------------------------------------------------------
    int         num_cfg_regs()         const { return static_cast<int>(cfg_.size()); }
    int         num_tile_counters()    const { return static_cast<int>(tile_counters_.size()); }
    const Config& config()             const { return cfg_params_; }

    void print_state(SplRegType type) const;

private:
    Config cfg_params_;

    // cfg[word_offset]
    std::vector<int32_t> cfg_;

    // instrBuffer[0]
    int32_t instr_buf_ = -1;

    // mop[thread * MOP_PER_THREAD + entry]
    std::array<int32_t, MOP_PER_THREAD * MAX_THREADS> mop_{};

    // mop_sync[thread_id]
    std::array<int32_t, MAX_THREADS> mop_sync_{};

    // idle_sync[thread_id]
    std::array<int32_t, MAX_THREADS> idle_sync_{};

    // semaphores[bank * SEM_ENTRIES_PER_BANK + sem_idx * NUM_SEM_FIELDS + field]
    std::vector<int32_t> semaphores_;

    // tile_counters[idx][field]
    std::vector<std::array<int32_t, TILE_CNT_FIELDS>> tile_counters_;

    static constexpr std::array<int32_t, TILE_CNT_FIELDS> TILE_CNT_INIT = {
        -1, -1, 0, 0, -1, -1, -1, -1
    };
};

} // namespace neosim::units
