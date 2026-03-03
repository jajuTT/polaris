// SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "neosim/units/tensix_spl_reg.hpp"

#include <cassert>
#include <cstdio>

namespace neosim::units {

TensixSplReg::TensixSplReg(const Config& cfg)
    : cfg_params_(cfg)
{
    // cfg registers: size = (cfg_end - cfg_start) / cfg_bytes_per_reg
    assert(cfg.cfg_bytes_per_reg > 0);
    int num_cfg = static_cast<int>((cfg.cfg_end - cfg.cfg_start) / cfg.cfg_bytes_per_reg);
    cfg_.assign(num_cfg, -1);

    // mop: MOP_PER_THREAD × MAX_THREADS entries
    mop_.fill(-1);

    // mop_sync / idle_sync: one entry per thread
    mop_sync_.fill(-1);
    idle_sync_.fill(-1);

    // semaphores: NUM_SEM_BANKS banks, each with SEM_ENTRIES_PER_BANK entries
    semaphores_.assign(NUM_SEM_BANKS * SEM_ENTRIES_PER_BANK, -1);

    // tile_counters: size = (tile_cnt_end + 1 - tile_cnt_start) / tile_cnt_entry_bytes
    assert(cfg.tile_cnt_entry_bytes > 0);
    int num_tc = static_cast<int>(
        (cfg.tile_cnt_end + 1 - cfg.tile_cnt_start) / cfg.tile_cnt_entry_bytes);
    tile_counters_.resize(num_tc, TILE_CNT_INIT);
}

// ----------------------------------------------------------------
// MMR address decoding
// ----------------------------------------------------------------

TensixSplReg::MmrInfo TensixSplReg::is_mmr(uint32_t addr) const {
    const Config& c = cfg_params_;

    // cfg
    if (addr >= c.cfg_start && addr < c.cfg_end) {
        return {SplRegType::CFG, static_cast<int>((addr - c.cfg_start) / 4)};
    }

    // instrBuffer (single 4-byte register)
    if (addr >= c.instr_buf_start &&
        addr < c.instr_buf_start + c.instr_buf_bytes_per_reg) {
        return {SplRegType::INSTR_BUF, 0};
    }

    // mop: addr_size = 64 bytes (matches Python regTypeDict 'mop' addr_size)
    if (addr >= c.mop_start && addr < c.mop_start + 64u) {
        return {SplRegType::MOP, static_cast<int>((addr - c.mop_start) / 4)};
    }

    // idleSync: addr_size = 4 bytes at pcbuf + 0x04
    {
        uint32_t base = c.pcbuf_start + PCBUF_IDLE_SYNC_OFFSET;
        if (addr >= base && addr < base + 4u) {
            return {SplRegType::IDLE_SYNC, static_cast<int>((addr - base) / 4)};
        }
    }

    // mopSync: addr_size = 4 bytes at pcbuf + 0x08
    {
        uint32_t base = c.pcbuf_start + PCBUF_MOP_SYNC_OFFSET;
        if (addr >= base && addr < base + 4u) {
            return {SplRegType::MOP_SYNC, static_cast<int>((addr - base) / 4)};
        }
    }

    // semaphores: addr_size = 64 bytes at pcbuf + 0x80
    {
        uint32_t base = c.pcbuf_start + PCBUF_SEM_OFFSET;
        if (addr >= base && addr < base + 64u) {
            return {SplRegType::SEMAPHORES, static_cast<int>((addr - base) / 4)};
        }
    }

    // tile_counters
    if (addr >= c.tile_cnt_start && addr <= c.tile_cnt_end) {
        return {SplRegType::TILE_COUNTERS, static_cast<int>((addr - c.tile_cnt_start) / 4)};
    }

    return {SplRegType::NONE, -1};
}

// ----------------------------------------------------------------
// Flat register read/write
// ----------------------------------------------------------------

int32_t TensixSplReg::read_reg(int offset, SplRegType type) const {
    switch (type) {
        case SplRegType::CFG:
            assert(offset >= 0 && offset < static_cast<int>(cfg_.size()));
            return cfg_[offset];

        case SplRegType::INSTR_BUF:
            assert(offset == 0);
            return instr_buf_;

        case SplRegType::MOP:
            assert(offset >= 0 && offset < static_cast<int>(mop_.size()));
            return mop_[offset];

        case SplRegType::MOP_SYNC:
            assert(offset >= 0 && offset < MAX_THREADS);
            return mop_sync_[offset];

        case SplRegType::IDLE_SYNC:
            assert(offset >= 0 && offset < MAX_THREADS);
            return idle_sync_[offset];

        case SplRegType::SEMAPHORES:
            assert(offset >= 0 && offset < static_cast<int>(semaphores_.size()));
            return semaphores_[offset];

        default:
            assert(false && "Use tile_counter accessors for TILE_COUNTERS");
            return -1;
    }
}

void TensixSplReg::write_reg(int offset, int32_t value, SplRegType type) {
    switch (type) {
        case SplRegType::CFG:
            assert(offset >= 0 && offset < static_cast<int>(cfg_.size()));
            cfg_[offset] = value;
            break;

        case SplRegType::INSTR_BUF:
            assert(offset == 0);
            instr_buf_ = value;
            break;

        case SplRegType::MOP:
            assert(offset >= 0 && offset < static_cast<int>(mop_.size()));
            mop_[offset] = value;
            break;

        case SplRegType::MOP_SYNC:
            assert(offset >= 0 && offset < MAX_THREADS);
            mop_sync_[offset] = value;
            break;

        case SplRegType::IDLE_SYNC:
            assert(offset >= 0 && offset < MAX_THREADS);
            idle_sync_[offset] = value;
            break;

        case SplRegType::SEMAPHORES:
            assert(offset >= 0 && offset < static_cast<int>(semaphores_.size()));
            semaphores_[offset] = value;
            break;

        default:
            assert(false && "Use tile_counter accessors for TILE_COUNTERS");
    }
}

// ----------------------------------------------------------------
// Tile counter sub-field accessors
// ----------------------------------------------------------------

int32_t TensixSplReg::read_tile_counter(int idx, TileCounterField field) const {
    assert(idx >= 0 && idx < static_cast<int>(tile_counters_.size()));
    return tile_counters_[idx][static_cast<int>(field)];
}

void TensixSplReg::write_tile_counter(int idx, TileCounterField field, int32_t value) {
    assert(idx >= 0 && idx < static_cast<int>(tile_counters_.size()));
    tile_counters_[idx][static_cast<int>(field)] = value;
}

// ----------------------------------------------------------------
// Semaphore flat-array accessors
// ----------------------------------------------------------------

int32_t TensixSplReg::read_semaphore(int bank, int sem_idx, SemField field) const {
    assert(bank >= 0 && bank < NUM_SEM_BANKS);
    assert(sem_idx >= 0 && sem_idx < NUM_SEM_PER_BANK);
    int flat = bank * SEM_ENTRIES_PER_BANK
             + sem_idx * NUM_SEM_FIELDS
             + static_cast<int>(field);
    assert(flat >= 0 && flat < static_cast<int>(semaphores_.size()));
    return semaphores_[flat];
}

void TensixSplReg::write_semaphore(int bank, int sem_idx, SemField field, int32_t value) {
    assert(bank >= 0 && bank < NUM_SEM_BANKS);
    assert(sem_idx >= 0 && sem_idx < NUM_SEM_PER_BANK);
    int flat = bank * SEM_ENTRIES_PER_BANK
             + sem_idx * NUM_SEM_FIELDS
             + static_cast<int>(field);
    assert(flat >= 0 && flat < static_cast<int>(semaphores_.size()));
    semaphores_[flat] = value;
}

// ----------------------------------------------------------------
// Diagnostic
// ----------------------------------------------------------------

void TensixSplReg::print_state(SplRegType type) const {
    switch (type) {
        case SplRegType::CFG:
            std::printf("cfg[%zu entries]:\n", cfg_.size());
            for (int i = 0; i < static_cast<int>(cfg_.size()); ++i) {
                if (cfg_[i] != -1) std::printf("  [%d]=0x%x\n", i, cfg_[i]);
            }
            break;
        case SplRegType::MOP:
            std::printf("mop[%zu entries]:\n", mop_.size());
            for (int i = 0; i < static_cast<int>(mop_.size()); ++i) {
                if (mop_[i] != -1) std::printf("  [%d]=0x%x\n", i, mop_[i]);
            }
            break;
        case SplRegType::TILE_COUNTERS:
            std::printf("tileCounters[%zu entries]:\n", tile_counters_.size());
            for (int i = 0; i < static_cast<int>(tile_counters_.size()); ++i) {
                std::printf("  [%d]: tiles_avail=%d space_avail=%d\n",
                            i,
                            tile_counters_[i][static_cast<int>(TileCounterField::TILES_AVAILABLE)],
                            tile_counters_[i][static_cast<int>(TileCounterField::SPACE_AVAILABLE)]);
            }
            break;
        default:
            std::printf("print_state: type not handled\n");
            break;
    }
}

} // namespace neosim::units
