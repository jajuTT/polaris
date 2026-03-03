// SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "neosim/units/replay.hpp"

#include <cassert>

namespace neosim::units {

void ReplayState::update_mode(int load, int exec_while_loading,
                               int start_idx, int len)
{
    prev_mode_ = mode_;

    if (load == 1) {
        mode_ = (exec_while_loading == 1) ? Mode::LOAD_EXECUTE : Mode::LOAD;
    } else {
        mode_ = Mode::EXECUTE;
    }

    start_idx_ = start_idx;
    len_       = len;
    batch_len_ = -1;  // matches Python: replayBatchLen starts at -1 then loadReplayList pre-increments
}

void ReplayState::load_replay_list(isa::InstrPtr ins)
{
    assert((mode_ == Mode::LOAD || mode_ == Mode::LOAD_EXECUTE)
           && "load_replay_list called when not in LOAD / LOAD_EXECUTE mode");
    assert(start_idx_ >= 0 && "start_idx_ not set; call update_mode first");

    // Python increments batch_len_ BEFORE indexing (so it starts at -1 and
    // immediately becomes 0 on first call, giving slot [start_idx + 0]).
    ++batch_len_;

    const int slot = start_idx_ + batch_len_;
    assert(slot >= 0 && slot < REPLAY_BUF_SIZE && "replay slot out of range");
    replay_list_[static_cast<std::size_t>(slot)] = std::move(ins);

    // When the last slot in the window is loaded, return to passthrough.
    if (batch_len_ == len_ - 1) {
        mode_      = Mode::PASSTHROUGH;
        batch_len_ = -1;
    }
}

isa::InstrPtr ReplayState::exec_replay_list()
{
    if (mode_ != Mode::EXECUTE) return nullptr;

    assert(start_idx_ >= 0 && "start_idx_ not set; call update_mode first");

    // Python increments batch_len_ BEFORE indexing here too.
    ++batch_len_;

    const int slot = start_idx_ + batch_len_;
    assert(slot >= 0 && slot < REPLAY_BUF_SIZE && "replay slot out of range");
    isa::InstrPtr ins = replay_list_[static_cast<std::size_t>(slot)];

    // When the last slot in the window has been executed, reset to passthrough.
    if (batch_len_ == len_ - 1) {
        mode_      = Mode::PASSTHROUGH;
        batch_len_ = -1;
    }

    return ins;
}

} // namespace neosim::units
