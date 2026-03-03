// SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "neosim/isa/instruction.hpp"

#include <array>
#include <cassert>
#include <cstdint>

namespace neosim::units {

/// Replay state machine — replaces Python class `replayState` in t3sim.py.
///
/// Controls whether instructions are passed through as normal, loaded into the
/// 32-entry replay buffer, or executed from the replay buffer.  The mode is
/// driven by REPLAY instructions via update_mode().
///
/// Terminology (matches Python):
///   mode=0  PASSTHROUGH    — normal instruction flow
///   mode=1  LOAD           — load into buffer, skip functional execution
///   mode=2  LOAD_EXECUTE   — load into buffer AND execute functionally
///   mode=3  EXECUTE        — replay from buffer (no new instructions consumed)
class ReplayState {
public:
    /// Replay modes — values match Python rMode integers.
    enum class Mode : int {
        PASSTHROUGH  = 0,
        LOAD         = 1,
        LOAD_EXECUTE = 2,
        EXECUTE      = 3,
    };

    ReplayState() = default;

    // ----------------------------------------------------------------
    // Mode transitions (driven by REPLAY instruction attributes)
    // ----------------------------------------------------------------

    /// Set replay mode from attributes decoded out of a REPLAY instruction.
    ///
    /// @param load               `replay_load_mode` attribute (1 = enter load)
    /// @param exec_while_loading `replay_execute_while_loading` attribute
    /// @param start_idx          `replay_start_idx` attribute
    /// @param len                `replay_len` attribute
    ///
    /// Mode selection (mirrors Python updateRMode):
    ///   load==1 && exec_while_loading==1  →  LOAD_EXECUTE
    ///   load==1 && exec_while_loading!=1  →  LOAD
    ///   load!=1                           →  EXECUTE
    void update_mode(int load, int exec_while_loading, int start_idx, int len);

    // ----------------------------------------------------------------
    // Buffer operations
    // ----------------------------------------------------------------

    /// Store @p ins in the replay buffer (called during LOAD / LOAD_EXECUTE).
    ///
    /// Inserts at slot (start_idx_ + batch_len_) and increments batch_len_.
    /// When batch_len_ reaches len_ (buffer window full), resets to PASSTHROUGH.
    void load_replay_list(isa::InstrPtr ins);

    /// Return the next instruction from the replay buffer (called during EXECUTE).
    ///
    /// Returns the instruction at slot (start_idx_ + batch_len_) and increments
    /// batch_len_.  When batch_len_ reaches len_ (whole window replayed), resets
    /// to PASSTHROUGH.  Returns nullptr if mode != EXECUTE.
    isa::InstrPtr exec_replay_list();

    // ----------------------------------------------------------------
    // Accessors
    // ----------------------------------------------------------------

    Mode mode()      const { return mode_; }
    Mode prev_mode() const { return prev_mode_; }
    bool in_passthrough() const { return mode_ == Mode::PASSTHROUGH; }

    int start_idx() const { return start_idx_; }
    int replay_len() const { return len_; }
    int batch_len() const { return batch_len_; }

private:
    static constexpr int REPLAY_BUF_SIZE = 32;

    Mode mode_      = Mode::PASSTHROUGH;
    Mode prev_mode_ = Mode::PASSTHROUGH;

    int start_idx_ = -1;
    int len_       = -1;
    int batch_len_ = -1;

    std::array<isa::InstrPtr, REPLAY_BUF_SIZE> replay_list_{};
};

} // namespace neosim::units
