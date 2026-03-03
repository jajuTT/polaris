// SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "neosim/isa/instruction.hpp"

#include <cassert>
#include <cstdint>
#include <deque>

namespace neosim::units {

/// Reorder buffer — replaces Python class `rob` in t3sim.py.
///
/// Maintains a head-to-tail FIFO of (insId, Instruction) pairs.  Instructions
/// are appended at the tail with append() which returns a monotonically
/// increasing insId.  The "blocking" pop / head-of-ROB operations in the
/// Python simulator translate to non-blocking check methods here; callers
/// re-schedule (via Sparta events) when the condition is not yet satisfied.
///
/// All operations are O(n) in the worst case; typical ROB depths are ≤ 4.
class Rob {
public:
    Rob() = default;

    // ----------------------------------------------------------------
    // Modifiers
    // ----------------------------------------------------------------

    /// Insert instruction at tail; return the assigned insId (monotonically
    /// increasing, starting from 0).
    uint32_t append(isa::InstrPtr ins);

    /// If insId is at the head of the ROB, remove it and return true.
    /// Otherwise return false (caller should re-schedule and retry).
    /// Mirrors Python rob.popRob() without blocking.
    bool pop_head(uint32_t ins_id);

    /// Remove insId from any position in the ROB.  No-op if not found.
    /// Mirrors Python rob.removeRob().
    void remove(uint32_t ins_id);

    // ----------------------------------------------------------------
    // Queries
    // ----------------------------------------------------------------

    /// Return true if insId is the current head of the ROB.
    /// Mirrors Python rob.headOfRob() without blocking.
    bool is_head(uint32_t ins_id) const;

    /// Return the instruction with the given insId, or nullptr if not found.
    /// Mirrors Python rob.findRob().
    isa::InstrPtr find(uint32_t ins_id) const;

    int  size()  const { return static_cast<int>(id_rob_.size()); }
    bool empty() const { return id_rob_.empty(); }

    /// Current head insId (-1 cast to uint32_t when empty).
    uint32_t head_id() const;

    /// The insId that will be assigned to the next append() call.
    uint32_t next_id() const { return next_id_; }

private:
    uint32_t              next_id_ = 0;          ///< ID counter
    std::deque<uint32_t>  id_rob_;               ///< insIds in insertion order
    std::deque<isa::InstrPtr> val_rob_;          ///< corresponding instructions
};

} // namespace neosim::units
