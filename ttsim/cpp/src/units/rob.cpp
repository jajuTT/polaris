// SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "neosim/units/rob.hpp"

#include <cassert>

namespace neosim::units {

uint32_t Rob::append(isa::InstrPtr ins)
{
    const uint32_t id = next_id_++;
    id_rob_.push_back(id);
    val_rob_.push_back(std::move(ins));
    return id;
}

bool Rob::pop_head(uint32_t ins_id)
{
    if (id_rob_.empty() || id_rob_.front() != ins_id) {
        return false;
    }
    id_rob_.pop_front();
    val_rob_.pop_front();
    return true;
}

void Rob::remove(uint32_t ins_id)
{
    for (int i = 0; i < static_cast<int>(id_rob_.size()); ++i) {
        if (id_rob_[static_cast<std::size_t>(i)] == ins_id) {
            id_rob_.erase(id_rob_.begin() + static_cast<std::ptrdiff_t>(i));
            val_rob_.erase(val_rob_.begin() + static_cast<std::ptrdiff_t>(i));
            return;
        }
    }
    // Not found — silently ignore (matches Python removeRob behaviour when
    // the instruction has already been removed via another path).
}

bool Rob::is_head(uint32_t ins_id) const
{
    return !id_rob_.empty() && id_rob_.front() == ins_id;
}

isa::InstrPtr Rob::find(uint32_t ins_id) const
{
    for (int i = 0; i < static_cast<int>(id_rob_.size()); ++i) {
        const std::size_t idx = static_cast<std::size_t>(i);
        if (id_rob_[idx] == ins_id) {
            return val_rob_[idx];
        }
    }
    return nullptr;
}

uint32_t Rob::head_id() const
{
    assert(!id_rob_.empty() && "head_id() called on empty ROB");
    return id_rob_.front();
}

} // namespace neosim::units
