// SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "neosim/units/pipe_resource.hpp"

#include <cassert>
#include <cstdio>

namespace neosim::units {

PipeResource::PipeResource(int num_pipes, int num_threads, bool enable_threadwise)
    : num_pipes_(num_pipes)
    , num_threads_(num_threads)
    , enable_threadwise_(enable_threadwise)
    , curr_state_(num_pipes, std::vector<int>(num_threads, 0))
{}

// ----------------------------------------------------------------
// Non-blocking accessors
// ----------------------------------------------------------------

int PipeResource::read_rsrc_state(int pipe_id, int thread_id) const {
    assert(pipe_id >= 0 && pipe_id < num_pipes_);
    assert(thread_id >= 0 && thread_id < num_threads_);
    return curr_state_[pipe_id][thread_id];
}

bool PipeResource::set_rsrc_state(int pipe_id, int thread_id, int v) {
    assert(pipe_id >= 0 && pipe_id < num_pipes_);
    assert(thread_id >= 0 && thread_id < num_threads_);

    // Python semantics: wait until state != v, then set it to v.
    // Non-blocking equivalent: return false if already v (caller must retry).
    if (curr_state_[pipe_id][thread_id] == v) {
        return false;
    }

    if (!enable_threadwise_) {
        // Global: update ALL threads for this pipe.
        for (int th = 0; th < num_threads_; ++th) {
            curr_state_[pipe_id][th] = v;
        }
    } else {
        curr_state_[pipe_id][thread_id] = v;
    }
    return true;
}

PipeResource::CheckResult
PipeResource::check_rsrc_state(int pipe_id, int thread_id, int v,
                                int prev_consec, int required) const {
    assert(pipe_id >= 0 && pipe_id < num_pipes_);
    assert(thread_id >= 0 && thread_id < num_threads_);

    if (curr_state_[pipe_id][thread_id] != v) {
        // State does not match: reset consecutive counter.
        return {false, 0};
    }

    // State matches: check consecutive count.
    // Python: condition met when cyc >= c (with cyc starting at 0).
    // At prev_consec == 0 and required == 0: immediately done.
    if (prev_consec >= required) {
        return {true, prev_consec};
    }
    return {false, prev_consec + 1};
}

// ----------------------------------------------------------------
// Diagnostic
// ----------------------------------------------------------------

void PipeResource::print_state() const {
    std::printf("PipeResource[pipes=%d, threads=%d, threadwise=%d]:\n",
                num_pipes_, num_threads_, enable_threadwise_);
    for (int p = 0; p < num_pipes_; ++p) {
        std::printf("  pipe[%d]: ", p);
        for (int t = 0; t < num_threads_; ++t) {
            std::printf("th[%d]=%d ", t, curr_state_[p][t]);
        }
        std::printf("\n");
    }
}

} // namespace neosim::units
