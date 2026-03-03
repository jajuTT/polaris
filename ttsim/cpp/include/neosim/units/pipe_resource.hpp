// SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <vector>

namespace neosim::units {

/// Per-pipe resource tracker — replaces Python class `pipeResource` (t3sim.py).
///
/// Tracks an integer state per (pipe, thread) pair.  Typical values:
///   0 = idle, 1 = busy.
///
/// Non-blocking:  set_rsrc_state and check_rsrc_state return immediately.
/// Callers (ThreadUnit / PipeUnit in Track 6/7) are responsible for
/// re-scheduling via Sparta events when a condition is not met.
class PipeResource {
public:
    /// Result returned by check_rsrc_state.
    struct CheckResult {
        bool done;         ///< condition is fully satisfied
        int  consec_count; ///< updated consecutive-cycle counter (pass back on next call)
    };

    /// @param num_pipes      Number of execution pipes.
    /// @param num_threads    Number of Tensix threads.
    /// @param enable_threadwise  When true, set_rsrc_state updates only the
    ///        addressed thread; when false it updates ALL threads for that pipe
    ///        (global resource semantics).  Mirrors Python `enableThreadwisePipeStates`.
    PipeResource(int num_pipes, int num_threads, bool enable_threadwise = true);

    // ----------------------------------------------------------------
    // Non-blocking accessors
    // ----------------------------------------------------------------

    /// Return current state for (pipe_id, thread_id) without side-effects.
    int read_rsrc_state(int pipe_id, int thread_id) const;

    /// Attempt to change pipe state to v.
    ///
    /// Mirrors Python setRsrcState semantics: waits until state != v before
    /// setting it to v.  In C++ this is non-blocking:
    ///   @return true  if state was != v and has now been set to v.
    ///   @return false if state was already v (caller must retry next cycle).
    bool set_rsrc_state(int pipe_id, int thread_id, int v);

    /// Check that pipe state has been v for at least `required` consecutive
    /// cycles.  The caller tracks `prev_consec` across calls.
    ///
    ///   @param prev_consec  Consecutive-cycle count from the previous call.
    ///                       Pass 0 on the first call.
    ///   @param required     Minimum consecutive cycles needed (default 0 =
    ///                       just check state == v this cycle).
    ///   @return {done=true}  when state==v and prev_consec >= required.
    ///   @return {done=false} with updated consec_count to pass on next call.
    CheckResult check_rsrc_state(int pipe_id, int thread_id, int v,
                                 int prev_consec = 0, int required = 0) const;

    // ----------------------------------------------------------------
    // Diagnostic accessors
    // ----------------------------------------------------------------
    int  num_pipes()   const { return num_pipes_; }
    int  num_threads() const { return num_threads_; }
    bool threadwise()  const { return enable_threadwise_; }

    void print_state() const;

private:
    int  num_pipes_;
    int  num_threads_;
    bool enable_threadwise_;

    // Indexed [pipe_id][thread_id]
    std::vector<std::vector<int>> curr_state_;
};

} // namespace neosim::units
