// SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "neosim/units/mem_req.hpp"

#include <cassert>
#include <unordered_map>

namespace neosim::units {

/// L1 scratchpad RAM latency model — replaces Python class `scratchpadRam`
/// (scratchpad.py).
///
/// This class is a **pure latency model**: it does not store memory contents.
/// The Python simulator never actually reads or writes data — it only models
/// the delay imposed by L1 accesses.
///
/// ## Latency model (from Python scratchpadRam)
///   - Read:  `latency_rd` cycles (from config key `latency_l1`, typically 10).
///   - Write: 1 cycle (hardcoded in Python as `self.latencyWr = 1`).
///
/// ## Track 4 scope (data model only)
/// The full Sparta DataInPort/DataOutPort wiring and per-pipe SimPy processes
/// that Python uses are introduced in Track 7 (ScratchpadUnit).  This class
/// provides the state and timing logic that Track 7 will wrap.
///
/// ## Usage pattern (used by Track 7 PipeUnit callbacks)
/// ```
/// // On receiving a request from the pipe:
/// int lat = scratchpad.submit(req);   // records in-flight, returns latency
/// response_event_.schedule(lat);      // fires after `lat` simulated cycles
///
/// // In the response event callback:
/// MemReq done_req = scratchpad.complete(req_id);  // removes from tracker
/// // forward done_req back to the requesting pipe
/// ```
class Scratchpad {
public:
    // ----------------------------------------------------------------
    // Configuration
    // ----------------------------------------------------------------

    struct Config {
        int latency_rd   = 10; ///< read latency in cycles  (from config latency_l1)
        int latency_wr   = 1;  ///< write latency in cycles (hardcoded; see §8.3 of plan)
        int num_cores    = 1;
        int num_engines  = 1;  ///< number of pipe engines per core
        bool enable_trace = false;
    };

    explicit Scratchpad(const Config& cfg);

    // ----------------------------------------------------------------
    // Request lifecycle
    // ----------------------------------------------------------------

    /// Submit a request.
    ///
    /// Records the request in the in-flight tracker and returns the number of
    /// cycles the caller must wait before calling complete().
    ///
    /// Mirrors Python scratchpadRam.arbitrate:
    ///   WR → latencyWr cycles, RD → latencyRd cycles.
    ///
    /// @param req  The request to submit (stored by value in the tracker).
    /// @return     Latency in simulated cycles (latency_wr or latency_rd).
    int submit(const MemReq& req);

    /// Mark a request as complete.
    ///
    /// Removes the request from the in-flight tracker and returns it so the
    /// caller can forward the response to the originating pipe.
    ///
    /// Asserts that req_id is currently in-flight.
    MemReq complete(uint32_t req_id);

    // ----------------------------------------------------------------
    // Inspection
    // ----------------------------------------------------------------

    bool is_in_flight(uint32_t req_id) const;

    int  num_in_flight() const { return static_cast<int>(req_trk_.size()); }
    int  num_ports()     const { return num_ports_; }
    int  latency_rd()    const { return cfg_.latency_rd; }
    int  latency_wr()    const { return cfg_.latency_wr; }

    void print_in_flight() const;

private:
    Config cfg_;
    int    num_ports_; ///< num_cores × num_engines (mirrors Python numPorts)

    /// In-flight request tracker: req_id → MemReq copy.
    /// Mirrors Python scratchpadRam.reqTrk dict.
    std::unordered_map<uint32_t, MemReq> req_trk_;
};

} // namespace neosim::units
