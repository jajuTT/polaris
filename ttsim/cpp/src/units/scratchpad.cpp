// SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "neosim/units/scratchpad.hpp"

#include <cassert>
#include <cstdio>

namespace neosim::units {

// ----------------------------------------------------------------
// MemReq — static counter definition
// ----------------------------------------------------------------

std::atomic<uint32_t> MemReq::counter_{0};

// ----------------------------------------------------------------
// MemReq — diagnostic methods
// ----------------------------------------------------------------

void MemReq::print() const {
    std::printf("MemReq[id=%u op=%s addr=0x%x bytes=%u "
                "src=%s target=%s ins_id=%s thread=%d core=%d]\n",
                req_id_,
                op_ == MemOp::READ ? "RD" : "WR",
                addr_,
                bytes_,
                src_      ? src_->c_str()      : "(unset)",
                target_   ? target_->c_str()   : "(unset)",
                ins_id_   ? std::to_string(*ins_id_).c_str() : "(unset)",
                thread_id_,
                core_id_);
}

void MemReq::trace(int cycle) const {
    if (target_ && *target_ == "L1") {
        std::printf("L1_TRACE: %d,%s,0x%x,%s,0x%x\n",
                    cycle,
                    op_ == MemOp::READ ? "RD" : "WR",
                    addr_,
                    src_ ? src_->c_str() : "(unset)",
                    bytes_);
    }
}

// ----------------------------------------------------------------
// Scratchpad — construction
// ----------------------------------------------------------------

Scratchpad::Scratchpad(const Config& cfg)
    : cfg_(cfg)
    , num_ports_(cfg.num_cores * cfg.num_engines)
{}

// ----------------------------------------------------------------
// Scratchpad — request lifecycle
// ----------------------------------------------------------------

int Scratchpad::submit(const MemReq& req) {
    assert(req_trk_.find(req.req_id()) == req_trk_.end()
           && "req_id already in-flight");
    req_trk_.emplace(req.req_id(), req);
    return (req.op() == MemOp::WRITE) ? cfg_.latency_wr : cfg_.latency_rd;
}

MemReq Scratchpad::complete(uint32_t req_id) {
    auto it = req_trk_.find(req_id);
    assert(it != req_trk_.end() && "req_id not found in in-flight tracker");
    MemReq done = std::move(it->second);
    req_trk_.erase(it);
    return done;
}

bool Scratchpad::is_in_flight(uint32_t req_id) const {
    return req_trk_.count(req_id) > 0;
}

void Scratchpad::print_in_flight() const {
    std::printf("Scratchpad in-flight (%d requests):\n",
                static_cast<int>(req_trk_.size()));
    for (const auto& [id, req] : req_trk_) {
        req.print();
    }
}

} // namespace neosim::units
