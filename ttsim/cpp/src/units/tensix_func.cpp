// SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "neosim/units/tensix_func.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace neosim::units {

// ============================================================
// Module-local helpers
// ============================================================

/// Per-bit XOR toggle for a 4-bit value.  Mirrors Python toggle().
[[maybe_unused]]
static int toggle(int val, int mask) {
    assert(val <= 15 && "toggle: val must fit in 4 bits");
    int result = 0;
    for (int i = 0; i < 4; ++i) {
        const int vbit = (val  >> i) & 1;
        const int mbit = (mask >> i) & 1;
        result |= ((mbit ? (1 - vbit) : vbit) << i);
    }
    return result;
}

/// Extract pipe names from a stall-resource bitmask.
/// stallRsrcList order: TDMA, SYNC, PACK, UNPACK, XMOV, THCON, MATH, CFG, SFPU
static std::vector<std::string> pipes_from_stall_bitmask(
    int stall_bits,
    const TensixFunc::PipeGroups& pipe_grps,
    const std::vector<std::string>& pipes)
{
    static const std::string stall_rsrc_list[] = {
        "TDMA", "SYNC", "PACK", "UNPACK", "XMOV", "THCON", "MATH", "CFG", "SFPU"
    };
    std::vector<std::string> dst;
    for (int i = 0; i < 9; ++i) {
        if (!(stall_bits & (1 << i))) continue;
        const auto it = pipe_grps.find(stall_rsrc_list[i]);
        if (it == pipe_grps.end()) continue;
        for (const auto& p : it->second) {
            const auto pit = std::find(pipes.begin(), pipes.end(), p);
            if (pit == pipes.end()) continue;
            if (std::find(dst.begin(), dst.end(), p) == dst.end())
                dst.push_back(p);
        }
    }
    return dst;
}

/// Convert a log2-one-hot sem_sel to semaphore-index-in-bank.
static int sem_sel_to_id(int sem_sel) {
    assert(sem_sel > 0 && (sem_sel & (sem_sel - 1)) == 0
           && "sem_sel must be a positive power-of-2");
    int id = 0;
    int s  = sem_sel;
    while (s > 1) { s >>= 1; ++id; }
    return id;
}

// ============================================================
// Constructor
// ============================================================

TensixFunc::TensixFunc(const Config& cfg, TensixSplReg& spl_regs)
    : cfg_(cfg), spl_regs_(spl_regs)
{}

// ============================================================
// max_autoloop_iterations / mop_cfg_addr
// ============================================================

int TensixFunc::max_autoloop_iterations() const {
    const int max_count = std::max(
        spl_regs_.get_cfg_reg_max_possible_value("THCON_UNPACKER2_REG0_INSTRN_COUNT"),
        spl_regs_.get_cfg_reg_max_possible_value("THCON_PACKER1_REG0_INSTRN_COUNT"));
    const int max_loop = std::max(
        spl_regs_.get_cfg_reg_max_possible_value("THCON_UNPACKER2_REG0_INSTRN_LOOP_COUNT"),
        spl_regs_.get_cfg_reg_max_possible_value("THCON_PACKER1_REG0_INSTRN_LOOP_COUNT"));
    return (max_count + 1) * (max_loop + 1);
}

uint32_t TensixFunc::mop_cfg_addr() const {
    return spl_regs_.config().mop_start;
}

// ============================================================
// Tile-dimension / format helpers
// ============================================================

std::pair<std::string, int>
TensixFunc::get_cb_format(const isa::Instruction& ins) const
{
    assert(cfg_.arch == "ttqs" && "get_cb_format: unsupported arch");
    const auto attrs = ins.get_attr();
    assert(attrs.count("Buffer_Descriptor_Table_Sel")
           && "get_cb_format: missing Buffer_Descriptor_Table_Sel");
    const int sel = attrs.at("Buffer_Descriptor_Table_Sel");
    const std::string reg_name =
        "BUFFER_DESCRIPTOR_TABLE_REG" + std::to_string(sel) + "_TILE_FORMAT";
    return {reg_name, spl_regs_.read_cfg_reg(reg_name)};
}

std::pair<std::string, int>
TensixFunc::get_cb_dim(const isa::Instruction& ins, char dim) const
{
    assert(cfg_.arch == "ttqs" && "get_cb_dim: unsupported arch");
    const auto attrs = ins.get_attr();
    assert(attrs.count("Buffer_Descriptor_Table_Sel")
           && "get_cb_dim: missing Buffer_Descriptor_Table_Sel");
    const int sel = attrs.at("Buffer_Descriptor_Table_Sel");
    const std::string reg_name =
        "BUFFER_DESCRIPTOR_TABLE_REG" + std::to_string(sel) + "_TILE_" + dim + "_DIM";
    return {reg_name, spl_regs_.read_cfg_reg(reg_name)};
}

std::pair<std::string, int>
TensixFunc::get_reg_format([[maybe_unused]] const isa::Instruction& ins,
                           const std::string& ex_pipe) const
{
    assert(cfg_.arch == "ttqs" && "get_reg_format: unsupported arch");
    std::string reg_name;
    const auto unpack_it = cfg_.pipe_grps.find("UNPACK");
    const auto pack_it   = cfg_.pipe_grps.find("PACK");

    bool is_unpack = unpack_it != cfg_.pipe_grps.end() &&
                     std::find(unpack_it->second.begin(), unpack_it->second.end(), ex_pipe)
                         != unpack_it->second.end();
    bool is_pack   = pack_it   != cfg_.pipe_grps.end() &&
                     std::find(pack_it->second.begin(), pack_it->second.end(), ex_pipe)
                         != pack_it->second.end();

    if (is_unpack)
        reg_name = "THCON_" + ex_pipe + "_REG0_OUT_DATA_FORMAT";
    else if (is_pack)
        reg_name = "THCON_" + ex_pipe + "_REG0_IN_DATA_FORMAT";
    else
        assert(false && "get_reg_format: pipe not in UNPACK or PACK group");

    return {reg_name, spl_regs_.read_cfg_reg(reg_name)};
}

int TensixFunc::get_num_datums(const isa::Instruction& ins) const
{
    assert(cfg_.arch == "ttqs" && "get_num_datums: unsupported arch");
    const int z = get_cb_dim(ins, 'Z').second;
    const int y = get_cb_dim(ins, 'Y').second;
    const int x = get_cb_dim(ins, 'X').second;
    return z * y * x;
}

int TensixFunc::compute_num_datums_from_row_cnt_enc(const isa::Instruction& ins) const
{
    if (cfg_.llk_group == 0 || cfg_.llk_group == 1) {
        const auto attrs = ins.get_attr();
        const int enc = attrs.count("Row_Cnt_Enc") ? attrs.at("Row_Cnt_Enc") : 0;
        assert(enc >= 0 && enc <= 5 && "Row_Cnt_Enc out of range [0,5]");
        switch (enc) {
            case 0: return 4 * 16;
            case 1: return 2 * 16;
            case 2: return 1 * 16;
            case 3: return 16 / 2;
            case 4: return 16 / 4;
            case 5: return 16 / 8;
            default: assert(false && "unhandled Row_Cnt_Enc"); return 4 * 16;
        }
    }
    return 4 * 16;
}

// ============================================================
// UNPACR common prefix parser
// ============================================================

void TensixFunc::parse_unpacr_prefix(const isa::Instruction& ins,
                                      std::vector<int>& dst,
                                      std::map<int, int>& vld,
                                      std::map<int, int>& bank,
                                      std::string& ex_pipe) const
{
    using R = isa::RegIndex;
    const std::string& mn = ins.mnemonic.value_or("");
    int reg_id;

    if (mn.find("UNPACR_DEST") != std::string::npos) {
        reg_id  = static_cast<int>(R::DST);
        ex_pipe = "UNPACKER0";
    } else if (mn.find("UNPACR0") != std::string::npos) {
        reg_id  = static_cast<int>(R::SRC_A);
        ex_pipe = "UNPACKER0";
    } else if (mn.find("UNPACR1") != std::string::npos) {
        reg_id  = static_cast<int>(R::SRC_B);
        ex_pipe = "UNPACKER1";
    } else if (mn.find("UNPACR2") != std::string::npos) {
        reg_id  = static_cast<int>(R::SRC_S);
        ex_pipe = "UNPACKER2";
    } else {
        assert(false && "parse_unpacr_prefix: unrecognised UNPACR mnemonic");
        return;
    }

    dst.push_back(reg_id);

    // Check both attribute name spellings used across ISA variants.
    const auto attrs = ins.get_attr();
    int valid = 0;
    if (attrs.count("SetDatValid")) valid = attrs.at("SetDatValid");
    else if (attrs.count("Set_Dvalid")) valid = attrs.at("Set_Dvalid");

    vld[reg_id]  = valid ? 1 : 0;
    bank[reg_id] = valid ? 1 : 0;
}

// ============================================================
// UNPACR handlers
// ============================================================

int TensixFunc::exec_unpacr_ti(isa::Instruction& ins) {
    std::vector<int> src, dst;
    std::map<int, int> vld, bank;
    std::string ex_pipe;

    parse_unpacr_prefix(ins, dst, vld, bank, ex_pipe);

    assert(!ex_pipe.empty());
    const auto [cb_fmt_name, src_fmt] = get_cb_format(ins);
    const auto [reg_fmt_name, dst_fmt] = get_reg_format(ins, ex_pipe);
    const int num_datums = get_num_datums(ins);

    ins.set_src_int(src);
    ins.set_dst_int(dst);
    ins.set_vld_upd_mask(vld);
    ins.set_bank_upd_mask(bank);
    ins.set_ex_pipe(ex_pipe);
    ins.set_num_datums(num_datums);
    ins.set_src_format(src_fmt);
    ins.set_dst_format(dst_fmt);
    return static_cast<int>(ins.get_addr()) + 4;
}

int TensixFunc::exec_unpacr_t(isa::Instruction& ins) {
    return exec_unpacr_ti(ins); // same logic
}

int TensixFunc::exec_unpacr_s(isa::Instruction& ins) {
    std::vector<int> src, dst;
    std::map<int, int> vld, bank;
    std::string ex_pipe;

    parse_unpacr_prefix(ins, dst, vld, bank, ex_pipe);

    assert(!ex_pipe.empty());
    const auto [cb_fmt_name, src_fmt] = get_cb_format(ins);
    const auto [reg_fmt_name, dst_fmt] = get_reg_format(ins, ex_pipe);
    const int num_datums = get_num_datums(ins);

    ins.set_src_int(src);
    ins.set_dst_int(dst);
    ins.set_vld_upd_mask(vld);
    ins.set_bank_upd_mask(bank);
    ins.set_ex_pipe(ex_pipe);
    ins.set_num_datums(num_datums);
    ins.set_src_format(src_fmt);
    ins.set_dst_format(dst_fmt);
    return static_cast<int>(ins.get_addr()) + 4;
}

int TensixFunc::exec_unpacr_tm(isa::Instruction& ins) {
    using R = isa::RegIndex;
    std::vector<int> src, dst;
    std::map<int, int> vld, bank;
    std::string ex_pipe;

    const auto attrs = ins.get_attr();
    const int unpack_type = attrs.count("Unpack_Type") ? attrs.at("Unpack_Type") : -1;
    int valid = attrs.count("SetDatValid") ? attrs.at("SetDatValid") : 0;

    int reg_id;
    switch (unpack_type) {
        case 1: case 3: case 6:
            reg_id  = static_cast<int>(R::SRC_A);
            ex_pipe = "UNPACKER0";
            break;
        case 2: case 4:
            reg_id  = static_cast<int>(R::SRC_B);
            ex_pipe = "UNPACKER1";
            break;
        case 5:
            reg_id  = static_cast<int>(R::SRC_S);
            ex_pipe = "UNPACKER2";
            break;
        case 0:
            // Write to Metadata registers — handled by UNPACKER0 but not
            // modelled as a register write in the performance model.
            ex_pipe = "UNPACKER0";
            ins.set_ex_pipe(ex_pipe);
            return static_cast<int>(ins.get_addr()) + 4;
        default:
            assert(false && "exec_unpacr_tm: unhandled Unpack_Type");
            return static_cast<int>(ins.get_addr()) + 4;
    }

    dst.push_back(reg_id);
    vld[reg_id]  = valid ? 1 : 0;
    bank[reg_id] = valid ? 1 : 0;

    const auto [cb_fmt_name, src_fmt] = get_cb_format(ins);
    const auto [reg_fmt_name, dst_fmt] = get_reg_format(ins, ex_pipe);
    const int num_datums = get_num_datums(ins);

    ins.set_src_int(src);
    ins.set_dst_int(dst);
    ins.set_vld_upd_mask(vld);
    ins.set_bank_upd_mask(bank);
    ins.set_ex_pipe(ex_pipe);
    ins.set_num_datums(num_datums);
    ins.set_src_format(src_fmt);
    ins.set_dst_format(dst_fmt);
    return static_cast<int>(ins.get_addr()) + 4;
}

int TensixFunc::exec_unpacr_tz(isa::Instruction& ins) {
    using R = isa::RegIndex;
    std::vector<int> src, dst;
    std::map<int, int> vld, bank;
    std::string ex_pipe;

    const auto attrs = ins.get_attr();
    const int unpack_sel = attrs.count("Unpack_Sel") ? attrs.at("Unpack_Sel") : 0;
    const int valid      = attrs.count("SetDatValid") ? attrs.at("SetDatValid") : 0;

    int reg_id;
    switch (unpack_sel) {
        case 0:
            reg_id  = static_cast<int>(R::SRC_A);
            ex_pipe = "UNPACKER0";
            break;
        case 1:
            reg_id  = static_cast<int>(R::SRC_B);
            ex_pipe = "UNPACKER1";
            break;
        case 3:
            reg_id  = static_cast<int>(R::DST);
            ex_pipe = "UNPACKER0";
            break;
        default:
            assert(false && "exec_unpacr_tz: unhandled Unpack_Sel");
            return static_cast<int>(ins.get_addr()) + 4;
    }

    dst.push_back(reg_id);
    vld[reg_id]  = valid ? 1 : 0;
    bank[reg_id] = valid ? 1 : 0;

    const auto [cb_fmt_name, src_fmt] = get_cb_format(ins);
    const auto [reg_fmt_name, dst_fmt] = get_reg_format(ins, ex_pipe);
    const int num_datums = compute_num_datums_from_row_cnt_enc(ins);

    ins.set_src_int(src);
    ins.set_dst_int(dst);
    ins.set_vld_upd_mask(vld);
    ins.set_bank_upd_mask(bank);
    ins.set_ex_pipe(ex_pipe);
    ins.set_num_datums(num_datums);
    ins.set_src_format(src_fmt);
    ins.set_dst_format(dst_fmt);
    return static_cast<int>(ins.get_addr()) + 4;
}

int TensixFunc::exec_unpacr_nop(isa::Instruction& ins) {
    using R = isa::RegIndex;
    std::vector<int> src, dst;
    std::map<int, int> vld, bank;
    std::string ex_pipe;

    const auto attrs = ins.get_attr();
    const bool is_ttqs = ins.kind.has_value() &&
                         *ins.kind == ttdecode::isa::instruction_kind::ttqs;

    int reg_id;
    int valid = 0;
    if (is_ttqs) {
        const int sel = attrs.count("Unpacker_Select") ? attrs.at("Unpacker_Select") : 0;
        valid = attrs.count("Set_Dvalid") ? attrs.at("Set_Dvalid") : 0;
        switch (sel) {
            case 0: reg_id = static_cast<int>(R::SRC_A); ex_pipe = "UNPACKER0"; break;
            case 1: reg_id = static_cast<int>(R::SRC_B); ex_pipe = "UNPACKER1"; break;
            default:
                assert(false && "exec_unpacr_nop: unhandled Unpacker_Select");
                return static_cast<int>(ins.get_addr()) + 4;
        }
    } else {
        const int sel = attrs.count("Unpack_block_selection") ?
                        attrs.at("Unpack_block_selection") : 0;
        switch (sel) {
            case 0: reg_id = static_cast<int>(R::SRC_A); ex_pipe = "UNPACKER0"; break;
            case 1: reg_id = static_cast<int>(R::SRC_B); ex_pipe = "UNPACKER1"; break;
            default:
                assert(false && "exec_unpacr_nop: unhandled Unpack_block_selection");
                return static_cast<int>(ins.get_addr()) + 4;
        }
    }

    dst.push_back(reg_id);
    vld[reg_id]  = valid ? 1 : 0;
    bank[reg_id] = valid ? 1 : 0;

    ins.set_src_int(src);
    ins.set_dst_int(dst);
    ins.set_vld_upd_mask(vld);
    ins.set_bank_upd_mask(bank);
    ins.set_ex_pipe(ex_pipe);
    ins.set_num_datums(0);
    ins.set_src_format(0);
    ins.set_dst_format(0);
    return static_cast<int>(ins.get_addr()) + 4;
}

int TensixFunc::exec_unpacr(isa::Instruction& ins) {
    using R = isa::RegIndex;
    // Generic UNPACR (non-ttqs / wh / bh)
    std::vector<int> src, dst;
    std::map<int, int> vld, bank;
    std::string ex_pipe;

    const auto attrs = ins.get_attr();
    const int sel   = attrs.count("Unpack_block_selection") ?
                      attrs.at("Unpack_block_selection") : 0;
    const int valid = attrs.count("SetDatValid") ? attrs.at("SetDatValid") : 0;

    int reg_id;
    switch (sel) {
        case 0: reg_id = static_cast<int>(R::SRC_A); ex_pipe = "UNPACKER0"; break;
        case 1: reg_id = static_cast<int>(R::SRC_B); ex_pipe = "UNPACKER1"; break;
        default:
            assert(false && "exec_unpacr: unhandled Unpack_block_selection");
            return static_cast<int>(ins.get_addr()) + 4;
    }

    dst.push_back(reg_id);
    vld[reg_id]  = valid ? 1 : 0;
    bank[reg_id] = valid ? 1 : 0;

    const auto [cb_fmt_name, src_fmt]  = get_cb_format(ins);
    const auto [reg_fmt_name, dst_fmt] = get_reg_format(ins, ex_pipe);
    constexpr int NUM_DATUMS_TILE = 16 * 16;

    ins.set_src_int(src);
    ins.set_dst_int(dst);
    ins.set_vld_upd_mask(vld);
    ins.set_bank_upd_mask(bank);
    ins.set_ex_pipe(ex_pipe);
    ins.set_num_datums(NUM_DATUMS_TILE);
    ins.set_src_format(src_fmt);
    ins.set_dst_format(dst_fmt);
    return static_cast<int>(ins.get_addr()) + 4;
}

// ============================================================
// PACR handlers
// ============================================================

int TensixFunc::exec_pacr_ti(isa::Instruction& ins) {
    using R = isa::RegIndex;
    std::vector<int> src, dst;
    std::map<int, int> vld, bank;
    std::string ex_pipe;

    const std::string& mn = ins.mnemonic.value_or("");
    int reg_id;
    if (mn.find("PACR0") != std::string::npos) {
        reg_id  = static_cast<int>(R::DST);
        ex_pipe = "PACKER0";
    } else if (mn.find("PACR1") != std::string::npos) {
        reg_id  = static_cast<int>(R::SRC_S);
        ex_pipe = "PACKER1";
    } else {
        assert(false && "exec_pacr_ti: unhandled PACR mnemonic prefix");
        return static_cast<int>(ins.get_addr()) + 4;
    }

    src.push_back(reg_id);
    const auto attrs = ins.get_attr();
    const int clr = attrs.count("ClrDatValid") ? attrs.at("ClrDatValid") : 0;
    vld[reg_id]  = clr ? 1 : 0;
    bank[reg_id] = clr ? 1 : 0;

    // For packer: CB format → dst_format, register format → src_format
    const auto [cb_fmt_name, dst_fmt]  = get_cb_format(ins);
    const auto [reg_fmt_name, src_fmt] = get_reg_format(ins, ex_pipe);
    const int num_datums = get_num_datums(ins);

    ins.set_src_int(src);
    ins.set_dst_int(dst);
    ins.set_vld_upd_mask(vld);
    ins.set_bank_upd_mask(bank);
    ins.set_ex_pipe(ex_pipe);
    ins.set_num_datums(num_datums);
    ins.set_src_format(src_fmt);
    ins.set_dst_format(dst_fmt);
    return static_cast<int>(ins.get_addr()) + 4;
}

int TensixFunc::exec_pacr_stride(isa::Instruction& ins) {
    using R = isa::RegIndex;
    std::vector<int> src, dst;
    std::map<int, int> vld, bank;
    std::string ex_pipe;

    const auto attrs = ins.get_attr();
    const int packer_sel = attrs.count("PackerSel") ? attrs.at("PackerSel") : 0;
    const int clr        = attrs.count("ClrDatValid") ? attrs.at("ClrDatValid") : 0;

    int reg_id;
    switch (packer_sel) {
        case 0: reg_id = static_cast<int>(R::DST);   ex_pipe = "PACKER0"; break;
        case 1: reg_id = static_cast<int>(R::SRC_S); ex_pipe = "PACKER1"; break;
        default:
            assert(false && "exec_pacr_stride: unhandled PackerSel");
            return static_cast<int>(ins.get_addr()) + 4;
    }

    src.push_back(reg_id);
    vld[reg_id]  = clr ? 1 : 0;
    bank[reg_id] = clr ? 1 : 0;

    const auto [cb_fmt_name, dst_fmt]  = get_cb_format(ins);
    const auto [reg_fmt_name, src_fmt] = get_reg_format(ins, ex_pipe);
    const int num_datums = get_num_datums(ins);

    ins.set_src_int(src);
    ins.set_dst_int(dst);
    ins.set_vld_upd_mask(vld);
    ins.set_bank_upd_mask(bank);
    ins.set_ex_pipe(ex_pipe);
    ins.set_num_datums(num_datums);
    ins.set_src_format(src_fmt);
    ins.set_dst_format(dst_fmt);
    return static_cast<int>(ins.get_addr()) + 4;
}

int TensixFunc::exec_pacr_untilize(isa::Instruction& ins) {
    using R = isa::RegIndex;
    std::vector<int> src, dst;
    std::map<int, int> vld, bank;
    std::string ex_pipe;

    const auto attrs   = ins.get_attr();
    const int packer_sel = attrs.count("Packer_Sel") ? attrs.at("Packer_Sel") : 0;
    const int clr        = attrs.count("ClrDatValid") ? attrs.at("ClrDatValid") : 0;

    assert(packer_sel == 0 && "exec_pacr_untilize: Packer_Sel==1 is reserved");
    const int reg_id = static_cast<int>(R::DST);
    ex_pipe = "PACKER0";

    src.push_back(reg_id);
    vld[reg_id]  = clr ? 1 : 0;
    bank[reg_id] = clr ? 1 : 0;

    const auto [cb_fmt_name, dst_fmt]  = get_cb_format(ins);
    const auto [reg_fmt_name, src_fmt] = get_reg_format(ins, ex_pipe);
    const int num_datums = compute_num_datums_from_row_cnt_enc(ins);

    ins.set_src_int(src);
    ins.set_dst_int(dst);
    ins.set_vld_upd_mask(vld);
    ins.set_bank_upd_mask(bank);
    ins.set_ex_pipe(ex_pipe);
    ins.set_num_datums(num_datums);
    ins.set_src_format(src_fmt);
    ins.set_dst_format(dst_fmt);
    return static_cast<int>(ins.get_addr()) + 4;
}

// ============================================================
// Math / pool handlers
// ============================================================

// Helper used by ELWADD/ELWSUB/ELWMUL/MVMUL/MVMULDI/GAPOOL/GMPOOL.
static int exec_elw_like(isa::Instruction& ins, int clear_dvalid) {
    using R = isa::RegIndex;
    std::vector<int> src, dst;
    std::map<int, int> vld, bank;

    dst.push_back(static_cast<int>(R::DST));   vld[3] = 0; bank[3] = 0;
    src.push_back(static_cast<int>(R::SRC_A)); vld[0] = 0; bank[0] = 0;
    src.push_back(static_cast<int>(R::SRC_B)); vld[1] = 0; bank[1] = 0;

    switch (clear_dvalid) {
        case 1: vld[0] = 1; bank[0] = 1; break;
        case 2: vld[1] = 1; bank[1] = 1; break;
        case 3: vld[0] = 1; bank[0] = 1; vld[1] = 1; bank[1] = 1; break;
        default: break;
    }

    ins.set_src_int(src);
    ins.set_dst_int(dst);
    ins.set_vld_upd_mask(vld);
    ins.set_bank_upd_mask(bank);
    return static_cast<int>(ins.get_addr()) + 4;
}

int TensixFunc::exec_gpool(isa::Instruction& ins) {
    const auto attrs = ins.get_attr();
    return exec_elw_like(ins, attrs.count("clear_dvalid") ? attrs.at("clear_dvalid") : 0);
}

int TensixFunc::exec_elwadd(isa::Instruction& ins) {
    const auto attrs = ins.get_attr();
    return exec_elw_like(ins, attrs.count("clear_dvalid") ? attrs.at("clear_dvalid") : 0);
}

int TensixFunc::exec_elwsub(isa::Instruction& ins) {
    const auto attrs = ins.get_attr();
    return exec_elw_like(ins, attrs.count("clear_dvalid") ? attrs.at("clear_dvalid") : 0);
}

int TensixFunc::exec_elwmul(isa::Instruction& ins) {
    const auto attrs = ins.get_attr();
    return exec_elw_like(ins, attrs.count("clear_dvalid") ? attrs.at("clear_dvalid") : 0);
}

int TensixFunc::exec_mvmul(isa::Instruction& ins) {
    const auto attrs = ins.get_attr();
    return exec_elw_like(ins, attrs.count("clear_dvalid") ? attrs.at("clear_dvalid") : 0);
}

int TensixFunc::exec_mvmuldi(isa::Instruction& ins) {
    const auto attrs = ins.get_attr();
    return exec_elw_like(ins, attrs.count("clear_dvalid") ? attrs.at("clear_dvalid") : 0);
}

// ============================================================
// Trivial no-op handlers
// ============================================================

int TensixFunc::exec_atgetm(isa::Instruction& ins)  { return static_cast<int>(ins.get_addr()) + 4; }
int TensixFunc::exec_atrelm(isa::Instruction& ins)  { return static_cast<int>(ins.get_addr()) + 4; }
int TensixFunc::exec_dmanop(isa::Instruction& ins)  { return static_cast<int>(ins.get_addr()) + 4; }
int TensixFunc::exec_nop(isa::Instruction& ins)     { return static_cast<int>(ins.get_addr()) + 4; }
int TensixFunc::exec_setadcxx(isa::Instruction& ins){ return static_cast<int>(ins.get_addr()) + 4; }
int TensixFunc::exec_setadcxy(isa::Instruction& ins){ return static_cast<int>(ins.get_addr()) + 4; }
int TensixFunc::exec_setadczw(isa::Instruction& ins){ return static_cast<int>(ins.get_addr()) + 4; }
int TensixFunc::exec_wrcfg(isa::Instruction& ins)  { return static_cast<int>(ins.get_addr()) + 4; }
int TensixFunc::exec_zerosrc(isa::Instruction& ins) { return static_cast<int>(ins.get_addr()) + 4; }
int TensixFunc::exec_zeroacc(isa::Instruction& ins) { return static_cast<int>(ins.get_addr()) + 4; }
int TensixFunc::exec_sfpconfig(isa::Instruction& ins){ return static_cast<int>(ins.get_addr()) + 4; }
int TensixFunc::exec_sfploadi(isa::Instruction& ins){ return static_cast<int>(ins.get_addr()) + 4; }
int TensixFunc::exec_dst_tile_face_row_idx(isa::Instruction& ins){ return static_cast<int>(ins.get_addr()) + 4; }
int TensixFunc::exec_src_tile_face_row_idx(isa::Instruction& ins){ return static_cast<int>(ins.get_addr()) + 4; }

// ============================================================
// Semaphore handlers
// ============================================================

// Common semaphore operand extraction.
static void sem_extract(const isa::Instruction& ins,
                         int& sem_id_in_bank, int& bank_id)
{
    assert(ins.operands.has_value() && "sem_extract: missing operands");
    const auto& all = ins.operands->all;
    assert(all.count("sem_sel") && "sem_extract: missing sem_sel");
    const int sem_sel = all.at("sem_sel");
    sem_id_in_bank = sem_sel_to_id(sem_sel);

    const bool is_ttqs = ins.kind.has_value() &&
                         *ins.kind == ttdecode::isa::instruction_kind::ttqs;
    bank_id = (is_ttqs && all.count("sem_bank_sel")) ? all.at("sem_bank_sel") : 0;
}

int TensixFunc::exec_seminit(isa::Instruction& ins) {
    int sem_id_in_bank, bank_id;
    sem_extract(ins, sem_id_in_bank, bank_id);

    const auto& all = ins.operands->all;
    const int32_t init_val = all.count("init_value") ? all.at("init_value") : 0;
    const int32_t max_val  = all.count("max_value")  ? all.at("max_value")  : 0;

    using SF = TensixSplReg::SemField;

    // Check for pipes_to_stall (reinit case)
    const int32_t stalled_bits =
        spl_regs_.read_semaphore(bank_id, sem_id_in_bank, SF::PIPES_TO_STALL);
    const int32_t thread_id_of_pipes =
        spl_regs_.read_semaphore(bank_id, sem_id_in_bank, SF::THREAD_ID_OF_PIPES);

    if (stalled_bits > 0) {
        // Reinit: unstall any previously stalled pipes.
        std::vector<std::string> pipe_names;
        for (int i = 0; i < 32 && i < static_cast<int>(cfg_.pipes.size()); ++i) {
            if ((stalled_bits >> i) & 1) pipe_names.push_back(cfg_.pipes[i]);
        }
        if (!pipe_names.empty()) {
            if (thread_id_of_pipes >= 0) {
                ins.set_pipes_thread_id(static_cast<uint32_t>(thread_id_of_pipes));
            }
            ins.set_dst_pipes(pipe_names);
        }
    }

    // Write semaphore fields
    spl_regs_.write_semaphore(bank_id, sem_id_in_bank, SF::ID,            sem_id_in_bank);
    spl_regs_.write_semaphore(bank_id, sem_id_in_bank, SF::BANK,          bank_id);
    spl_regs_.write_semaphore(bank_id, sem_id_in_bank, SF::INIT_VALUE,    init_val);
    spl_regs_.write_semaphore(bank_id, sem_id_in_bank, SF::MAX_VALUE,     max_val);
    spl_regs_.write_semaphore(bank_id, sem_id_in_bank, SF::CURRENT_VALUE, init_val);
    spl_regs_.write_semaphore(bank_id, sem_id_in_bank, SF::PIPES_TO_STALL,    0);
    spl_regs_.write_semaphore(bank_id, sem_id_in_bank, SF::THREAD_ID_OF_PIPES, -1);

    return static_cast<int>(ins.get_addr()) + 4;
}

int TensixFunc::exec_semget(isa::Instruction& ins) {
    int sem_id_in_bank, bank_id;
    sem_extract(ins, sem_id_in_bank, bank_id);

    using SF = TensixSplReg::SemField;
    int32_t cur = spl_regs_.read_semaphore(bank_id, sem_id_in_bank, SF::CURRENT_VALUE);
    spl_regs_.write_semaphore(bank_id, sem_id_in_bank, SF::CURRENT_VALUE, cur - 1);

    const int32_t stalled_bits =
        spl_regs_.read_semaphore(bank_id, sem_id_in_bank, SF::PIPES_TO_STALL);
    if (stalled_bits > 0) {
        std::vector<std::string> pipe_names;
        for (int i = 0; i < 32 && i < static_cast<int>(cfg_.pipes.size()); ++i)
            if ((stalled_bits >> i) & 1) pipe_names.push_back(cfg_.pipes[i]);

        const int32_t tid =
            spl_regs_.read_semaphore(bank_id, sem_id_in_bank, SF::THREAD_ID_OF_PIPES);
        if (tid >= 0)
            ins.set_pipes_thread_id(static_cast<uint32_t>(tid));
        ins.set_dst_pipes(pipe_names);

        spl_regs_.write_semaphore(bank_id, sem_id_in_bank, SF::PIPES_TO_STALL,    0);
        spl_regs_.write_semaphore(bank_id, sem_id_in_bank, SF::THREAD_ID_OF_PIPES,-1);
    }

    return static_cast<int>(ins.get_addr()) + 4;
}

int TensixFunc::exec_sempost(isa::Instruction& ins) {
    int sem_id_in_bank, bank_id;
    sem_extract(ins, sem_id_in_bank, bank_id);

    using SF = TensixSplReg::SemField;
    int32_t cur = spl_regs_.read_semaphore(bank_id, sem_id_in_bank, SF::CURRENT_VALUE);
    spl_regs_.write_semaphore(bank_id, sem_id_in_bank, SF::CURRENT_VALUE, cur + 1);

    const int32_t stalled_bits =
        spl_regs_.read_semaphore(bank_id, sem_id_in_bank, SF::PIPES_TO_STALL);
    if (stalled_bits > 0) {
        std::vector<std::string> pipe_names;
        for (int i = 0; i < 32 && i < static_cast<int>(cfg_.pipes.size()); ++i)
            if ((stalled_bits >> i) & 1) pipe_names.push_back(cfg_.pipes[i]);

        const int32_t tid =
            spl_regs_.read_semaphore(bank_id, sem_id_in_bank, SF::THREAD_ID_OF_PIPES);
        if (tid >= 0)
            ins.set_pipes_thread_id(static_cast<uint32_t>(tid));
        ins.set_dst_pipes(pipe_names);

        spl_regs_.write_semaphore(bank_id, sem_id_in_bank, SF::PIPES_TO_STALL,    0);
        spl_regs_.write_semaphore(bank_id, sem_id_in_bank, SF::THREAD_ID_OF_PIPES,-1);
    }

    return static_cast<int>(ins.get_addr()) + 4;
}

int TensixFunc::exec_semwait(isa::Instruction& ins) {
    int sem_id_in_bank, bank_id;
    sem_extract(ins, sem_id_in_bank, bank_id);

    using SF = TensixSplReg::SemField;

    // Auto-initialise uninitialised semaphore (mirrors Python behaviour).
    int32_t sem_id_stored =
        spl_regs_.read_semaphore(bank_id, sem_id_in_bank, SF::ID);
    if (sem_id_stored == -1) {
        spl_regs_.write_semaphore(bank_id, sem_id_in_bank, SF::ID,            sem_id_in_bank);
        spl_regs_.write_semaphore(bank_id, sem_id_in_bank, SF::BANK,          bank_id);
        spl_regs_.write_semaphore(bank_id, sem_id_in_bank, SF::INIT_VALUE,    0);
        spl_regs_.write_semaphore(bank_id, sem_id_in_bank, SF::MAX_VALUE,     0);
        spl_regs_.write_semaphore(bank_id, sem_id_in_bank, SF::CURRENT_VALUE, 0);
        spl_regs_.write_semaphore(bank_id, sem_id_in_bank, SF::PIPES_TO_STALL, 0);
    }

    const int32_t max_val = spl_regs_.read_semaphore(bank_id, sem_id_in_bank, SF::MAX_VALUE);
    const int32_t cur_val = spl_regs_.read_semaphore(bank_id, sem_id_in_bank, SF::CURRENT_VALUE);

    // Determine wait condition from operands.
    const auto& all = ins.operands->all;
    const int wait_cond = all.count("wait_sem_cond") ? all.at("wait_sem_cond") : 0;
    spl_regs_.write_semaphore(bank_id, sem_id_in_bank, SF::WAIT_SEM_COND, wait_cond);

    // Stall condition: wait_cond==1 && cur==0, OR wait_cond==2 && cur==max
    const bool stall_pipes =
        ((wait_cond == 1) && (cur_val == 0)) ||
        ((wait_cond == 2) && (cur_val == max_val));

    // stall_res bitmask from operands (prefer 'all', fall back to attrs)
    int stall_bits = 0;
    if (all.count("stall_res")) stall_bits = all.at("stall_res");
    else {
        const auto attrs = ins.get_attr();
        if (attrs.count("stall_res")) stall_bits = attrs.at("stall_res");
    }

    if (stall_pipes && stall_bits) {
        const auto dst_pipes =
            pipes_from_stall_bitmask(stall_bits, cfg_.pipe_grps, cfg_.pipes);
        if (!dst_pipes.empty()) {
            ins.set_dst_pipes(dst_pipes);

            int32_t pipes_bits =
                spl_regs_.read_semaphore(bank_id, sem_id_in_bank, SF::PIPES_TO_STALL);
            if (pipes_bits < 0) pipes_bits = 0;
            for (const auto& p : dst_pipes) {
                auto pit = std::find(cfg_.pipes.begin(), cfg_.pipes.end(), p);
                if (pit != cfg_.pipes.end())
                    pipes_bits |= (1 << static_cast<int>(
                        std::distance(cfg_.pipes.begin(), pit)));
            }
            spl_regs_.write_semaphore(bank_id, sem_id_in_bank, SF::PIPES_TO_STALL, pipes_bits);
            spl_regs_.write_semaphore(bank_id, sem_id_in_bank, SF::THREAD_ID_OF_PIPES,
                                      static_cast<int32_t>(ins.get_thread_id()));
        }
    } else {
        spl_regs_.write_semaphore(bank_id, sem_id_in_bank, SF::PIPES_TO_STALL,    0);
        spl_regs_.write_semaphore(bank_id, sem_id_in_bank, SF::THREAD_ID_OF_PIPES,-1);
    }

    return static_cast<int>(ins.get_addr()) + 4;
}

// ============================================================
// ADC / cfg / RWC handlers
// ============================================================

void TensixFunc::apply_cfg_reg_write(uint32_t addr, int32_t val, isa::Instruction& ins)
{
    using RC = TensixSplReg::CfgRegUpdateClass;
    using R  = isa::RegIndex;

    spl_regs_.write_reg(static_cast<int>(addr), val, TensixSplReg::SplRegType::CFG);

    const auto update_class = spl_regs_.get_cfg_reg_update_class(static_cast<int>(addr));

    if (update_class == RC::DEST_TARGET_REG_CFG_MATH) {
        if (spl_regs_.is_dst_reg_programmed() &&
            spl_regs_.update_dst_reg_bank_id(static_cast<int>(addr))) {
            ins.set_bank_upd_mask({{static_cast<int>(R::DST), 1}});
        }
    } else if (update_class == RC::DEST_DVALID_CTRL) {
        const auto cvi = spl_regs_.get_dst_reg_cond_valids(static_cast<int>(addr));
        const int dst_reg = static_cast<int>(R::DST);

        // Populate cond_chk and cond_wri maps for the DST register at context_id.
        // The maps are indexed by [reg_id][thread/context_id].
        const int ctx = cvi.context_id;
        // We extend existing maps (or create new ones) for the DST register.
        std::map<int, std::map<int, int>> cond_chk;
        std::map<int, std::map<int, int>> cond_wri;
        for (int r = 0; r < static_cast<int>(isa::RegIndex::COUNT); ++r) {
            for (int t = 0; t < TensixSplReg::MAX_THREADS; ++t) {
                cond_chk[r][t] = static_cast<int>(isa::ValueStatus::IGNORE);
                cond_wri[r][t] = static_cast<int>(isa::ValueStatus::IGNORE);
            }
        }
        cond_chk[dst_reg][ctx] = cvi.read_mask;
        cond_wri[dst_reg][ctx] = cvi.write_mask;
        ins.set_cond_chk_vld_upd(cond_chk);
        ins.set_cond_wri_vld_upd(cond_wri);
    }
    // BUFFER_DESCRIPTOR_TABLE_REG and UNKNOWN: no instruction-state side-effects
}

int TensixFunc::exec_setc16(isa::Instruction& ins) {
    const auto attrs = ins.get_attr();
    const uint32_t addr = static_cast<uint32_t>(
        attrs.count("setc16_reg") ? attrs.at("setc16_reg") : 0);
    const int32_t val = attrs.count("setc16_value") ? attrs.at("setc16_value") : 0;
    apply_cfg_reg_write(addr, val, ins);
    return static_cast<int>(ins.get_addr()) + 4;
}

int TensixFunc::exec_setrwc(isa::Instruction& ins) {
    using R = isa::RegIndex;
    std::vector<int> src, dst;
    std::map<int, int> vld, bank;

    const auto attrs = ins.get_attr();
    const int clear_ab_vld = attrs.count("clear_ab_vld") ? attrs.at("clear_ab_vld") : 0;

    if (clear_ab_vld & 0x1) {
        src.push_back(static_cast<int>(R::SRC_A));
        vld[0] = 1; bank[0] = 1;
    }
    if (clear_ab_vld & 0x2) {
        src.push_back(static_cast<int>(R::SRC_B));
        vld[1] = 1; bank[1] = 1;
    }

    ins.set_src_int(src);
    ins.set_dst_int(dst);
    ins.set_vld_upd_mask(vld);
    ins.set_bank_upd_mask(bank);
    return static_cast<int>(ins.get_addr()) + 4;
}

int TensixFunc::exec_stallwait(isa::Instruction& ins) {
    // stall_res bitmask in operands.all['stall_res']
    int stall_bits = 0;
    if (ins.operands.has_value() && ins.operands->all.count("stall_res"))
        stall_bits = ins.operands->all.at("stall_res");

    const auto dst_pipes =
        pipes_from_stall_bitmask(stall_bits, cfg_.pipe_grps, cfg_.pipes);

    // wait_res_idx_0/1/2 → src_pipes (llk_group 0 only)
    std::vector<std::string> src_pipes;
    if (cfg_.llk_group == 0) {
        static const std::string wait_attrs[] = {
            "wait_res_idx_0", "wait_res_idx_1", "wait_res_idx_2"
        };
        const auto gattrs = ins.get_attr();
        for (const auto& wkey : wait_attrs) {
            if (!gattrs.count(wkey)) continue;
            const int code = gattrs.at(wkey);
            std::string grp;
            switch (code) {
                case 0x01: grp = "THCON"; break;
                case 0x02: case 0x03: case 0x04: case 0x05: case 0x06: case 0x07:
                    grp = "UNPACK"; break;
                case 0x08: case 0x09: case 0x0A: case 0x0B:
                    grp = "PACK"; break;
                case 0x0C: grp = "MATH";  break;
                case 0x11: grp = "XMOV";  break;
                case 0x13: grp = "SFPU";  break;
                case 0x14: grp = "CFG";   break;
                default: break;
            }
            if (grp.empty()) continue;
            const auto it = cfg_.pipe_grps.find(grp);
            if (it == cfg_.pipe_grps.end()) continue;
            for (const auto& p : it->second) {
                const auto pit = std::find(cfg_.pipes.begin(), cfg_.pipes.end(), p);
                if (pit == cfg_.pipes.end()) continue;
                if (std::find(src_pipes.begin(), src_pipes.end(), p) == src_pipes.end())
                    src_pipes.push_back(p);
            }
        }
    }

    ins.set_dst_pipes(dst_pipes);
    ins.set_src_pipes(src_pipes);
    return static_cast<int>(ins.get_addr()) + 4;
}

int TensixFunc::exec_rmwcib(isa::Instruction& ins) {
    const auto& op    = ins.get_op();
    const auto  attrs = ins.get_attr();
    const uint32_t addr = static_cast<uint32_t>(
        attrs.count("CfgRegAddr") ? attrs.at("CfgRegAddr") : 0);
    const int32_t data = attrs.count("Data") ? attrs.at("Data") : 0;
    const int32_t mask = attrs.count("Mask") ? attrs.at("Mask") : 0;
    const int32_t masked_data = data & mask;

    int32_t old_val = spl_regs_.read_reg(static_cast<int>(addr),
                                         TensixSplReg::SplRegType::CFG);
    if (old_val == -1) old_val = 0;

    int32_t new_val;
    if      (op == "RMWCIB0") new_val = old_val | (masked_data << 0);
    else if (op == "RMWCIB1") new_val = old_val | (masked_data << 8);
    else if (op == "RMWCIB2") new_val = old_val | (masked_data << 16);
    else                      new_val = old_val | (masked_data << 24); // RMWCIB3

    apply_cfg_reg_write(addr, new_val, ins);
    return static_cast<int>(ins.get_addr()) + 4;
}

// ============================================================
// DVALID / MOV handlers
// ============================================================

int TensixFunc::exec_clrdvalid(isa::Instruction& ins) {
    using R = isa::RegIndex;
    std::vector<int> src, dst;
    std::map<int, int> vld, bank;

    const auto attrs = ins.get_attr();
    const int cleardvalid   = attrs.count("cleardvalid")    ? attrs.at("cleardvalid")    : 0;
    const int cleardvalid_s = attrs.count("cleardvalid_S")  ? attrs.at("cleardvalid_S")  : 0;
    const int dest_pulse    = attrs.count("dest_pulse_last")? attrs.at("dest_pulse_last"): 0;

    switch (cleardvalid) {
        case 1:
            src.push_back(static_cast<int>(R::SRC_A));
            vld[0] = 1; bank[0] = 1;
            break;
        case 2:
            src.push_back(static_cast<int>(R::SRC_B));
            vld[1] = 1; bank[1] = 1;
            break;
        case 3:
            src.push_back(static_cast<int>(R::SRC_A));
            src.push_back(static_cast<int>(R::SRC_B));
            vld[0] = 1; bank[0] = 1;
            vld[1] = 1; bank[1] = 1;
            break;
        default: break;
    }

    if (cleardvalid_s) {
        src.push_back(static_cast<int>(R::SRC_S));
        vld[2] = 1; bank[2] = 1;
    }

    // dest_pulse_last: 1/2/4 → dst sets valid; 8 → src clears valid
    switch (dest_pulse) {
        case 1: case 2: case 4:
            dst.push_back(static_cast<int>(R::DST));
            vld[3] = 1; bank[3] = 1;
            break;
        case 8:
            src.push_back(static_cast<int>(R::DST));
            vld[3] = 1; bank[3] = 1;
            break;
        default: break;
    }

    ins.set_src_int(src);
    ins.set_dst_int(dst);
    ins.set_vld_upd_mask(vld);
    ins.set_bank_upd_mask(bank);
    return static_cast<int>(ins.get_addr()) + 4;
}

int TensixFunc::exec_mov(isa::Instruction& ins) {
    using R = isa::RegIndex;
    std::vector<int> src, dst;
    std::map<int, int> vld, bank;
    const std::string& op = ins.get_op();

    if      (op == "MOVA2D") {
        dst.push_back(static_cast<int>(R::DST));   vld[3]=0; bank[3]=0;
        src.push_back(static_cast<int>(R::SRC_A)); vld[0]=0; bank[0]=0;
    } else if (op == "MOVB2D") {
        dst.push_back(static_cast<int>(R::DST));   vld[3]=0; bank[3]=0;
        src.push_back(static_cast<int>(R::SRC_B)); vld[1]=0; bank[1]=0;
    } else if (op == "MOVD2A") {
        dst.push_back(static_cast<int>(R::SRC_A)); vld[0]=0; bank[0]=0;
        src.push_back(static_cast<int>(R::DST));   vld[3]=0; bank[3]=0;
    } else if (op == "MOVD2B") {
        dst.push_back(static_cast<int>(R::SRC_B)); vld[1]=0; bank[1]=0;
        src.push_back(static_cast<int>(R::DST));   vld[3]=0; bank[3]=0;
    } else if (op == "MOVB2A") {
        dst.push_back(static_cast<int>(R::SRC_A)); vld[0]=0; bank[0]=0;
        src.push_back(static_cast<int>(R::SRC_B)); vld[1]=0; bank[1]=0;
    } else {
        assert(false && "exec_mov: unrecognised MOV opcode");
    }

    ins.set_src_int(src);
    ins.set_dst_int(dst);
    ins.set_vld_upd_mask(vld);
    ins.set_bank_upd_mask(bank);
    return static_cast<int>(ins.get_addr()) + 4;
}

int TensixFunc::exec_trnsp(isa::Instruction& ins) {
    using R = isa::RegIndex;
    std::vector<int> src;
    std::map<int, int> vld, bank;
    const std::string& op = ins.get_op();

    if (op == "TRNSPSRCA") {
        src.push_back(static_cast<int>(R::SRC_A)); vld[0]=0; bank[0]=0;
    } else {
        src.push_back(static_cast<int>(R::SRC_B)); vld[1]=0; bank[1]=0;
    }

    ins.set_src_int(src);
    ins.set_vld_upd_mask(vld);
    ins.set_bank_upd_mask(bank);
    return static_cast<int>(ins.get_addr()) + 4;
}

// ============================================================
// SFPU handlers
// ============================================================

int TensixFunc::exec_sfpload(isa::Instruction& ins) {
    if (!ins.operands.has_value()) return static_cast<int>(ins.get_addr()) + 4;
    const auto& attrs_map = ins.operands->attributes;

    std::vector<int> src;
    std::map<int, int> vld, bank;

    // Bit 10 of dest_reg_addr selects SRCS (2) vs Dest (3)
    const int addr = attrs_map.count("dest_reg_addr") ? attrs_map.at("dest_reg_addr") : 0;
    const int done = attrs_map.count("done") ? attrs_map.at("done") : 0;
    const int src_reg_id = ((addr >> 10) & 1) ? 2 : 3; // SRC_S=2, DST=3
    const int flag = done ? 1 : 0;

    src.push_back(src_reg_id);
    vld[src_reg_id]  = flag;
    bank[src_reg_id] = flag;

    ins.set_src_int(src);
    ins.set_vld_upd_mask(vld);
    ins.set_bank_upd_mask(bank);
    return static_cast<int>(ins.get_addr()) + 4;
}

int TensixFunc::exec_sfpstore(isa::Instruction& ins) {
    if (!ins.operands.has_value()) return static_cast<int>(ins.get_addr()) + 4;
    const auto& attrs_map = ins.operands->attributes;

    std::vector<int> dst;
    std::map<int, int> vld, bank;

    const int addr = attrs_map.count("dest_reg_addr") ? attrs_map.at("dest_reg_addr") : 0;
    const int done = attrs_map.count("done") ? attrs_map.at("done") : 0;
    const int dst_reg_id = ((addr >> 10) & 1) ? 2 : 3;
    const int flag = done ? 1 : 0;

    dst.push_back(dst_reg_id);
    vld[dst_reg_id]  = flag;
    bank[dst_reg_id] = flag;

    ins.set_dst_int(dst);
    ins.set_vld_upd_mask(vld);
    ins.set_bank_upd_mask(bank);
    return static_cast<int>(ins.get_addr()) + 4;
}

int TensixFunc::exec_sfpnop(isa::Instruction& ins) {
    using R = isa::RegIndex;
    std::vector<int> src, dst;
    std::map<int, int> vld, bank;

    const auto attrs = ins.get_attr();
    const int dest_done  = attrs.count("dest_done")  ? attrs.at("dest_done")  : 0;
    const int srcs_wr    = attrs.count("srcs_wr_done")? attrs.at("srcs_wr_done"): 0;

    if (dest_done) {
        src.push_back(static_cast<int>(R::DST));
        vld[3]  = 1; bank[3] = 1;
    }
    if (srcs_wr) {
        dst.push_back(static_cast<int>(R::SRC_S));
        vld[2]  = 1; bank[2] = 1;
    }

    ins.set_src_int(src);
    ins.set_dst_int(dst);
    ins.set_vld_upd_mask(vld);
    ins.set_bank_upd_mask(bank);
    return static_cast<int>(ins.get_addr()) + 4;
}

int TensixFunc::exec_sfpu_math_i12(isa::Instruction& ins) {
    // Just advance PC — no register valid-mask side-effects for SFPU math.
    return static_cast<int>(ins.get_addr()) + 4;
}

int TensixFunc::exec_sfpu_math(isa::Instruction& ins) {
    return static_cast<int>(ins.get_addr()) + 4;
}

// ============================================================
// Replay / tile counters
// ============================================================

int TensixFunc::exec_replay(isa::Instruction& ins, int /*cycle*/) {
    const auto attrs = ins.get_attr();
    ins.set_mem_info("replay_load_mode",
                     attrs.count("load_mode") ? attrs.at("load_mode") : 0);
    ins.set_mem_info("replay_execute_while_loading",
                     attrs.count("execute_while_loading") ? attrs.at("execute_while_loading") : 0);
    ins.set_mem_info("replay_start_idx",
                     attrs.count("start_idx") ? attrs.at("start_idx") : 0);
    ins.set_mem_info("replay_len",
                     attrs.count("len") ? attrs.at("len") : 0);
    return static_cast<int>(ins.get_addr()) + 4;
}

int TensixFunc::exec_push_tiles(isa::Instruction& ins, int /*cycle*/) {
    const auto attrs = ins.get_attr();
    const int buffer   = attrs.count("buffer_sel") ? attrs.at("buffer_sel") : 0;
    const int num_tiles = attrs.count("num_tiles") ? attrs.at("num_tiles") : 0;

    const int32_t tiles_avail = spl_regs_.read_tile_counter(
        buffer, TensixSplReg::TileCounterField::TILES_AVAILABLE);
    const int32_t space_avail = spl_regs_.read_tile_counter(
        buffer, TensixSplReg::TileCounterField::SPACE_AVAILABLE);
    spl_regs_.write_tile_counter(buffer, TensixSplReg::TileCounterField::TILES_AVAILABLE,
                                  tiles_avail + num_tiles);
    spl_regs_.write_tile_counter(buffer, TensixSplReg::TileCounterField::SPACE_AVAILABLE,
                                  space_avail - num_tiles);
    return static_cast<int>(ins.get_addr()) + 4;
}

int TensixFunc::exec_pop_tiles(isa::Instruction& ins, int /*cycle*/) {
    const auto attrs = ins.get_attr();
    const int buffer   = attrs.count("buffer_sel") ? attrs.at("buffer_sel") : 0;
    const int num_tiles = attrs.count("num_tiles") ? attrs.at("num_tiles") : 0;

    const int32_t tiles_avail = spl_regs_.read_tile_counter(
        buffer, TensixSplReg::TileCounterField::TILES_AVAILABLE);
    const int32_t space_avail = spl_regs_.read_tile_counter(
        buffer, TensixSplReg::TileCounterField::SPACE_AVAILABLE);
    spl_regs_.write_tile_counter(buffer, TensixSplReg::TileCounterField::TILES_AVAILABLE,
                                  tiles_avail - num_tiles);
    spl_regs_.write_tile_counter(buffer, TensixSplReg::TileCounterField::SPACE_AVAILABLE,
                                  space_avail + num_tiles);
    return static_cast<int>(ins.get_addr()) + 4;
}

int TensixFunc::exec_wait_tiles(isa::Instruction& ins, int /*cycle*/) {
    const auto attrs = ins.get_attr();
    const int buffer      = attrs.count("buffer_sel") ? attrs.at("buffer_sel") : 0;
    const int target      = attrs.count("num_tiles")  ? attrs.at("num_tiles")  : 0;

    int stall_bits = 0;
    if (ins.operands.has_value() && ins.operands->all.count("stall_res"))
        stall_bits = ins.operands->all.at("stall_res");

    const auto dst_pipes =
        pipes_from_stall_bitmask(stall_bits, cfg_.pipe_grps, cfg_.pipes);
    ins.set_dst_pipes(dst_pipes);

    const int32_t tiles_avail = spl_regs_.read_tile_counter(
        buffer, TensixSplReg::TileCounterField::TILES_AVAILABLE);

    std::vector<std::string> src_pipe_names;
    if (tiles_avail < target) {
        ins.set_ex_pipe("SYNC");
        const auto it = cfg_.pipe_grps.find("SYNC");
        if (it != cfg_.pipe_grps.end()) {
            for (const auto& p : it->second) {
                const auto pit = std::find(cfg_.pipes.begin(), cfg_.pipes.end(), p);
                if (pit != cfg_.pipes.end())
                    src_pipe_names.push_back(p);
            }
        }
    }
    ins.set_src_pipes(src_pipe_names);
    return static_cast<int>(ins.get_addr()) + 4;
}

int TensixFunc::exec_wait_free(isa::Instruction& ins, int /*cycle*/) {
    const auto attrs = ins.get_attr();
    const int buffer = attrs.count("buffer_sel") ? attrs.at("buffer_sel") : 0;
    const int target = attrs.count("num_tiles")  ? attrs.at("num_tiles")  : 0;

    int stall_bits = 0;
    if (ins.operands.has_value() && ins.operands->all.count("stall_res"))
        stall_bits = ins.operands->all.at("stall_res");

    const auto dst_pipes =
        pipes_from_stall_bitmask(stall_bits, cfg_.pipe_grps, cfg_.pipes);
    ins.set_dst_pipes(dst_pipes);

    const int32_t space_avail = spl_regs_.read_tile_counter(
        buffer, TensixSplReg::TileCounterField::SPACE_AVAILABLE);

    std::vector<std::string> src_pipe_names;
    if (space_avail < target) {
        ins.set_ex_pipe("SYNC");
        const auto it = cfg_.pipe_grps.find("SYNC");
        if (it != cfg_.pipe_grps.end()) {
            for (const auto& p : it->second) {
                const auto pit = std::find(cfg_.pipes.begin(), cfg_.pipes.end(), p);
                if (pit != cfg_.pipes.end())
                    src_pipe_names.push_back(p);
            }
        }
    }
    ins.set_src_pipes(src_pipe_names);
    return static_cast<int>(ins.get_addr()) + 4;
}

// ============================================================
// MOP expansion
// ============================================================

std::pair<int, int> TensixFunc::parse_mop(const isa::Instruction& ins) const {
    const auto attrs = ins.get_attr();
    const int mop_type = attrs.count("mop_type") ? attrs.at("mop_type") : -1;
    assert(mop_type == 1 && "parse_mop: only double-loop MOP (mop_type==1) is supported");

    int mop_cfg_bank = 0;
    const bool is_ttqs = ins.kind.has_value() &&
                         *ins.kind == ttdecode::isa::instruction_kind::ttqs;
    if (is_ttqs) {
        const int done = attrs.count("done") ? attrs.at("done") : 0;
        assert(done == 0 && "parse_mop: dual-bank MOP CFG not supported");
    }
    return {mop_type, mop_cfg_bank};
}

std::vector<uint32_t> TensixFunc::build_ins_from_mop(isa::Instruction& ins) {
    const auto [mop_type, bank] = parse_mop(ins);
    (void)bank; // always 0

    const int thread = static_cast<int>(ins.get_thread_id());
    const int base   = 64 * thread;

    const uint32_t NOP = 0x2000000u;

    const int32_t loop0_len         = spl_regs_.read_reg(base + 0, TensixSplReg::SplRegType::MOP);
    const int32_t loop1_len         = spl_regs_.read_reg(base + 1, TensixSplReg::SplRegType::MOP);
    const int32_t loop_start_instr0 = spl_regs_.read_reg(base + 2, TensixSplReg::SplRegType::MOP);
    const int32_t loop_end_instr0   = spl_regs_.read_reg(base + 3, TensixSplReg::SplRegType::MOP);
    const int32_t loop_end_instr1   = spl_regs_.read_reg(base + 4, TensixSplReg::SplRegType::MOP);
    const int32_t loop_instr0       = spl_regs_.read_reg(base + 5, TensixSplReg::SplRegType::MOP);
    const int32_t loop_instr1       = spl_regs_.read_reg(base + 6, TensixSplReg::SplRegType::MOP);
    const int32_t loop0_last_instr  = spl_regs_.read_reg(base + 7, TensixSplReg::SplRegType::MOP);
    const int32_t loop1_last_instr  = spl_regs_.read_reg(base + 8, TensixSplReg::SplRegType::MOP);
    // reg[base+9] = mop_sw_ctrl (read but unused)

    auto is_nop = [&](int32_t w) { return static_cast<uint32_t>(w) == NOP; };

    std::vector<uint32_t> result;
    result.reserve(static_cast<size_t>(loop0_len * loop1_len * 2 + 4));

    for (int i = 0; i < loop0_len; ++i) {
        // Outer-loop start instruction
        if (!is_nop(loop_start_instr0))
            result.push_back(static_cast<uint32_t>(loop_start_instr0));

        for (int j = 0; j < loop1_len; ++j) {
            const bool last_inner = (j == loop1_len - 1);
            const bool last_outer = (i == loop0_len - 1);

            if (last_inner && last_outer) {
                // Both loops finishing
                if (!is_nop(loop_instr0) && !is_nop(loop_instr1) && !is_nop(loop0_last_instr)) {
                    result.push_back(static_cast<uint32_t>(loop_instr0));
                    result.push_back(static_cast<uint32_t>(loop0_last_instr));
                } else if (!is_nop(loop_instr0) && !is_nop(loop0_last_instr)) {
                    result.push_back(static_cast<uint32_t>(loop0_last_instr));
                } else {
                    assert(!is_nop(loop_instr0) && "MOP: at least one body instruction required");
                    result.push_back(static_cast<uint32_t>(loop_instr0));
                }
            } else if (last_inner) {
                // Only inner loop finishing
                if (!is_nop(loop_instr0) && !is_nop(loop_instr1) && !is_nop(loop1_last_instr)) {
                    result.push_back(static_cast<uint32_t>(loop_instr0));
                    result.push_back(static_cast<uint32_t>(loop1_last_instr));
                } else if (!is_nop(loop_instr0) && !is_nop(loop1_last_instr)) {
                    result.push_back(static_cast<uint32_t>(loop1_last_instr));
                } else {
                    assert(!is_nop(loop_instr0) && "MOP: at least one body instruction required");
                    result.push_back(static_cast<uint32_t>(loop_instr0));
                }
            } else {
                // Normal iteration
                if (!is_nop(loop_instr0)) result.push_back(static_cast<uint32_t>(loop_instr0));
                if (!is_nop(loop_instr1)) result.push_back(static_cast<uint32_t>(loop_instr1));
            }
        }

        // Outer-loop epilogue after last outer iteration
        if (i == loop0_len - 1) {
            if (!is_nop(loop_end_instr0)) result.push_back(static_cast<uint32_t>(loop_end_instr0));
            if (!is_nop(loop_end_instr1)) result.push_back(static_cast<uint32_t>(loop_end_instr1));
        }
    }

    return result;
}

// ============================================================
// Main dispatch
// ============================================================

int TensixFunc::exec_tt_ins(isa::Instruction& ins, int cycle) {
    const std::string& op = ins.get_op();

    // ── UNPACR family ────────────────────────────────────────────────
    if (op == "UNPACR0_TILE_INC"   || op == "UNPACR1_TILE_INC"   ||
        op == "UNPACR2_TILE_INC"   || op == "UNPACR_DEST_TILE_INC" ||
        op == "UNPACR0_FACE_INC"   || op == "UNPACR1_FACE_INC"   ||
        op == "UNPACR2_FACE_INC"   || op == "UNPACR_DEST_FACE_INC" ||
        op == "UNPACR0_ROW_INC"    || op == "UNPACR1_ROW_INC"    ||
        op == "UNPACR2_ROW_INC"    || op == "UNPACR_DEST_ROW_INC")
        return exec_unpacr_ti(ins);

    if (op == "UNPACR0_TILE"       || op == "UNPACR1_TILE"       ||
        op == "UNPACR2_TILE"       || op == "UNPACR_DEST_TILE"   ||
        op == "UNPACR0_FACE"       || op == "UNPACR1_FACE"       ||
        op == "UNPACR2_FACE"       || op == "UNPACR_DEST_FACE"   ||
        op == "UNPACR0_ROW"        || op == "UNPACR1_ROW"        ||
        op == "UNPACR2_ROW"        || op == "UNPACR_DEST_ROW")
        return exec_unpacr_t(ins);

    if (op == "UNPACR0_STRIDE"  || op == "UNPACR1_STRIDE"  ||
        op == "UNPACR2_STRIDE"  || op == "UNPACR_DEST_STRIDE")
        return exec_unpacr_s(ins);

    if (op == "UNPACR")            return exec_unpacr(ins);
    if (op == "UNPACR_TILE_MISC")  return exec_unpacr_tm(ins);
    if (op == "UNPACR_TILIZE")     return exec_unpacr_tz(ins);
    if (op == "UNPACR_NOP")        return exec_unpacr_nop(ins);

    // ── PACR family ──────────────────────────────────────────────────
    if (op == "PACR0_TILE"    || op == "PACR1_TILE"    || op == "PACR2_TILE"    ||
        op == "PACR0_FACE"    || op == "PACR1_FACE"    || op == "PACR2_FACE"    ||
        op == "PACR0_ROW"     || op == "PACR1_ROW"     || op == "PACR2_ROW"     ||
        op == "PACR0_TILE_INC"|| op == "PACR1_TILE_INC"|| op == "PACR2_TILE_INC"||
        op == "PACR0_FACE_INC"|| op == "PACR1_FACE_INC"|| op == "PACR2_FACE_INC"||
        op == "PACR0_ROW_INC" || op == "PACR1_ROW_INC" || op == "PACR2_ROW_INC")
        return exec_pacr_ti(ins);

    if (op == "PACR_STRIDE")    return exec_pacr_stride(ins);
    if (op == "PACR_UNTILIZE")  return exec_pacr_untilize(ins);

    // ── Math / pool ──────────────────────────────────────────────────
    if (op == "GAPOOL" || op == "GMPOOL") return exec_gpool(ins);
    if (op == "ELWADD")  return exec_elwadd(ins);
    if (op == "ELWSUB")  return exec_elwsub(ins);
    if (op == "ELWMUL")  return exec_elwmul(ins);
    if (op == "MVMUL")   return exec_mvmul(ins);
    if (op == "MVMULDI") return exec_mvmuldi(ins);

    // ── Trivial no-ops ───────────────────────────────────────────────
    if (op == "ATGETM")  return exec_atgetm(ins);
    if (op == "ATRELM")  return exec_atrelm(ins);
    if (op == "DMANOP")  return exec_dmanop(ins);
    if (op == "NOP")     return exec_nop(ins);
    if (op == "SETADCXX")return exec_setadcxx(ins);
    if (op == "SETADCXY")return exec_setadcxy(ins);
    if (op == "SETADCZW")return exec_setadczw(ins);
    if (op == "WRCFG")   return exec_wrcfg(ins);
    if (op == "ZEROSRC") return exec_zerosrc(ins);
    if (op == "ZEROACC") return exec_zeroacc(ins);

    // ── Semaphore ────────────────────────────────────────────────────
    if (op == "SEMINIT") return exec_seminit(ins);
    if (op == "SEMGET")  return exec_semget(ins);
    if (op == "SEMPOST") return exec_sempost(ins);
    if (op == "SEMWAIT") return exec_semwait(ins);

    // ── cfg / RWC ────────────────────────────────────────────────────
    if (op == "SETC16")   return exec_setc16(ins);
    if (op == "SETRWC")   return exec_setrwc(ins);
    if (op == "STALLWAIT")return exec_stallwait(ins);
    if (op == "RMWCIB0" || op == "RMWCIB1" ||
        op == "RMWCIB2" || op == "RMWCIB3")
        return exec_rmwcib(ins);

    // ── DVALID / MOV ─────────────────────────────────────────────────
    if (op == "CLEARDVALID") return exec_clrdvalid(ins);
    if (op == "MOVA2D"  || op == "MOVB2D" || op == "MOVD2A" ||
        op == "MOVD2B"  || op == "MOVB2A")
        return exec_mov(ins);
    if (op == "TRNSPSRCA" || op == "TRNSPSRCB") return exec_trnsp(ins);

    // ── SFPU ─────────────────────────────────────────────────────────
    if (op == "SFPLOAD")  return exec_sfpload(ins);
    if (op == "SFPLOADI") return exec_sfploadi(ins);
    if (op == "SFPCONFIG")return exec_sfpconfig(ins);
    if (op == "SFPSTORE") return exec_sfpstore(ins);
    if (op == "SFPNOP")   return exec_sfpnop(ins);

    if (op == "SFPABS"  || op == "SFPAND"    || op == "SFPARECIP" ||
        op == "SFPCOMPC"|| op == "SFPDIVP2"  || op == "SFPENCC"   ||
        op == "SFPEXEXP"|| op == "SFPEXMAN"  || op == "SFPGT"     ||
        op == "SFPIADD" || op == "SFPLE"     || op == "SFPLZ"     ||
        op == "SFPMOV"  || op == "SFPNOT"    || op == "SFPOR"     ||
        op == "SFPPOPC" || op == "SFPPUSHC"  || op == "SFPSETCC"  ||
        op == "SFPSETEXP"|| op == "SFPSETMAN"|| op == "SFPSETSGN" ||
        op == "SFPSHFT" || op == "SFPTRANSP" || op == "SFPXOR")
        return exec_sfpu_math_i12(ins);

    if (op == "SFPADD" || op == "SFPMAD" || op == "SFPMUL" || op == "SFPMUL24")
        return exec_sfpu_math(ins);

    // ── Tile index ───────────────────────────────────────────────────
    if (op == "SET_DST_TILE_FACE_ROW_IDX" || op == "INC_DST_TILE_FACE_ROW_IDX")
        return exec_dst_tile_face_row_idx(ins);
    if (op == "SET_SRC_TILE_FACE_ROW_IDX" || op == "INC_SRC_TILE_FACE_ROW_IDX")
        return exec_src_tile_face_row_idx(ins);

    // ── Replay / tile counters ───────────────────────────────────────
    if (op == "REPLAY")     return exec_replay(ins, cycle);
    if (op == "PUSH_TILES") return exec_push_tiles(ins, cycle);
    if (op == "POP_TILES")  return exec_pop_tiles(ins, cycle);
    if (op == "WAIT_TILES") return exec_wait_tiles(ins, cycle);
    if (op == "WAIT_FREE")  return exec_wait_free(ins, cycle);

    // ── Unknown opcode ───────────────────────────────────────────────
    std::printf("WARNING: TensixFunc: unsupported opcode '%s'\n", op.c_str());
    return static_cast<int>(ins.get_addr()) + 4;
}

} // namespace neosim::units
