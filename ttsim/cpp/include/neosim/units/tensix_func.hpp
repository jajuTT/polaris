// SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "neosim/isa/instruction.hpp"
#include "neosim/units/tensix_spl_reg.hpp"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace neosim::units {

/// Tensix instruction execution engine — replaces Python class `tensixFunc`.
///
/// Dispatches Tensix (ttqs / ttwh) instructions to per-opcode handlers that:
///   - Annotate the Instruction token in-place (ex_pipe, src/dst int lists,
///     vld/bank masks, src/dst formats, num_datums, src/dst pipes, etc.)
///   - Return the next program-counter address (PC + 4 for all opcodes).
///
/// MOP expansion (hardware double-loop sequencer) is also implemented here via
/// `build_ins_from_mop()`.
///
/// No Sparta dependency — pure C++20 data model, testable on macOS ARM64.
class TensixFunc {
public:
    using PipeGroups = std::map<std::string, std::vector<std::string>>;

    struct Config {
        int         core_id   = 0;
        std::string arch;       ///< "ttqs", "ttwh"
        int         llk_group = 0; ///< from args['llkVersionTag'].group (0, 1, or 2)
        /// Pipe groups: UNPACK, PACK, MATH, SFPU, CFG, XMOV, THCON, TDMA, SYNC
        PipeGroups  pipe_grps;
        /// Flat ordered pipe name list (index = pipe ID used in src_pipes/dst_pipes)
        std::vector<std::string> pipes;
    };

    explicit TensixFunc(const Config& cfg, TensixSplReg& spl_regs);

    // ----------------------------------------------------------------
    // Main entry points
    // ----------------------------------------------------------------

    /// Main Tensix instruction dispatcher.
    ///
    /// Annotates @p ins in-place (ex_pipe, src/dst int lists, vld/bank masks,
    /// src/dst formats, num_datums, src/dst pipes, etc.).
    /// Returns next PC address (PC + 4 for all instructions including REPLAY;
    /// for REPLAY the caller reads replay_* fields from ins.mem_info_).
    int exec_tt_ins(isa::Instruction& ins, int cycle);

    /// Expand a MOP instruction into raw 32-bit instruction words.
    ///
    /// Reads 10 MOP cfg registers from TensixSplReg for ins.get_thread_id().
    /// Returns the ordered word list with NOPs (0x2000000) filtered out.
    std::vector<uint32_t> build_ins_from_mop(isa::Instruction& ins);

    /// Upper bound on UNPACKER2/PACKER1 autoloop iterations.
    /// = (max_instrn_count + 1) * (max_instrn_loop_count + 1)
    int max_autoloop_iterations() const;

    /// MOP cfg register base address from TensixSplReg.
    uint32_t mop_cfg_addr() const;

private:
    // ----------------------------------------------------------------
    // Per-opcode instruction handlers (all annotate ins and return next PC)
    // ----------------------------------------------------------------

    // UNPACR family (Tile / Face / Row variants with and without INC)
    int exec_unpacr_ti(isa::Instruction& ins);   ///< UNPACR*_*_INC
    int exec_unpacr_t(isa::Instruction& ins);    ///< UNPACR*_TILE / FACE / ROW
    int exec_unpacr_s(isa::Instruction& ins);    ///< UNPACR*_STRIDE
    int exec_unpacr_tm(isa::Instruction& ins);   ///< UNPACR_TILE_MISC
    int exec_unpacr_tz(isa::Instruction& ins);   ///< UNPACR_TILIZE
    int exec_unpacr_nop(isa::Instruction& ins);  ///< UNPACR_NOP
    int exec_unpacr(isa::Instruction& ins);      ///< UNPACR (generic, non-ttqs)

    // PACR family
    int exec_pacr_ti(isa::Instruction& ins);       ///< PACR*_TILE/FACE/ROW[_INC]
    int exec_pacr_stride(isa::Instruction& ins);
    int exec_pacr_untilize(isa::Instruction& ins);

    // Math / pool
    int exec_gpool(isa::Instruction& ins);       ///< GAPOOL, GMPOOL
    int exec_elwadd(isa::Instruction& ins);
    int exec_elwsub(isa::Instruction& ins);
    int exec_elwmul(isa::Instruction& ins);
    int exec_mvmul(isa::Instruction& ins);
    int exec_mvmuldi(isa::Instruction& ins);

    // Trivial no-op families
    int exec_atgetm(isa::Instruction& ins);
    int exec_atrelm(isa::Instruction& ins);
    int exec_dmanop(isa::Instruction& ins);
    int exec_nop(isa::Instruction& ins);

    // Semaphore
    int exec_seminit(isa::Instruction& ins);
    int exec_semget(isa::Instruction& ins);
    int exec_sempost(isa::Instruction& ins);
    int exec_semwait(isa::Instruction& ins);

    // ADC / cfg / RWC
    int exec_setadcxx(isa::Instruction& ins);
    int exec_setadcxy(isa::Instruction& ins);
    int exec_setadczw(isa::Instruction& ins);
    int exec_setc16(isa::Instruction& ins);
    int exec_setrwc(isa::Instruction& ins);
    int exec_stallwait(isa::Instruction& ins);
    int exec_wrcfg(isa::Instruction& ins);
    int exec_zerosrc(isa::Instruction& ins);
    int exec_zeroacc(isa::Instruction& ins);

    // DVALID / MOV
    int exec_clrdvalid(isa::Instruction& ins);
    int exec_mov(isa::Instruction& ins);         ///< MOVA2D, MOVB2D, MOVD2A, MOVD2B, MOVB2A
    int exec_trnsp(isa::Instruction& ins);       ///< TRNSPSRCA / TRNSPSRCB
    int exec_rmwcib(isa::Instruction& ins);      ///< RMWCIB0-3

    // SFPU
    int exec_sfpload(isa::Instruction& ins);
    int exec_sfploadi(isa::Instruction& ins);
    int exec_sfpconfig(isa::Instruction& ins);
    int exec_sfpstore(isa::Instruction& ins);
    int exec_sfpnop(isa::Instruction& ins);
    int exec_sfpu_math_i12(isa::Instruction& ins); ///< 24+ SFPU single-operand
    int exec_sfpu_math(isa::Instruction& ins);     ///< SFPADD, SFPMAD, SFPMUL, SFPMUL24

    // Tile index
    int exec_dst_tile_face_row_idx(isa::Instruction& ins);
    int exec_src_tile_face_row_idx(isa::Instruction& ins);

    // Replay / tile counters
    int exec_replay(isa::Instruction& ins, int cycle);
    int exec_push_tiles(isa::Instruction& ins, int cycle);
    int exec_pop_tiles(isa::Instruction& ins, int cycle);
    int exec_wait_tiles(isa::Instruction& ins, int cycle);
    int exec_wait_free(isa::Instruction& ins, int cycle);

    // ----------------------------------------------------------------
    // MOP helper
    // ----------------------------------------------------------------

    /// Validates MOP attrs; returns {mop_type, mop_cfg_bank}.
    std::pair<int, int> parse_mop(const isa::Instruction& ins) const;

    // ----------------------------------------------------------------
    // Cfg-write instruction-state side-effects (for SETC16 / RMWCIB)
    // ----------------------------------------------------------------

    /// Implements the instruction-level side-effects of a cfg register write:
    /// writes @p val to the CFG store, then updates @p ins vld/bank masks and
    /// cond_chk/cond_wri maps for DEST_DVALID_CTRL class registers.
    void apply_cfg_reg_write(uint32_t addr, int32_t val, isa::Instruction& ins);

    // ----------------------------------------------------------------
    // Tile-dimension / format helpers (all read cfg registers)
    // ----------------------------------------------------------------

    /// Returns {reg_name, format_value} from BUFFER_DESCRIPTOR_TABLE cfg reg.
    std::pair<std::string, int> get_cb_format(const isa::Instruction& ins) const;

    /// Returns {reg_name, dim_value} for dimension @p dim ('X', 'Y', or 'Z').
    std::pair<std::string, int> get_cb_dim(const isa::Instruction& ins, char dim) const;

    /// Returns {reg_name, format_value} from THCON_{ex_pipe}_REG0_{IN/OUT}_DATA_FORMAT.
    std::pair<std::string, int> get_reg_format(const isa::Instruction& ins,
                                               const std::string& ex_pipe) const;

    /// Compute total num_datums = Z * Y * X from tile-dimension cfg registers.
    int get_num_datums(const isa::Instruction& ins) const;

    /// Compute num_datums from Row_Cnt_Enc attribute (for TILIZE / UNTILIZE).
    int compute_num_datums_from_row_cnt_enc(const isa::Instruction& ins) const;

    // ----------------------------------------------------------------
    // UNPACR common helper
    // ----------------------------------------------------------------

    /// Derive reg_id and ex_pipe from the UNPACR mnemonic prefix.
    /// Populates @p dst, @p vld, @p bank from SetDatValid / Set_Dvalid attr.
    void parse_unpacr_prefix(const isa::Instruction& ins,
                              std::vector<int>& dst,
                              std::map<int, int>& vld,
                              std::map<int, int>& bank,
                              std::string& ex_pipe) const;

    // ----------------------------------------------------------------
    // Data members
    // ----------------------------------------------------------------

    Config        cfg_;
    TensixSplReg& spl_regs_;
};

} // namespace neosim::units
