#pragma once

#include "neosim/isa/instruction.hpp"
#include "elf/elf_parser.hpp"
#include "isa/isa.hpp"

#include <map>
#include <set>
#include <string>
#include <vector>

namespace neosim::isa {

/// Maps function names to their decoded instruction sequences.
/// Key = function symbol name (e.g. "brisc_kernel_main").
/// Value = instructions in address order.
using FunctionMap = std::map<std::string, std::vector<Instruction>>;

/// Wraps ttdecode::elf::parser to produce neosim::isa::Instruction objects.
///
/// Architecture is auto-detected from the ELF's .riscv.attributes section —
/// no external arch_tag parameter is required or accepted.
///
/// The instruction sets (loaded from YAML) are cached in the ElfLoader object,
/// so multiple decode_function() calls do not re-parse YAML files.
class ElfLoader {
public:
    /// Open an ELF file and cache its instruction-kind set and ISA YAML.
    ///
    /// Throws std::runtime_error if the file cannot be opened or its
    /// architecture attributes cannot be read.
    explicit ElfLoader(const std::string& elf_path);

    /// Decode every function found in the ELF.
    FunctionMap decode_all() const;

    /// Decode a single named function.
    ///
    /// Throws std::out_of_range if the function is not found in the ELF.
    std::vector<Instruction> decode_function(const std::string& fn_name) const;

    // ------------------------------------------------------------------
    // Diagnostics / accessors
    // ------------------------------------------------------------------

    /// Instruction kinds detected from the ELF (e.g. {rv32, ttqs}).
    const std::set<ttdecode::isa::instruction_kind>& kinds() const { return kinds_; }

    /// Returns true if the underlying parser opened the file successfully.
    bool is_valid() const { return parser_.is_valid(); }

private:
    /// Convert a raw decoded_instruction → Instruction.
    static Instruction wrap(const ttdecode::decode::decoded_instruction& di);

    ttdecode::elf::parser                      parser_;
    std::set<ttdecode::isa::instruction_kind>  kinds_;
    ttdecode::isa::instruction_sets            sets_;
};

} // namespace neosim::isa
