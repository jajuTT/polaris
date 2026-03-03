#include "neosim/isa/elf_loader.hpp"

#include <stdexcept>

namespace neosim::isa {

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

ElfLoader::ElfLoader(const std::string& elf_path)
    : parser_(elf_path)
{
    if (!parser_.is_valid()) {
        throw std::runtime_error("ElfLoader: cannot open or parse ELF file '" + elf_path + "'");
    }

    kinds_ = parser_.get_instruction_kinds();
    sets_  = ttdecode::isa::get_instruction_sets(kinds_);
}

// ---------------------------------------------------------------------------
// Decode helpers
// ---------------------------------------------------------------------------

Instruction ElfLoader::wrap(const ttdecode::decode::decoded_instruction& di) {
    return Instruction(di);
}

// ---------------------------------------------------------------------------
// decode_all
// ---------------------------------------------------------------------------

FunctionMap ElfLoader::decode_all() const {
    // parser_.decode(sets_) returns map<function_symbol, decoded_instructions>
    const auto raw = parser_.decode(sets_);

    FunctionMap result;
    for (const auto& [sym, dis] : raw) {
        std::vector<Instruction> instrs;
        instrs.reserve(dis.size());
        for (const auto& di : dis) {
            instrs.push_back(wrap(di));
        }
        result.emplace(sym.name, std::move(instrs));
    }
    return result;
}

// ---------------------------------------------------------------------------
// decode_function
// ---------------------------------------------------------------------------

std::vector<Instruction> ElfLoader::decode_function(const std::string& fn_name) const {
    // Retrieve the function symbol by name (throws if not found).
    const ttdecode::elf::function_symbol sym = parser_.get_function(fn_name);

    const auto raw = parser_.decode(sym, sets_);

    std::vector<Instruction> result;
    result.reserve(raw.size());
    for (const auto& di : raw) {
        result.push_back(wrap(di));
    }
    return result;
}

} // namespace neosim::isa
