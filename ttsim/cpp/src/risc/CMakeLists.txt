# Track 3: TRISC register file.
# Track 5a: TRISC memory model and RV32 execution engine.
target_sources(neosim PRIVATE
  trisc_regs.cpp
  trisc_mem_func.cpp
  trisc_func.cpp
)
