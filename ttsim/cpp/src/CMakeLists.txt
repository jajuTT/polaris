# NeoSim library.
#
# Migration track progression:
#   Track 2:  isa/instruction.cpp, isa/elf_loader.cpp  ← current
#   Track 3:  add units/tensix_reg.cpp, units/risc_reg.cpp, ...
#   Track 4:  add units/scratchpad.cpp
#   Track 5a: add risc/trisc_func.cpp, risc/trisc_regs.cpp, risc/trisc_mem.cpp
#   Track 5b: add units/tensix_func.cpp
#   Track 6:  add units/thread_unit.cpp, units/rob.cpp, units/replay.cpp
#   Track 7:  add units/pipe_unit.cpp
#   Track 8:  add core/neosim_top.cpp, core/tensix_core.cpp, main.cpp

add_library(neosim STATIC)

target_include_directories(neosim PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

# target_link_libraries(PUBLIC) propagation of ttdecode's include dirs is
# unreliable across CMake versions when the dependency comes from an
# EXCLUDE_FROM_ALL subdirectory accessed via a cross-directory ALIAS target.
# Add the ttdecode include path explicitly as belt-and-suspenders.
if(DEFINED NEOSIM_TTDECODE_INCLUDE_DIR)
  target_include_directories(neosim PUBLIC
    $<BUILD_INTERFACE:${NEOSIM_TTDECODE_INCLUDE_DIR}>
  )
endif()

target_compile_options(neosim PRIVATE -Werror -Wall -Wextra)

target_link_libraries(neosim PUBLIC
  deps::ttdecode
  deps::sparta
  deps::yaml-cpp
  deps::spdlog
)

add_subdirectory(isa)
# Track 3+: add_subdirectory(units)
# Track 5a: add_subdirectory(risc)
# Track 8:  add_subdirectory(core)
