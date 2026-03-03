# Track 3: Register file data-model sources.
# Track 4: L1 memory latency model.
# Track 5b: Tensix instruction execution engine.
# Track 6: Thread execution engine data models.
target_sources(neosim PRIVATE
  tensix_reg.cpp
  risc_reg.cpp
  pipe_resource.cpp
  tensix_spl_reg.cpp
  scratchpad.cpp
  tensix_func.cpp
  rob.cpp
  replay.cpp
  thread_unit.cpp
)
