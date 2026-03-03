# Unit tests — one executable per Sparta unit (migration plan §7.1).
# Add tests here as each Track completes.
#
# Usage:
#   neosim_unit_test(test_tensix_reg test_tensix_reg.cpp)
#   neosim_unit_test(test_risc_reg   test_risc_reg.cpp)
#   ...

function(neosim_unit_test NAME SOURCE)
  add_executable(${NAME} ${SOURCE})
  target_link_libraries(${NAME} PRIVATE
    neosim
    deps::gtest_main
  )
  add_test(NAME ${NAME} COMMAND ${NAME})
  set_tests_properties(${NAME} PROPERTIES LABELS "unit")
endfunction()

# Track 2: Instruction representation
neosim_unit_test(test_instruction test_instruction.cpp)

# Track 3+: uncomment as units are implemented
# neosim_unit_test(test_tensix_reg   test_tensix_reg.cpp)
# neosim_unit_test(test_risc_reg     test_risc_reg.cpp)
# neosim_unit_test(test_pipe_resource test_pipe_resource.cpp)
# neosim_unit_test(test_tensix_spl_reg test_tensix_spl_reg.cpp)
# neosim_unit_test(test_rob          test_rob.cpp)
# neosim_unit_test(test_replay       test_replay.cpp)
# neosim_unit_test(test_scratchpad   test_scratchpad.cpp)
# neosim_unit_test(test_trisc_func   test_trisc_func.cpp)
# neosim_unit_test(test_tensix_func  test_tensix_func.cpp)
# neosim_unit_test(test_decoder_service test_decoder_service.cpp)
