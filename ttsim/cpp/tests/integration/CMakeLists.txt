# Integration tests (migration plan §T8.4–T8.6).
# Each test is an executable that runs the full NeoSim pipeline against real ELF files.
#
# POLARIS_DIR is defined to the absolute path of the polaris/ root directory so
# that integration tests can resolve relative paths from the inputcfg JSON files.

function(neosim_integration_test NAME SOURCE)
  # Derive POLARIS_DIR as two levels above CMAKE_SOURCE_DIR (which is polaris/ttsim/cpp).
  get_filename_component(_polaris_dir "${CMAKE_SOURCE_DIR}/../.." ABSOLUTE)

  add_executable(${NAME} ${SOURCE})
  target_link_libraries(${NAME} PRIVATE
    neosim
    deps::gtest_main
  )
  target_compile_definitions(${NAME} PRIVATE
    POLARIS_DIR="${_polaris_dir}"
  )
  add_test(NAME ${NAME} COMMAND ${NAME})
  set_tests_properties(${NAME} PROPERTIES LABELS "integration")
endfunction()

# T8.4 / T8.5 / T8.6: elwadd end-to-end test against feb20 ELFs.
neosim_integration_test(test_neo_sim test_neo_sim.cpp)
