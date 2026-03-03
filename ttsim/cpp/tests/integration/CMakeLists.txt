# Integration tests (migration plan §7.2).
# Add tests here as Tracks 6-8 complete.

function(neosim_integration_test NAME SOURCE)
  add_executable(${NAME} ${SOURCE})
  target_link_libraries(${NAME} PRIVATE
    neosim
    deps::gtest_main
  )
  add_test(NAME ${NAME} COMMAND ${NAME})
  set_tests_properties(${NAME} PROPERTIES LABELS "integration")
endfunction()

# Track 6+: uncomment as integration scenarios are ready
# neosim_integration_test(test_single_thread_risc    test_single_thread_risc.cpp)
# neosim_integration_test(test_single_thread_tensix  test_single_thread_tensix.cpp)
# neosim_integration_test(test_multi_thread          test_multi_thread.cpp)
# neosim_integration_test(test_unpacker_flow         test_unpacker_flow.cpp)
# neosim_integration_test(test_packer_flow           test_packer_flow.cpp)
