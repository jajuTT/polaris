# Stand-alone NeoSim examples.
# These are regular executables, not ctest tests.
# Build with: cmake --build <build_dir> --target neo_sim_example
# Run with:   ./build/examples/neo_sim_example [polaris_dir]
#
# POLARIS_DIR is defined to the absolute path of polaris/ so that
# the example can resolve relative paths without any command-line arg.

get_filename_component(_polaris_dir "${CMAKE_SOURCE_DIR}/../.." ABSOLUTE)

add_executable(neo_sim_example neo_sim_example.cpp)
target_link_libraries(neo_sim_example PRIVATE neosim)
target_compile_definitions(neo_sim_example PRIVATE
    POLARIS_DIR="${_polaris_dir}"
)
