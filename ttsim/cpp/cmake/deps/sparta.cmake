# cmake/deps/sparta.cmake
# Resolve Sparta from the sparcians/map repository.
#
# Platform behaviour:
#   Linux / x86_64  — full ExternalProject build → STATIC imported target.
#   macOS ARM64     — source-only fetch → INTERFACE target (headers for IDE).
#                     Linking will fail; compile on a Linux machine.
#
# Local source cache: __third_party/sparta-src/   (MAP repository clone)
# Install prefix:     __third_party/sparta-install/ (Linux full build only)
#
# Provides: deps::sparta

include_guard(GLOBAL)

option(FETCH_DEPS "Allow fetching dependencies at configure time" ON)
set(SPARTA_GIT_TAG "map_v2.1.20" CACHE STRING "Git tag/branch/commit to fetch for MAP/Sparta")

# Detect Apple Silicon at configure time
if(CMAKE_HOST_SYSTEM_NAME STREQUAL "Darwin" AND
   CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL "arm64")
  set(_SPARTA_DEFAULT_HEADERS_ONLY TRUE)
else()
  set(_SPARTA_DEFAULT_HEADERS_ONLY FALSE)
endif()

option(SPARTA_HEADERS_ONLY
  "Use Sparta headers without building (auto-enabled on macOS ARM64)"
  ${_SPARTA_DEFAULT_HEADERS_ONLY})

set(_SPARTA_REPO        "https://github.com/sparcians/map.git")
set(_SPARTA_SOURCE_DIR  "${CMAKE_SOURCE_DIR}/__third_party/sparta-src")
set(_SPARTA_INSTALL_DIR "${CMAKE_SOURCE_DIR}/__third_party/sparta-install")
set(_SPARTA_LIB         "${_SPARTA_INSTALL_DIR}/lib/libsparta.a")

# ----------------------------------------------------------------
# 1) Already installed: find_package from install prefix (Linux only)
# ----------------------------------------------------------------
if(NOT SPARTA_HEADERS_ONLY AND EXISTS "${_SPARTA_INSTALL_DIR}")
  list(APPEND CMAKE_PREFIX_PATH "${_SPARTA_INSTALL_DIR}")
  find_package(Sparta CONFIG QUIET
    PATHS "${_SPARTA_INSTALL_DIR}/lib/cmake/sparta"
          "${_SPARTA_INSTALL_DIR}"
    NO_DEFAULT_PATH)
  if(Sparta_FOUND AND TARGET Sparta::sparta)
    if(NOT TARGET deps::sparta)
      add_library(deps::sparta ALIAS Sparta::sparta)
    endif()
    message(STATUS "Sparta: using installed package at ${_SPARTA_INSTALL_DIR}")
    return()
  endif()
  # find_package didn't work; fall through to manual IMPORTED target check
  if(EXISTS "${_SPARTA_LIB}")
    # Pre-create the include dir: CMake 3.22+ validates INTERFACE_INCLUDE_DIRECTORIES
    # of IMPORTED targets at generate time. Sparta may not install headers to
    # sparta-install/include/ on all builds; pre-creating avoids the path check.
    file(MAKE_DIRECTORY "${_SPARTA_INSTALL_DIR}/include")
    add_library(deps_sparta STATIC IMPORTED GLOBAL)
    set_target_properties(deps_sparta PROPERTIES IMPORTED_LOCATION "${_SPARTA_LIB}")
    target_include_directories(deps_sparta INTERFACE "${_SPARTA_INSTALL_DIR}/include")
    # Sparta requires Boost; declare its transitive deps here if find_package
    # didn't resolve them. Adjust as needed on the target machine.
    find_package(Boost QUIET COMPONENTS system filesystem timer serialization regex)
    if(Boost_FOUND)
      target_link_libraries(deps_sparta INTERFACE
        Boost::system Boost::filesystem Boost::timer
        Boost::serialization Boost::regex)
    endif()
    if(NOT TARGET deps::sparta)
      add_library(deps::sparta ALIAS deps_sparta)
    endif()
    message(STATUS "Sparta: using pre-built library at ${_SPARTA_LIB}")
    return()
  endif()
endif()

# ----------------------------------------------------------------
# Helper: create INTERFACE target pointing to Sparta source headers.
# Used for both the headers-only path and as a stub before the full
# ExternalProject build completes.
# ----------------------------------------------------------------
macro(_neosim_sparta_interface_target)
  if(NOT TARGET deps_sparta)
    add_library(deps_sparta INTERFACE IMPORTED GLOBAL)
    # Sparta's public headers live under sparta/include/ in the MAP tree.
    # The sparta/ directory itself is also needed for some internal includes.
    # Pre-create the directories so CMake does not fail the path-existence
    # check at generate time (the actual headers arrive via ExternalProject
    # during the build step, not at configure time).
    file(MAKE_DIRECTORY
      "${_SPARTA_SOURCE_DIR}/sparta/include"
      "${_SPARTA_SOURCE_DIR}/sparta"
    )
    target_include_directories(deps_sparta INTERFACE
      "${_SPARTA_SOURCE_DIR}/sparta/include"
      "${_SPARTA_SOURCE_DIR}/sparta"
    )
  endif()
  if(NOT TARGET deps::sparta)
    add_library(deps::sparta ALIAS deps_sparta)
  endif()
endmacro()

# ----------------------------------------------------------------
# 2) Source already present in __third_party/sparta-src/
# ----------------------------------------------------------------
if(EXISTS "${_SPARTA_SOURCE_DIR}/sparta/CMakeLists.txt")
  if(SPARTA_HEADERS_ONLY)
    _neosim_sparta_interface_target()
    message(STATUS "Sparta: headers-only at ${_SPARTA_SOURCE_DIR} (macOS ARM64 or SPARTA_HEADERS_ONLY=ON)")
    message(WARNING "Sparta is headers-only — linking will fail. Build on a Linux/x86_64 machine.")
    return()
  endif()
  # Source present, proceed to ExternalProject build below
  message(STATUS "Sparta: source found at ${_SPARTA_SOURCE_DIR}, will build")
endif()

# ----------------------------------------------------------------
# 3) Fetch source from GitHub using FetchContent (configure-time download)
# ----------------------------------------------------------------
if(NOT EXISTS "${_SPARTA_SOURCE_DIR}/sparta/CMakeLists.txt")
  if(NOT FETCH_DEPS)
    message(FATAL_ERROR
      "Sparta source not found at ${_SPARTA_SOURCE_DIR}.\n"
      "Clone ${_SPARTA_REPO} there manually, or enable FETCH_DEPS=ON.")
  endif()

  find_package(Git QUIET)
  if(NOT GIT_FOUND)
    message(FATAL_ERROR "Git is required to fetch Sparta but was not found.")
  endif()

  message(STATUS "Sparta: fetching from ${_SPARTA_REPO} (tag: ${SPARTA_GIT_TAG})...")
  
  include(FetchContent)
  FetchContent_Declare(sparta_source
    GIT_REPOSITORY "${_SPARTA_REPO}"
    GIT_TAG        "${SPARTA_GIT_TAG}"
    GIT_SHALLOW    TRUE
    GIT_SUBMODULES_RECURSE TRUE
    SOURCE_DIR     "${_SPARTA_SOURCE_DIR}"
  )
  
  # Populate downloads at configure time (not build time like ExternalProject)
  FetchContent_Populate(sparta_source)
  
  # Patch: Fix std::filesystem::extension bug in MemoryProfiler.cpp
  # Bug exists in v2.1.20: std::filesystem::extension(path) should be path(path).extension()
  set(_MEMORY_PROFILER_FILE "${_SPARTA_SOURCE_DIR}/sparta/src/MemoryProfiler.cpp")
  if(EXISTS "${_MEMORY_PROFILER_FILE}")
    file(READ "${_MEMORY_PROFILER_FILE}" _MP_CONTENT)
    string(REPLACE 
      "std::filesystem::extension(dest_file_)"
      "std::filesystem::path(dest_file_).extension()"
      _MP_CONTENT "${_MP_CONTENT}")
    file(WRITE "${_MEMORY_PROFILER_FILE}" "${_MP_CONTENT}")
    message(STATUS "Sparta: patched MemoryProfiler.cpp std::filesystem::extension bug")
  endif()
  
  if(SPARTA_HEADERS_ONLY)
    _neosim_sparta_interface_target()
    message(STATUS "Sparta: fetched source for headers-only use (macOS ARM64)")
    message(WARNING "Sparta is headers-only — linking will fail. Build on a Linux/x86_64 machine.")
    return()
  endif()
  
  message(STATUS "Sparta: source fetched to ${_SPARTA_SOURCE_DIR}")
endif()

# ----------------------------------------------------------------
# 4) Build as subdirectory (Linux / x86_64)
# ----------------------------------------------------------------
if(NOT SPARTA_HEADERS_ONLY)
  # Use add_subdirectory instead of ExternalProject to share CMake context.
  # This allows Sparta to see deps::yaml-cpp and other parent project targets.
  
  # Sparta's CMakeLists.txt unconditionally adds test/ and example/ as
  # subdirectories (lines 256-257). Even with EXCLUDE_FROM_ALL, CMake still
  # configures those subdirs and validates all dependencies. To prevent errors,
  # we inject a wrapper CMakeLists.txt that builds only the sparta library.
  
  set(_SPARTA_WRAPPER_DIR "${CMAKE_BINARY_DIR}/__sparta_wrapper")
  file(MAKE_DIRECTORY "${_SPARTA_WRAPPER_DIR}")
  
  # Create a minimal CMakeLists.txt that includes only sparta library sources
  file(WRITE "${_SPARTA_WRAPPER_DIR}/CMakeLists.txt" "
cmake_minimum_required(VERSION 3.19)
project(sparta_wrapper CXX)

# Inherit settings from parent - use Sparta's defaults
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(GEN_DEBUG_INFO ON CACHE BOOL \"\" FORCE)           # Sparta default: ON
set(USING_SIMDB ON CACHE BOOL \"\" FORCE)              # Sparta default: ON
set(COMPILE_WITH_PYTHON OFF CACHE BOOL \"\" FORCE)     # No Python bindings

set(SPARTA_BASE \"${_SPARTA_SOURCE_DIR}/sparta\")
set(SIMDB_BASE \"${_SPARTA_SOURCE_DIR}/sparta/simdb\")

# Verify SimDB submodule is present (required for USING_SIMDB=ON)
if(NOT EXISTS \"\${SIMDB_BASE}/CMakeLists.txt\")
  message(WARNING \"SimDB not found at \${SIMDB_BASE}. Attempting to initialize submodules...\")
  execute_process(
    COMMAND git submodule update --init --recursive
    WORKING_DIRECTORY \"${_SPARTA_SOURCE_DIR}\"
    RESULT_VARIABLE _SIMDB_INIT_RESULT
    OUTPUT_VARIABLE _SIMDB_INIT_OUTPUT
    ERROR_VARIABLE _SIMDB_INIT_ERROR
  )
  if(NOT _SIMDB_INIT_RESULT EQUAL 0)
    message(FATAL_ERROR \"Failed to initialize SimDB submodule. \"
      \"Error: \${_SIMDB_INIT_ERROR}\\n\"
      \"Manually run: cd ${_SPARTA_SOURCE_DIR} && git submodule update --init --recursive\")
  endif()
  if(NOT EXISTS \"\${SIMDB_BASE}/CMakeLists.txt\")
    message(FATAL_ERROR \"SimDB submodule still not found after git submodule update. \"
      \"The repository may need to be re-cloned.\")
  endif()
endif()

# Build SimDB first (Sparta depends on it)
# Disable SIMDB_PEDANTIC to prevent -Werror from propagating to Sparta
set(SIMDB_PEDANTIC OFF CACHE BOOL \"Disable SimDB pedantic warnings\" FORCE)
add_subdirectory(\"\${SIMDB_BASE}\" \"\${CMAKE_CURRENT_BINARY_DIR}/simdb\" EXCLUDE_FROM_ALL)

# Sparta needs Boost
find_package(Boost REQUIRED COMPONENTS system filesystem timer serialization regex)

# Sparta needs yaml-cpp; use parent's forwarding target
if(NOT TARGET yaml-cpp AND TARGET deps::yaml-cpp)
  # Create bare yaml-cpp target for Sparta's internal use
  add_library(yaml-cpp INTERFACE)
  target_link_libraries(yaml-cpp INTERFACE deps::yaml-cpp)
endif()

# Get git version
execute_process(COMMAND bash \"-c\" \"git describe --tags --always\"
  WORKING_DIRECTORY \${SPARTA_BASE}
  OUTPUT_VARIABLE GIT_REPO_VERSION RESULT_VARIABLE rc)
if(NOT rc EQUAL \"0\")
  set(GIT_REPO_VERSION \"unknown\")
endif()
string(STRIP \${GIT_REPO_VERSION} GIT_REPO_VERSION)

# Compiler flags (Sparta's defaults minus -Werror)
# Remove -Werror: we don't control Sparta's code quality, and it has warnings
# on GCC 11.4 (deprecated constexpr redeclarations, std::filesystem issues)
add_definitions(-DSPARTA_VERSION=\\\"\${GIT_REPO_VERSION}\\\" -DRAPIDJSON_NOMEMBERITERATORCLASS)
add_compile_options(-fPIC -Wdeprecated -pedantic -Wextra
  -Wall -Winline -Winit-self -Wno-unused-function -Wuninitialized
  -Wno-sequence-point -Wno-inline -Wno-unknown-pragmas -Woverloaded-virtual
  -Wno-unused-parameter -Wno-missing-field-initializers)

# Add debug info flags (GEN_DEBUG_INFO=ON)
if(CMAKE_BUILD_TYPE STREQUAL \"Release\" OR CMAKE_BUILD_TYPE STREQUAL \"MinSizeRel\")
  add_compile_options(-g)
  add_link_options(-g)
endif()

# Source files from Sparta's CMakeLists.txt
# Core sources (lines 37-91)
file(GLOB_RECURSE SPARTA_SOURCES \"\${SPARTA_BASE}/src/*.cpp\")
# Exclude Python bindings
list(FILTER SPARTA_SOURCES EXCLUDE REGEX \".*python.*\")

# Remove -Werror from compiler flags to allow Sparta warnings
# Must be done at directory scope before creating the target
string(REPLACE \"-Werror\" \"\" CMAKE_CXX_FLAGS \"\${CMAKE_CXX_FLAGS}\")
string(REPLACE \"-Werror\" \"\" CMAKE_CXX_FLAGS_RELEASE \"\${CMAKE_CXX_FLAGS_RELEASE}\")
string(REPLACE \"-Werror\" \"\" CMAKE_CXX_FLAGS_DEBUG \"\${CMAKE_CXX_FLAGS_DEBUG}\")

add_library(sparta STATIC \${SPARTA_SOURCES})
target_compile_definitions(sparta PRIVATE -DSIMDB_ENABLED=1)
target_include_directories(sparta PUBLIC 
  \"\${SPARTA_BASE}\"
  \"\${SIMDB_BASE}/include\")
# Override any -Werror inherited from SimDB or parent scope
target_compile_options(sparta PRIVATE -Wno-error)
target_link_libraries(sparta PUBLIC
  Boost::system Boost::filesystem Boost::timer
  Boost::serialization Boost::regex yaml-cpp simdb)
")

  add_subdirectory("${_SPARTA_WRAPPER_DIR}" "${CMAKE_BINARY_DIR}/__sparta_build" EXCLUDE_FROM_ALL)

  if(NOT TARGET deps::sparta)
    add_library(deps::sparta ALIAS sparta)
  endif()

  message(STATUS "Sparta: built as subdirectory from ${_SPARTA_SOURCE_DIR}")
endif()
