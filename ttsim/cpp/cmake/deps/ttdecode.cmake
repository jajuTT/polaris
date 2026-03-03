# cmake/deps/ttdecode.cmake
# Resolve ttdecode from __third_party/ttdecode/ (preferred) or fetch from the
# private Tenstorrent GitLab.
#
# To use a local copy without network access:
#   Place (or symlink) the ttdecode source at:
#     polaris/ttsim/cpp/__third_party/ttdecode/
#   (i.e. the directory must contain a CMakeLists.txt)
#
# Provides: deps::ttdecode

include_guard(GLOBAL)

option(FETCH_DEPS "Allow fetching dependencies at configure time" ON)
set(TTDECODE_GIT_TAG "HEAD" CACHE STRING "Git tag/branch/commit to fetch for ttdecode")

set(_TTDECODE_PRIVATE_REPO "git@aus-gitlab.local.tenstorrent.com:arch/ttdecode.git")
set(_TTDECODE_DIR "${CMAKE_SOURCE_DIR}/__third_party/ttdecode")

# ----------------------------------------------------------------
# 1) Already configured upstream
# ----------------------------------------------------------------
if(TARGET ttdecode::ttdecode OR TARGET ttdecode)
  message(STATUS "ttdecode: already configured in parent project")
  if(NOT TARGET deps::ttdecode)
    # ttdecode::ttdecode is itself an ALIAS of ttdecode; always alias the real target.
    if(TARGET ttdecode)
      add_library(deps::ttdecode ALIAS ttdecode)
    else()
      # Defensive: ttdecode::ttdecode exists but real target name is unknown —
      # wrap it in an INTERFACE library to avoid alias-of-alias.
      add_library(deps_ttdecode INTERFACE)
      target_link_libraries(deps_ttdecode INTERFACE ttdecode::ttdecode)
      add_library(deps::ttdecode ALIAS deps_ttdecode)
    endif()
  endif()
  return()
endif()

# ----------------------------------------------------------------
# Helper: add ttdecode source dir as a subdirectory.
# Disables python bindings, whisper, and ttdecode's own tests.
# yaml-cpp: keep ENABLE_YAML ON so ttdecode/src finds deps::yaml-cpp
# (yaml-cpp.cmake already ran; include_guard prevents re-fetch).
# ----------------------------------------------------------------
macro(_neosim_add_ttdecode_subdir _src_dir)
  set(BUILD_PYTHON_BINDINGS OFF CACHE BOOL "ttdecode: no Python bindings"  FORCE)
  # ENABLE_WHISPER: let ttdecode decide (defaults to ON).
  # Whisper is a mandatory ttdecode dependency; it is fetched and built by
  # ttdecode's cmake/deps/whisper.cmake on Linux/x86-64.
  # On macOS the whisper build will fail — this is expected (macOS is
  # configure/IDE-only per CLAUDE.md); compile on Linux.
  set(ENABLE_YAML           ON  CACHE BOOL "ttdecode: yaml (already found)" FORCE)

  # Record the public include directory so src/CMakeLists.txt can add it
  # explicitly to neosim's include path.  target_link_libraries(PUBLIC)
  # propagation through EXCLUDE_FROM_ALL + cross-directory ALIAS targets is
  # unreliable across CMake versions; an explicit entry is belt-and-suspenders.
  set(NEOSIM_TTDECODE_INCLUDE_DIR "${_src_dir}/include"
      CACHE INTERNAL "ttdecode public include directory used by neosim targets")

  # Temporarily disable testing so ttdecode's test targets don't enter our
  # CTest suite. Restored immediately after add_subdirectory.
  set(_neosim_saved_bt ${BUILD_TESTING})
  set(BUILD_TESTING OFF CACHE BOOL "" FORCE)

  add_subdirectory(
    "${_src_dir}"
    "${CMAKE_BINARY_DIR}/__ttdecode_build"
    EXCLUDE_FROM_ALL
  )

  set(BUILD_TESTING ${_neosim_saved_bt} CACHE BOOL "" FORCE)
endmacro()

# ----------------------------------------------------------------
# 2) Source present in __third_party/ttdecode/
# ----------------------------------------------------------------
if(EXISTS "${_TTDECODE_DIR}/CMakeLists.txt")
  message(STATUS "ttdecode: using source at ${_TTDECODE_DIR}")
  _neosim_add_ttdecode_subdir("${_TTDECODE_DIR}")
  if(NOT TARGET deps::ttdecode)
    add_library(deps::ttdecode ALIAS ttdecode)
  endif()
  return()
endif()

# ----------------------------------------------------------------
# 3) Fetch from private GitLab using FetchContent (configure-time download)
# ----------------------------------------------------------------
if(NOT FETCH_DEPS)
  message(FATAL_ERROR
    "ttdecode not found at ${_TTDECODE_DIR}.\n"
    "Either:\n"
    "  - Copy/symlink the ttdecode source to __third_party/ttdecode/\n"
    "  - Enable FETCH_DEPS=ON to clone from ${_TTDECODE_PRIVATE_REPO}")
endif()

find_package(Git QUIET)
if(NOT GIT_FOUND)
  message(FATAL_ERROR "Git is required to fetch ttdecode but was not found.")
endif()

# Check network reachability before attempting clone
execute_process(
  COMMAND "${CMAKE_COMMAND}" -E env "GIT_TERMINAL_PROMPT=0"
          "${GIT_EXECUTABLE}" ls-remote --exit-code --heads "${_TTDECODE_PRIVATE_REPO}"
  RESULT_VARIABLE _TTDECODE_REACH_RESULT
  OUTPUT_QUIET
  ERROR_QUIET
)

if(NOT _TTDECODE_REACH_RESULT EQUAL 0)
  message(FATAL_ERROR
    "ttdecode not found at ${_TTDECODE_DIR} and "
    "${_TTDECODE_PRIVATE_REPO} is not reachable.\n"
    "Copy the ttdecode source to __third_party/ttdecode/ manually.")
endif()

message(STATUS "ttdecode: fetching from ${_TTDECODE_PRIVATE_REPO} (tag: ${TTDECODE_GIT_TAG})...")

include(FetchContent)
FetchContent_Declare(ttdecode_source
  GIT_REPOSITORY "${_TTDECODE_PRIVATE_REPO}"
  GIT_TAG        "${TTDECODE_GIT_TAG}"
  GIT_SHALLOW    TRUE
  SOURCE_DIR     "${_TTDECODE_DIR}"
)

# Populate downloads at configure time (not build time like ExternalProject)
FetchContent_Populate(ttdecode_source)

message(STATUS "ttdecode: source fetched to ${_TTDECODE_DIR}")
_neosim_add_ttdecode_subdir("${_TTDECODE_DIR}")
if(NOT TARGET deps::ttdecode)
  add_library(deps::ttdecode ALIAS ttdecode)
endif()
