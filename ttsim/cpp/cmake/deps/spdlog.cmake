# cmake/deps/spdlog.cmake
# Resolve spdlog from __third_party, system/toolchain, or fetch from Git.
# Provides: deps::spdlog

include_guard(GLOBAL)

option(FETCH_DEPS "Allow fetching dependencies at configure time" ON)
set(SPDLOG_GIT_TAG "v1.15.0" CACHE STRING "Git tag/branch/commit to fetch for spdlog")

set(_THIRD_PARTY_DIR "${CMAKE_SOURCE_DIR}/__third_party")

# 1) Try __third_party and system paths
if(EXISTS "${_THIRD_PARTY_DIR}")
  list(APPEND CMAKE_PREFIX_PATH "${_THIRD_PARTY_DIR}")
  file(GLOB _spdlog_prefixes
    "${_THIRD_PARTY_DIR}/spdlog*"
  )
  foreach(_p IN LISTS _spdlog_prefixes)
    if(IS_DIRECTORY "${_p}")
      list(APPEND CMAKE_PREFIX_PATH
        "${_p}"
        "${_p}/lib/cmake/spdlog"
        "${_p}/lib64/cmake/spdlog"
        "${_p}/cmake"
      )
    endif()
  endforeach()
endif()

find_package(spdlog CONFIG QUIET)

# 2) Fallback: fetch from Git
if(NOT spdlog_FOUND)
  if(NOT FETCH_DEPS)
    message(FATAL_ERROR
      "spdlog not found. Provide it via your package manager "
      "(set spdlog_DIR or CMAKE_PREFIX_PATH), or enable FETCH_DEPS to fetch it.")
  endif()

  include(FetchContent)

  set(SPDLOG_BUILD_EXAMPLE OFF CACHE BOOL "" FORCE)
  set(SPDLOG_BUILD_TESTS   OFF CACHE BOOL "" FORCE)
  set(SPDLOG_INSTALL       OFF CACHE BOOL "" FORCE)

  FetchContent_Declare(spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG        ${SPDLOG_GIT_TAG}
    # GIT_SHALLOW    TRUE
  )
  FetchContent_MakeAvailable(spdlog)
  message(STATUS "Fetched spdlog from Git (tag: ${SPDLOG_GIT_TAG})")
else()
  message(STATUS "Using system/toolchain-provided spdlog")
endif()

# 3) Provide stable alias: deps::spdlog
set(_SPDLOG_TARGET "")
foreach(_cand spdlog::spdlog spdlog)
  if(TARGET ${_cand})
    set(_SPDLOG_TARGET ${_cand})
    break()
  endif()
endforeach()

if(NOT _SPDLOG_TARGET)
  message(FATAL_ERROR "spdlog was expected to be available but no suitable CMake target was found.")
endif()

if(NOT TARGET deps::spdlog)
  add_library(deps_spdlog INTERFACE)
  target_link_libraries(deps_spdlog INTERFACE ${_SPDLOG_TARGET})
  add_library(deps::spdlog ALIAS deps_spdlog)
endif()
