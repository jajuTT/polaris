# cmake/deps/nlohmann-json.cmake
# Resolve nlohmann/json (header-only JSON library) from system or fetch from Git.
# Provides: deps::nlohmann_json

include_guard(GLOBAL)

option(FETCH_DEPS "Allow fetching dependencies at configure time" ON)
option(USE_SYSTEM_NLOHMANN_JSON "Prefer system/toolchain-provided nlohmann_json" ON)

# Minimum version to accept from system/toolchain
# Set to empty ("") to skip version enforcement.
set(NLOHMANN_JSON_MIN_VERSION "3.10" CACHE STRING "Minimum nlohmann_json version to find")

# Tag/branch/commit when fetching from Git
# Examples:
#   -DNLOHMANN_JSON_GIT_TAG=v3.11.3  # pin to release
#   -DNLOHMANN_JSON_GIT_TAG=develop  # track branch (non-reproducible)
set(NLOHMANN_JSON_GIT_TAG "v3.11.3" CACHE STRING "Git tag/branch/commit to fetch for nlohmann_json")

# 1) Check if already available (e.g., from previous FetchContent in this build tree)
set(_NLOHMANN_JSON_AVAILABLE FALSE)
foreach(_cand nlohmann_json::nlohmann_json nlohmann_json)
  if(TARGET ${_cand})
    set(_NLOHMANN_JSON_AVAILABLE TRUE)
    message(STATUS "nlohmann_json: already configured in this build")
    break()
  endif()
endforeach()

# 2) Try system/toolchain installation (skip if already available)
if(NOT _NLOHMANN_JSON_AVAILABLE AND USE_SYSTEM_NLOHMANN_JSON)
  set(_THIRD_PARTY_DIR "${CMAKE_SOURCE_DIR}/__third_party")
  set(_NLOHMANN_JSON_SEARCH_PATHS "")

  # Collect FetchContent artifact directories to explicitly exclude
  set(_NLOHMANN_JSON_IGNORE_PATHS "")
  if(EXISTS "${_THIRD_PARTY_DIR}")
    file(GLOB _nlohmann_json_artifacts
      "${_THIRD_PARTY_DIR}/nlohmann_json-build"
      "${_THIRD_PARTY_DIR}/nlohmann_json-src"
      "${_THIRD_PARTY_DIR}/nlohmann_json-subbuild"
      "${_THIRD_PARTY_DIR}/nlohmann-json-build"
      "${_THIRD_PARTY_DIR}/nlohmann-json-src"
      "${_THIRD_PARTY_DIR}/nlohmann-json-subbuild"
    )
    foreach(_artifact IN LISTS _nlohmann_json_artifacts)
      if(IS_DIRECTORY "${_artifact}")
        list(APPEND _NLOHMANN_JSON_IGNORE_PATHS "${_artifact}")
      endif()
    endforeach()
  endif()

  # Add validated __third_party subdirectories to search path
  # Exclude FetchContent artifacts (-build, -src, -subbuild suffixes)
  if(EXISTS "${_THIRD_PARTY_DIR}")
    file(GLOB _nlohmann_json_prefixes
      "${_THIRD_PARTY_DIR}/nlohmann_json"
      "${_THIRD_PARTY_DIR}/nlohmann-json"
      "${_THIRD_PARTY_DIR}/nlohmann_json-[0-9]*"
    )
    foreach(_p IN LISTS _nlohmann_json_prefixes)
      # Skip FetchContent artifacts
      if(IS_DIRECTORY "${_p}" AND
         NOT _p MATCHES ".*-(build|src|subbuild)$")
        list(APPEND _NLOHMANN_JSON_SEARCH_PATHS
          "${_p}"
          "${_p}/lib/cmake/nlohmann_json"
          "${_p}/lib64/cmake/nlohmann_json"
          "${_p}/cmake"
        )
      endif()
    endforeach()
  endif()

  # Temporarily suppress config file errors and exclude broken paths
  set(_ORIG_CMAKE_FIND_PACKAGE_WARN_NO_MODULE ${CMAKE_FIND_PACKAGE_WARN_NO_MODULE})
  set(CMAKE_FIND_PACKAGE_WARN_NO_MODULE OFF)
  set(_ORIG_CMAKE_MESSAGE_LOG_LEVEL ${CMAKE_MESSAGE_LOG_LEVEL})
  set(CMAKE_MESSAGE_LOG_LEVEL ERROR)  # Suppress warnings from broken configs
  set(_ORIG_CMAKE_IGNORE_PATH "${CMAKE_IGNORE_PATH}")
  list(APPEND CMAKE_IGNORE_PATH ${_NLOHMANN_JSON_IGNORE_PATHS})

  # Search explicitly in our validated paths, plus system locations
  # Note: We don't use NO_DEFAULT_PATH to allow system package managers
  if(NLOHMANN_JSON_MIN_VERSION)
    find_package(nlohmann_json ${NLOHMANN_JSON_MIN_VERSION} CONFIG QUIET
      PATHS ${_NLOHMANN_JSON_SEARCH_PATHS}
      NO_DEFAULT_PATH  # Only search our explicit paths first
    )
    # If not found in explicit paths, try system defaults
    if(NOT nlohmann_json_FOUND)
      find_package(nlohmann_json ${NLOHMANN_JSON_MIN_VERSION} CONFIG QUIET)
    endif()
  else()
    find_package(nlohmann_json CONFIG QUIET
      PATHS ${_NLOHMANN_JSON_SEARCH_PATHS}
      NO_DEFAULT_PATH  # Only search our explicit paths first
    )
    # If not found in explicit paths, try system defaults
    if(NOT nlohmann_json_FOUND)
      find_package(nlohmann_json CONFIG QUIET)
    endif()
  endif()

  # Restore message settings
  set(CMAKE_FIND_PACKAGE_WARN_NO_MODULE ${_ORIG_CMAKE_FIND_PACKAGE_WARN_NO_MODULE})
  set(CMAKE_MESSAGE_LOG_LEVEL ${_ORIG_CMAKE_MESSAGE_LOG_LEVEL})
  set(CMAKE_IGNORE_PATH "${_ORIG_CMAKE_IGNORE_PATH}")

  # Verify the found package provides the expected target
  # Note: Even if nlohmann_json_FOUND is TRUE, we need to check if targets are actually available
  # (config files from incomplete builds may claim success but lack targets)
  if(nlohmann_json_FOUND AND (TARGET nlohmann_json::nlohmann_json OR TARGET nlohmann_json))
    set(_NLOHMANN_JSON_AVAILABLE TRUE)
    message(STATUS "Using system/toolchain-provided nlohmann_json")
  elseif(nlohmann_json_FOUND)
    # Package config found but targets are missing - likely a broken installation
    message(STATUS "nlohmann_json config found but targets unavailable (possibly incomplete build); will fetch from Git")
    set(nlohmann_json_FOUND FALSE)
  endif()
endif()

# 3) Fallback: fetch from Git
if(NOT _NLOHMANN_JSON_AVAILABLE)
  if(NOT FETCH_DEPS)
    message(FATAL_ERROR
      "nlohmann/json not found. Provide it via your package manager "
      "(set nlohmann_json_DIR or CMAKE_PREFIX_PATH), or enable FETCH_DEPS to fetch it.")
  endif()

  include(FetchContent)

  set(JSON_BuildTests OFF CACHE BOOL "" FORCE)
  set(JSON_Install    OFF CACHE BOOL "" FORCE)

  FetchContent_Declare(nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        ${NLOHMANN_JSON_GIT_TAG}
    # GIT_SHALLOW    TRUE
  )
  FetchContent_MakeAvailable(nlohmann_json)
  set(_NLOHMANN_JSON_AVAILABLE TRUE)
endif()

# 4) Sanity check: verify target is available
if(NOT _NLOHMANN_JSON_AVAILABLE)
  message(FATAL_ERROR "nlohmann_json should have been available but wasn't found or fetched.")
endif()

# 5) Provide stable alias: deps::nlohmann_json
set(_NLOHMANN_JSON_TARGET "")
foreach(_cand nlohmann_json::nlohmann_json nlohmann_json)
  if(TARGET ${_cand})
    set(_NLOHMANN_JSON_TARGET ${_cand})
    break()
  endif()
endforeach()

if(NOT _NLOHMANN_JSON_TARGET)
  message(FATAL_ERROR "nlohmann_json was expected to be available but no suitable CMake target was found.")
endif()

if(NOT TARGET deps::nlohmann_json)
  add_library(deps_nlohmann_json INTERFACE)
  target_link_libraries(deps_nlohmann_json INTERFACE ${_NLOHMANN_JSON_TARGET})
  add_library(deps::nlohmann_json ALIAS deps_nlohmann_json)
endif()
