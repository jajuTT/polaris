# cmake/deps/nlohmann-json.cmake
# Resolve nlohmann/json (header-only JSON library) from system or fetch from Git.
# Exposes stable target: deps::nlohmann_json

include_guard(GLOBAL)

option(FETCH_DEPS "Allow fetching dependencies at configure time" ON)

# 1) Look for a vendored copy already in __third_party/
set(_NJ_THIRD_PARTY "${CMAKE_SOURCE_DIR}/__third_party")
if(EXISTS "${_NJ_THIRD_PARTY}/nlohmann_json/CMakeLists.txt")
  add_subdirectory("${_NJ_THIRD_PARTY}/nlohmann_json" _nlohmann_json EXCLUDE_FROM_ALL)
endif()

# 2) Try a system/package-manager installation
if(NOT TARGET nlohmann_json::nlohmann_json)
  find_package(nlohmann_json QUIET)
endif()

# 3) Fetch from GitHub if still not found
if(NOT TARGET nlohmann_json::nlohmann_json)
  if(NOT FETCH_DEPS)
    message(FATAL_ERROR
      "nlohmann/json not found. Either install it (e.g. brew install nlohmann-json), "
      "place it in __third_party/nlohmann_json/, or enable FETCH_DEPS.")
  endif()

  include(FetchContent)
  FetchContent_Declare(nlohmann_json
    URL      https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
    URL_HASH SHA256=d6c65aca6b1ed68e7a182f4757257b107ae403032760ed6ef121c9d55e81757d
  )
  set(JSON_BuildTests OFF CACHE BOOL "" FORCE)
  set(JSON_Install    OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(nlohmann_json)
  message(STATUS "Fetched nlohmann/json v3.11.3 via FetchContent")
endif()

# 4) Create stable alias target
if(NOT TARGET deps::nlohmann_json)
  add_library(deps_nlohmann_json INTERFACE)
  target_link_libraries(deps_nlohmann_json INTERFACE nlohmann_json::nlohmann_json)
  add_library(deps::nlohmann_json ALIAS deps_nlohmann_json)
endif()
