cmake_minimum_required(VERSION 3.20)
project(neosim LANGUAGES CXX)

if(NOT DEFINED CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 20)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
elseif(CMAKE_CXX_STANDARD LESS 20)
  message(FATAL_ERROR "neosim requires C++20 or newer.")
endif()

# Centralize all fetched/vendored dependencies under __third_party
set(FETCHCONTENT_BASE_DIR "${PROJECT_SOURCE_DIR}/__third_party")
if(EXISTS "${PROJECT_SOURCE_DIR}/__third_party")
  list(APPEND CMAKE_PREFIX_PATH "${PROJECT_SOURCE_DIR}/__third_party")
endif()

include(CTest)

# ----------------------------------------------------------------
# Dependencies (all mandatory)
# Resolution: __third_party/ → system → network fetch
# ----------------------------------------------------------------

# yaml-cpp first: both ttdecode and sparta depend on it
include("${PROJECT_SOURCE_DIR}/cmake/deps/yaml-cpp.cmake")
include("${PROJECT_SOURCE_DIR}/cmake/deps/spdlog.cmake")
include("${PROJECT_SOURCE_DIR}/cmake/deps/ttdecode.cmake")
include("${PROJECT_SOURCE_DIR}/cmake/deps/sparta.cmake")

if(BUILD_TESTING)
  include("${PROJECT_SOURCE_DIR}/cmake/deps/googletest.cmake")
endif()

# ----------------------------------------------------------------
# Project layout
# ----------------------------------------------------------------
add_subdirectory(src)

if(BUILD_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()
