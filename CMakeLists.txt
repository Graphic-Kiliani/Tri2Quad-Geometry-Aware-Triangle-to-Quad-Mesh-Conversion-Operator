cmake_minimum_required(VERSION 3.18)
project(quad_converter LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)
find_package(OpenMP QUIET)

set(lemon_dir "${CMAKE_CURRENT_BINARY_DIR}/lemon-1.3.1")
if(NOT EXISTS "${lemon_dir}/CMakeLists.txt")
  file(ARCHIVE_EXTRACT
    INPUT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/lemon-1.3.1.tar.gz"
    DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")
endif()
add_subdirectory("${lemon_dir}" lemon-build EXCLUDE_FROM_ALL)

pybind11_add_module(quad_converter
  python_bindings/module.cpp
  src/matching_lemon.cpp
  src/spiral_repair.cpp
  src/tri2quad.cpp
)
target_compile_definitions(quad_converter PRIVATE USE_LEMON)
target_include_directories(quad_converter PRIVATE
  include "${lemon_dir}" "${CMAKE_CURRENT_BINARY_DIR}/lemon-build")
target_link_libraries(quad_converter PRIVATE lemon)

if(OpenMP_CXX_FOUND)
  target_link_libraries(quad_converter PRIVATE OpenMP::OpenMP_CXX)
endif()
