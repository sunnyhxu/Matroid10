cmake_minimum_required(VERSION 3.16)
project(matroid10_gen_nonpaving LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Threads REQUIRED)

find_path(NAUTY_INCLUDE_DIR
  NAMES nauty.h
  PATH_SUFFIXES nauty
)

find_library(NAUTY_LIBRARY
  NAMES nauty nautyW nauty2
)

if(NOT NAUTY_INCLUDE_DIR OR NOT NAUTY_LIBRARY)
  message(FATAL_ERROR "nauty/Traces not found. Install nauty dev package in WSL (e.g. libnauty2-dev or nauty).")
endif()

add_executable(gen_nonpaving
  src/gen_nonpaving.cpp
)

target_include_directories(gen_nonpaving PRIVATE ${NAUTY_INCLUDE_DIR})
target_link_libraries(gen_nonpaving PRIVATE ${NAUTY_LIBRARY} Threads::Threads)

if(MSVC)
  target_compile_options(gen_nonpaving PRIVATE /W4)
else()
  target_compile_options(gen_nonpaving PRIVATE -Wall -Wextra -Wpedantic -O3)
endif()
