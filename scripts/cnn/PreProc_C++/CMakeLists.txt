cmake_minimum_required(VERSION 3.13)
project(emg_preprocessing CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_compile_options(-Wall -Wextra -pedantic-errors -ffp-contract=off)

add_library(emg_preprocessing STATIC
    src/biquad.cpp
    src/filter_design.cpp
    src/quantizer.cpp
    src/window_buffer.cpp
    src/emg_preprocessor.cpp)
target_include_directories(emg_preprocessing PUBLIC src)