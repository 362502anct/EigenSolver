cmake_minimum_required(VERSION 3.10)
project(EigenSolver)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Debug mode option (default: OFF)
option(ENABLE_DEBUG "Enable debug output" OFF)

# Add DEBUG definition if debug mode is enabled
if(ENABLE_DEBUG)
    add_definitions(-DDEBUG)
    message(STATUS "Debug output is ENABLED")
else()
    message(STATUS "Debug output is DISABLED")
endif()

# SIMD optimization options
option(ENABLE_AVX512 "Enable AVX-512 optimizations" OFF)
option(ENABLE_AVX2 "Enable AVX2 optimizations" ON)
option(ENABLE_AVX "Enable AVX optimizations" ON)
option(ENABLE_SSE "Enable SSE optimizations" ON)

# Determine SIMD flags
set(SIMD_FLAGS "")
set(SIMD_COMPILE_DEFINITIONS "")

if(ENABLE_AVX512)
    set(SIMD_FLAGS "-mavx512f -mavx512cd -mavx512vl")
    set(SIMD_COMPILE_DEFINITIONS "-D__AVX512F__")
    message(STATUS "SIMD: AVX-512 enabled")
elseif(ENABLE_AVX2)
    set(SIMD_FLAGS "-mavx2 -mfma")
    set(SIMD_COMPILE_DEFINITIONS "-D__AVX2__")
    message(STATUS "SIMD: AVX2 enabled")
elseif(ENABLE_AVX)
    set(SIMD_FLAGS "-mavx")
    set(SIMD_COMPILE_DEFINITIONS "-D__AVX__")
    message(STATUS "SIMD: AVX enabled")
elseif(ENABLE_SSE)
    set(SIMD_FLAGS "-msse4.2")
    set(SIMD_COMPILE_DEFINITIONS "-D__SSE2__")
    message(STATUS "SIMD: SSE enabled")
else()
    message(STATUS "SIMD: Disabled (using scalar implementations)")
endif()

message(STATUS "SIMD Flags: ${SIMD_FLAGS}")

# Find required packages
find_package(OpenMP REQUIRED)

# Try to find LAPACK and BLAS
find_package(LAPACK REQUIRED)
find_package(BLAS REQUIRED)

# Find LAPACKE (C interface to LAPACK)
find_library(LAPACKE_LIB lapacke)

# Include directories
include_directories(${CMAKE_SOURCE_DIR})

# Set compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra ${SIMD_FLAGS}")
add_definitions(${SIMD_COMPILE_DEFINITIONS})

# Add subdirectories
add_subdirectory(io)
add_subdirectory(matrix)
add_subdirectory(solver)
add_subdirectory(util)
add_subdirectory(main)
add_subdirectory(benchmark)

# Link directories (if needed for specific LAPACK/BLAS implementations)
# link_directories(/usr/local/lib)

# The main executable is defined in the main subdirectory
# This project uses subdirectory CMakeLists.txt files to organize the build
