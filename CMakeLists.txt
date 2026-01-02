cmake_minimum_required(VERSION 3.10)
project(EigenSolver)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra")

# Add subdirectories
add_subdirectory(io)
add_subdirectory(matrix)
add_subdirectory(solver)
add_subdirectory(util)
add_subdirectory(main)

# Link directories (if needed for specific LAPACK/BLAS implementations)
# link_directories(/usr/local/lib)

# The main executable is defined in the main subdirectory
# This project uses subdirectory CMakeLists.txt files to organize the build
