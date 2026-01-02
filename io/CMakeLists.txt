# Create a library for IO operations
add_library(io_lib
    MatrixMarketIO.cpp
)

# Link OpenMP to the IO library
target_link_libraries(io_lib 
    OpenMP::OpenMP_CXX
    matrix_lib
)

# Include the current directory and matrix directory
target_include_directories(io_lib PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/matrix
)