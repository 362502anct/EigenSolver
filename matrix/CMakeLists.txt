# Create a library for matrix operations
add_library(matrix_lib
    Matrix.cpp
)

# Link OpenMP to the matrix library
target_link_libraries(matrix_lib
    OpenMP::OpenMP_CXX
    ${LAPACKE_LIB}
    ${LAPACK_LIBRARIES}
    ${BLAS_LIBRARIES}
)

# Include the current directory
target_include_directories(matrix_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})