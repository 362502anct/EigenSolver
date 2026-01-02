# Create a library for solver operations
add_library(solver_lib
    EigenSolver.cpp
    QRSolver.cpp
    JacobiSolver.cpp
    PowerMethodSolver.cpp
    InversePowerMethodSolver.cpp
    DivideAndConquerSolver.cpp
)

# Link OpenMP and other libraries to the solver library
target_link_libraries(solver_lib 
    OpenMP::OpenMP_CXX
    ${LAPACK_LIBRARIES}
    ${BLAS_LIBRARIES}
    matrix_lib
    util_lib
)

# Include the current directory and other necessary directories
target_include_directories(solver_lib PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/matrix
    ${CMAKE_SOURCE_DIR}/util
)