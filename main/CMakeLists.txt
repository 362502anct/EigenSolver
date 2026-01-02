# Create the main executable
add_executable(eigensolver
    main.cpp
)

# Set output directory to the project root
set_target_properties(eigensolver PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}
)

# Link all necessary libraries to the main executable
target_link_libraries(eigensolver
    OpenMP::OpenMP_CXX
    ${LAPACK_LIBRARIES}
    ${BLAS_LIBRARIES}
    matrix_lib
    io_lib
    solver_lib
    util_lib
)

# Include all necessary directories
target_include_directories(eigensolver PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/matrix
    ${CMAKE_SOURCE_DIR}/io
    ${CMAKE_SOURCE_DIR}/solver
    ${CMAKE_SOURCE_DIR}/util
)