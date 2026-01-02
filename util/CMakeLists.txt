# Create a library for utility functions
add_library(util_lib
    Utils.cpp
)

# Link OpenMP to the utility library
target_link_libraries(util_lib 
    OpenMP::OpenMP_CXX
    matrix_lib
)

# Include the current directory and matrix directory
target_include_directories(util_lib PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/matrix
)