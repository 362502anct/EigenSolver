# Create a library for utility functions
add_library(util_lib
    Utils.cpp
    SIMDDispatcher.cpp
)

# Link OpenMP to the utility library
target_link_libraries(util_lib
    OpenMP::OpenMP_CXX
)

# Include directories
target_include_directories(util_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}
)