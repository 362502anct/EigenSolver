#ifndef DEBUG_H
#define DEBUG_H

#include <iostream>

// Debug mode macro definitions
// Usage:
// 1. Add -DDEBUG during compilation to enable debug mode
// 2. Or add add_definitions(-DDEBUG) in CMakeLists.txt
// 3. Or #define DEBUG in code (not recommended)

#ifdef DEBUG

    // Debug output macros - output to stderr to distinguish from normal output
    #define DEBUG_PRINT(msg) std::cerr << "[DEBUG] " << msg << std::endl
    #define DEBUG_PRINT_VAR(var) std::cerr << "[DEBUG] " << #var << " = " << (var) << std::endl
    #define DEBUG_ERROR(msg) std::cerr << "[ERROR] " << msg << std::endl

    // Debug output functions
    namespace Debug {
        inline void print(const std::string& msg) {
            std::cerr << "[DEBUG] " << msg << std::endl;
        }

        inline void error(const std::string& msg) {
            std::cerr << "[ERROR] " << msg << std::endl;
        }

        template<typename T>
        inline void printVar(const std::string& name, const T& value) {
            std::cerr << "[DEBUG] " << name << " = " << value << std::endl;
        }
    }

#else

    // Release mode: all debug output will be optimized out by compiler
    #define DEBUG_PRINT(msg) ((void)0)
    #define DEBUG_PRINT_VAR(var) ((void)0)
    #define DEBUG_ERROR(msg) ((void)0)

    namespace Debug {
        inline void print(const std::string&) {}
        inline void error(const std::string&) {}

        template<typename T>
        inline void printVar(const std::string&, const T&) {}
    }

#endif // DEBUG

#endif // DEBUG_H
