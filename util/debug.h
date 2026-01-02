#ifndef DEBUG_H
#define DEBUG_H

#include <iostream>

// Debug模式宏定义
// 使用方法：
// 1. 在编译时添加 -DDEBUG 开启debug模式
// 2. 或在CMakeLists.txt中添加 add_definitions(-DDEBUG)
// 3. 或在代码中 #define DEBUG (不推荐)

#ifdef DEBUG

    // Debug输出宏 - 输出到标准错误流以便与正常输出区分
    #define DEBUG_PRINT(msg) std::cerr << "[DEBUG] " << msg << std::endl
    #define DEBUG_PRINT_VAR(var) std::cerr << "[DEBUG] " << #var << " = " << (var) << std::endl
    #define DEBUG_ERROR(msg) std::cerr << "[ERROR] " << msg << std::endl

    // Debug信息输出函数
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

    // Release模式下，所有debug输出将被编译器优化掉
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
