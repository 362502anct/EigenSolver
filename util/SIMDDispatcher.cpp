#include "SIMDDispatcher.h"
#include "../matrix/MatrixOps_SIMD.h"
#include <iostream>
#include <cstring>

#ifdef _WIN32
#include <intrin.h>
#else
// For CPUID on Linux/macOS
#if defined(__x86_64__) || defined(__i386__)
#include <cpuid.h>
#endif
#endif

/**
 * @brief Get singleton instance
 *
 * @return Reference to the singleton instance
 */
SIMDDispatcher& SIMDDispatcher::getInstance() {
    static SIMDDispatcher instance;
    return instance;
}

/**
 * @brief Constructor - detects CPU and initializes function pointers
 */
SIMDDispatcher::SIMDDispatcher() {
    current_level = detectSIMDSupport();
    initializeFunctionPointers();

    std::cout << "SIMD Level: " << getSIMDLevelName() << std::endl;
}

/**
 * @brief Detect the highest supported SIMD level
 *
 * Detection priority: AVX512 → AVX2 → AVX → SSE → SCALAR
 *
 * @return Detected SIMD level
 */
SIMDLevel SIMDDispatcher::detectSIMDSupport() {
    if (checkAVX512()) return SIMDLevel::AVX512;
    if (checkAVX2()) return SIMDLevel::AVX2;
    if (checkAVX()) return SIMDLevel::AVX;
    if (checkSSE()) return SIMDLevel::SSE;
    return SIMDLevel::SCALAR;
}

/**
 * @brief Check for SSE2 support
 *
 * @return true if CPU supports SSE2
 */
bool SIMDDispatcher::checkSSE() {
#if defined(__x86_64__)
    // x86_64 always supports SSE2
    return true;
#elif defined(__i386__) || defined(_M_IX86)
    // Check for SSE2 on x86
    #ifdef _WIN32
        int cpuInfo[4] = {0};
        __cpuid(cpuInfo, 1);
        return (cpuInfo[3] & (1 << 26)) != 0;  // SSE2 bit
    #else
        unsigned int eax, ebx, ecx, edx;
        __get_cpuid(1, &eax, &ebx, &ecx, &edx);
        return (edx & bit_SSE2) != 0;
    #endif
#else
    // Non-x86 platforms
    return false;
#endif
}

/**
 * @brief Check for AVX support
 *
 * @return true if CPU supports AVX
 */
bool SIMDDispatcher::checkAVX() {
#if defined(__AVX__)
    // Compile-time support
    return true;
#elif defined(__x86_64__) || defined(__i386__)
    #ifdef _WIN32
        int cpuInfo[4] = {0};
        __cpuid(cpuInfo, 1);
        return (cpuInfo[2] & (1 << 28)) != 0;  // AVX bit
    #else
        unsigned int eax, ebx, ecx, edx;
        __get_cpuid(1, &eax, &ebx, &ecx, &edx);
        return (ecx & bit_AVX) != 0;
    #endif
#else
    return false;
#endif
}

/**
 * @brief Check for AVX2 support
 *
 * @return true if CPU supports AVX2
 */
bool SIMDDispatcher::checkAVX2() {
#if defined(__AVX2__)
    // Compile-time support
    return true;
#elif defined(__x86_64__) || defined(__i386__)
    #ifdef _WIN32
        int cpuInfo[4] = {0};
        __cpuidex(cpuInfo, 7, 0);
        return (cpuInfo[1] & (1 << 5)) != 0;  // AVX2 bit
    #else
        unsigned int eax, ebx, ecx, edx;
        __get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx);
        return (ebx & bit_AVX2) != 0;
    #endif
#else
    return false;
#endif
}

/**
 * @brief Check for AVX-512 Foundation support
 *
 * @return true if CPU supports AVX-512F
 */
bool SIMDDispatcher::checkAVX512() {
#if defined(__AVX512F__)
    // Compile-time support
    return true;
#elif defined(__x86_64__) || defined(__i386__)
    #ifdef _WIN32
        int cpuInfo[4] = {0};
        __cpuidex(cpuInfo, 7, 0);
        return (cpuInfo[1] & (1 << 16)) != 0;  // AVX512F bit
    #else
        unsigned int eax, ebx, ecx, edx;
        __get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx);
        return (ebx & bit_AVX512F) != 0;
    #endif
#else
    return false;
#endif
}

/**
 * @brief Initialize function pointers based on detected SIMD level
 *
 * Selects the optimal implementation for each operation
 */
void SIMDDispatcher::initializeFunctionPointers() {
    switch (current_level) {
#ifdef __AVX512F__
        case SIMDLevel::AVX512:
            scalar_multiply_impl = MatrixOps_SIMD::scalar_multiply_avx512;
            vector_add_impl = MatrixOps_SIMD::vector_add_avx512;
            vector_subtract_impl = MatrixOps_SIMD::vector_subtract_avx512;
            matrix_multiply_impl = MatrixOps_SIMD::matrix_multiply_avx512;
            sparse_dot_product_impl = MatrixOps_SIMD::sparse_dot_product_avx512;
            break;
#endif
#ifdef __AVX2__
        case SIMDLevel::AVX2:
            scalar_multiply_impl = MatrixOps_SIMD::scalar_multiply_avx2;
            vector_add_impl = MatrixOps_SIMD::vector_add_avx2;
            vector_subtract_impl = MatrixOps_SIMD::vector_subtract_avx2;
            matrix_multiply_impl = MatrixOps_SIMD::matrix_multiply_avx2;
            sparse_dot_product_impl = MatrixOps_SIMD::sparse_dot_product_avx2;
            break;
#endif
#ifdef __AVX__
        case SIMDLevel::AVX:
            scalar_multiply_impl = MatrixOps_SIMD::scalar_multiply_avx;
            vector_add_impl = MatrixOps_SIMD::vector_add_avx;
            vector_subtract_impl = MatrixOps_SIMD::vector_subtract_avx;
            matrix_multiply_impl = MatrixOps_SIMD::matrix_multiply_avx;
            sparse_dot_product_impl = MatrixOps_SIMD::sparse_dot_product_avx;
            break;
#endif
        case SIMDLevel::SSE:
            scalar_multiply_impl = MatrixOps_SIMD::scalar_multiply_sse;
            vector_add_impl = MatrixOps_SIMD::vector_add_sse;
            vector_subtract_impl = MatrixOps_SIMD::vector_subtract_sse;
            matrix_multiply_impl = MatrixOps_SIMD::matrix_multiply_sse;
            sparse_dot_product_impl = MatrixOps_SIMD::sparse_dot_product_sse;
            break;
        default:  // SCALAR
            scalar_multiply_impl = MatrixOps_SIMD::scalar_multiply_scalar;
            vector_add_impl = MatrixOps_SIMD::vector_add_scalar;
            vector_subtract_impl = MatrixOps_SIMD::vector_subtract_scalar;
            matrix_multiply_impl = MatrixOps_SIMD::matrix_multiply_scalar;
            sparse_dot_product_impl = MatrixOps_SIMD::sparse_dot_product_scalar;
    }
}

/**
 * @brief Get SIMD level name as string
 *
 * @return Human-readable SIMD level name
 */
const char* SIMDDispatcher::getSIMDLevelName() const {
    switch (current_level) {
        case SIMDLevel::AVX512: return "AVX-512";
        case SIMDLevel::AVX2:   return "AVX2";
        case SIMDLevel::AVX:    return "AVX";
        case SIMDLevel::SSE:    return "SSE";
        case SIMDLevel::SCALAR: return "Scalar";
        default:                return "Unknown";
    }
}

/**
 * @brief Print CPU SIMD information
 */
void SIMDDispatcher::printCPUInfo() {
    std::cout << "=== SIMD Information ===" << std::endl;
    std::cout << "Detected SIMD Level: " << getSIMDLevelName() << std::endl;
    std::cout << "Vector Width: " << getSIMDWidth(current_level) << " doubles" << std::endl;
    std::cout << "Alignment: " << getSIMDAlignment(current_level) << " bytes" << std::endl;

    // Print available features
    std::cout << "\nAvailable Features:" << std::endl;
    std::cout << "  SSE:    " << (checkSSE() ? "Yes" : "No") << std::endl;
    std::cout << "  AVX:    " << (checkAVX() ? "Yes" : "No") << std::endl;
    std::cout << "  AVX2:   " << (checkAVX2() ? "Yes" : "No") << std::endl;
    std::cout << "  AVX512: " << (checkAVX512() ? "Yes" : "No") << std::endl;
}

/**
 * @brief Manually set SIMD level (for testing)
 *
 * @param level Desired SIMD level
 *
 * @note This should only be used for testing purposes
 */
void SIMDDispatcher::setSIMDLevel(SIMDLevel level) {
    current_level = level;
    initializeFunctionPointers();
    std::cout << "SIMD Level manually set to: " << getSIMDLevelName() << std::endl;
}
