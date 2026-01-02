#ifndef SIMD_DISPATCHER_H
#define SIMD_DISPATCHER_H

#include "SIMDTraits.h"
#include <functional>
#include <cstdint>

/**
 * @file SIMDDispatcher.h
 * @brief Runtime SIMD detection and function dispatching system
 *
 * Provides automatic CPU feature detection and function pointer dispatching
 * for multiple SIMD instruction sets (SSE/AVX/AVX2/AVX-512)
 */

// Forward declarations for SIMD implementation classes
namespace MatrixOps_SIMD {
    // Scalar multiply functions
    void scalar_multiply_scalar(double* data, int n, double scalar);
    void scalar_multiply_sse(double* data, int n, double scalar);
    void scalar_multiply_avx(double* data, int n, double scalar);
    void scalar_multiply_avx2(double* data, int n, double scalar);
    void scalar_multiply_avx512(double* data, int n, double scalar);

    // Vector add functions
    void vector_add_scalar(double* dst, const double* src, int n);
    void vector_add_sse(double* dst, const double* src, int n);
    void vector_add_avx(double* dst, const double* src, int n);
    void vector_add_avx2(double* dst, const double* src, int n);
    void vector_add_avx512(double* dst, const double* src, int n);

    // Vector subtract functions
    void vector_subtract_scalar(double* dst, const double* src, int n);
    void vector_subtract_sse(double* dst, const double* src, int n);
    void vector_subtract_avx(double* dst, const double* src, int n);
    void vector_subtract_avx2(double* dst, const double* src, int n);
    void vector_subtract_avx512(double* dst, const double* src, int n);

    // Dense matrix multiply functions
    void matrix_multiply_scalar(double* C, const double* A, const double* B,
                                 int m, int k, int n);
    void matrix_multiply_sse(double* C, const double* A, const double* B,
                              int m, int k, int n);
    void matrix_multiply_avx(double* C, const double* A, const double* B,
                              int m, int k, int n);
    void matrix_multiply_avx2(double* C, const double* A, const double* B,
                               int m, int k, int n);
    void matrix_multiply_avx512(double* C, const double* A, const double* B,
                                 int m, int k, int n);

    // Sparse dot product functions
    double sparse_dot_product_scalar(const double* result_values, const int* result_indices,
                                      int result_nnz, const double* other_values,
                                      const int* other_indices, int other_nnz,
                                      double tolerance = 1e-12);
    double sparse_dot_product_sse(const double* result_values, const int* result_indices,
                                   int result_nnz, const double* other_values,
                                   const int* other_indices, int other_nnz,
                                   double tolerance = 1e-12);
    double sparse_dot_product_avx(const double* result_values, const int* result_indices,
                                   int result_nnz, const double* other_values,
                                   const int* other_indices, int other_nnz,
                                   double tolerance = 1e-12);
    double sparse_dot_product_avx2(const double* result_values, const int* result_indices,
                                    int result_nnz, const double* other_values,
                                    const int* other_indices, int other_nnz,
                                    double tolerance = 1e-12);
    double sparse_dot_product_avx512(const double* result_values, const int* result_indices,
                                     int result_nnz, const double* other_values,
                                     const int* other_indices, int other_nnz,
                                     double tolerance = 1e-12);
}

// Function pointer type definitions
using ScalarMultiplyFunc = void(*)(double* data, int n, double scalar);
using VectorAddFunc = void(*)(double* dst, const double* src, int n);
using VectorSubtractFunc = void(*)(double* dst, const double* src, int n);
using MatrixMultiplyFunc = void(*)(double* C, const double* A, const double* B,
                                    int m, int k, int n);
using SparseDotProductFunc = double(*)(const double* result_values, const int* result_indices,
                                        int result_nnz, const double* other_values,
                                        const int* other_indices, int other_nnz,
                                        double tolerance);

/**
 * @class SIMDDispatcher
 * @brief Singleton class for SIMD runtime detection and function dispatching
 *
 * This class:
 * - Detects CPU SIMD capabilities at runtime using CPUID
 * - Selects the optimal SIMD implementation
 * - Provides function pointers to the selected implementations
 * - Falls back gracefully to scalar versions on unsupported hardware
 */
class SIMDDispatcher {
public:
    /**
     * @brief Get singleton instance
     *
     * @return Reference to the singleton instance
     */
    static SIMDDispatcher& getInstance();

    /**
     * @brief Get current SIMD level
     *
     * @return Detected SIMD level
     */
    SIMDLevel getSIMDLevel() const { return current_level; }

    /**
     * @brief Get SIMD level name as string
     *
     * @return Human-readable SIMD level name
     */
    const char* getSIMDLevelName() const;

    /**
     * @brief Print CPU SIMD information to stdout
     */
    void printCPUInfo();  // Not const - calls non-const check methods

    /**
     * @brief Get scalar multiply function pointer
     *
     * @return Function pointer to the selected SIMD implementation
     */
    ScalarMultiplyFunc getScalarMultiplyFunc() const {
        return scalar_multiply_impl;
    }

    /**
     * @brief Get vector add function pointer
     *
     * @return Function pointer to the selected SIMD implementation
     */
    VectorAddFunc getVectorAddFunc() const {
        return vector_add_impl;
    }

    /**
     * @brief Get vector subtract function pointer
     *
     * @return Function pointer to the selected SIMD implementation
     */
    VectorSubtractFunc getVectorSubtractFunc() const {
        return vector_subtract_impl;
    }

    /**
     * @brief Get dense matrix multiply function pointer
     *
     * @return Function pointer to the selected SIMD implementation
     */
    MatrixMultiplyFunc getMatrixMultiplyFunc() const {
        return matrix_multiply_impl;
    }

    /**
     * @brief Get sparse dot product function pointer
     *
     * @return Function pointer to the selected SIMD implementation
     */
    SparseDotProductFunc getSparseDotProductFunc() const {
        return sparse_dot_product_impl;
    }

    /**
     * @brief Manually set SIMD level (for testing)
     *
     * @param level Desired SIMD level
     *
     * @note This should only be used for testing purposes
     */
    void setSIMDLevel(SIMDLevel level);

private:
    /**
     * @brief Private constructor (singleton pattern)
     *
     * Detects CPU capabilities and initializes function pointers
     */
    SIMDDispatcher();

    /**
     * @brief Destructor
     */
    ~SIMDDispatcher() = default;

    /**
     * @brief Delete copy constructor
     */
    SIMDDispatcher(const SIMDDispatcher&) = delete;

    /**
     * @brief Delete copy assignment operator
     */
    SIMDDispatcher& operator=(const SIMDDispatcher&) = delete;

    /**
     * @brief Detect supported SIMD level
     *
     * Checks CPUID for SIMD support and returns the highest available level
     *
     * @return Detected SIMD level
     */
    SIMDLevel detectSIMDSupport();

    /**
     * @brief Check for SSE support
     *
     * @return true if CPU supports SSE2
     */
    bool checkSSE();

    /**
     * @brief Check for AVX support
     *
     * @return true if CPU supports AVX
     */
    bool checkAVX();

    /**
     * @brief Check for AVX2 support
     *
     * @return true if CPU supports AVX2
     */
    bool checkAVX2();

    /**
     * @brief Check for AVX-512 support
     *
     * @return true if CPU supports AVX-512F
     */
    bool checkAVX512();

    /**
     * @brief Initialize function pointers based on detected SIMD level
     */
    void initializeFunctionPointers();

    // Current SIMD level
    SIMDLevel current_level;

    // Function pointer storage
    ScalarMultiplyFunc scalar_multiply_impl;
    VectorAddFunc vector_add_impl;
    VectorSubtractFunc vector_subtract_impl;
    MatrixMultiplyFunc matrix_multiply_impl;
    SparseDotProductFunc sparse_dot_product_impl;
};

/**
 * @brief Convenience macro to get SIMD dispatcher instance
 */
#define SIMD_DISPATCHER SIMDDispatcher::getInstance()

#endif // SIMD_DISPATCHER_H
