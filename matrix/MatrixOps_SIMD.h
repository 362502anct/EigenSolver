#ifndef MATRIX_OPS_SIMD_H
#define MATRIX_OPS_SIMD_H

/**
 * @file MatrixOps_SIMD.h
 * @brief SIMD-optimized matrix operations
 *
 * Provides multiple implementations of matrix operations using different
 * SIMD instruction sets: Scalar, SSE, AVX, AVX2, and AVX-512.
 *
 * The appropriate implementation is selected at runtime by SIMDDispatcher.
 */

#include <cstring>
#include <cmath>

// SIMD intrinsics headers
#if defined(__AVX512F__)
#include <immintrin.h>
#elif defined(__AVX2__)
#include <immintrin.h>
#elif defined(__AVX__)
#include <immintrin.h>
#elif defined(__SSE2__)
#include <emmintrin.h>
#endif

/**
 * @namespace MatrixOps_SIMD
 * @brief SIMD-optimized matrix operation implementations
 */
namespace MatrixOps_SIMD {

// ============================================================================
// Scalar Multiply Operations
// ============================================================================

/**
 * @brief Scalar multiplication - reference scalar implementation
 *
 * @param data Input/output array
 * @param n Array length
 * @param scalar Scalar multiplier
 */
inline void scalar_multiply_scalar(double* data, int n, double scalar) {
    for (int i = 0; i < n; ++i) {
        data[i] *= scalar;
    }
}

#if defined(__SSE2__)
/**
 * @brief Scalar multiplication - SSE implementation (128-bit, 2 doubles)
 *
 * @param data Input/output array
 * @param n Array length
 * @param scalar Scalar multiplier
 */
inline void scalar_multiply_sse(double* data, int n, double scalar) {
    __m128d scalar_vec = _mm_set1_pd(scalar);
    int i = 0;

    // Main loop: process 2 doubles at a time
    for (; i + 2 <= n; i += 2) {
        __m128d vec = _mm_loadu_pd(&data[i]);
        vec = _mm_mul_pd(vec, scalar_vec);
        _mm_storeu_pd(&data[i], vec);
    }

    // Handle remaining elements
    for (; i < n; ++i) {
        data[i] *= scalar;
    }
}
#endif

#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
/**
 * @brief Scalar multiplication - AVX implementation (256-bit, 4 doubles)
 *
 * @param data Input/output array
 * @param n Array length
 * @param scalar Scalar multiplier
 */
inline void scalar_multiply_avx(double* data, int n, double scalar) {
    __m256d scalar_vec = _mm256_set1_pd(scalar);
    int i = 0;

    // Main loop: process 4 doubles at a time
    for (; i + 4 <= n; i += 4) {
        __m256d vec = _mm256_loadu_pd(&data[i]);
        vec = _mm256_mul_pd(vec, scalar_vec);
        _mm256_storeu_pd(&data[i], vec);
    }

    // Handle remaining elements
    for (; i < n; ++i) {
        data[i] *= scalar;
    }
}
#endif

#if defined(__AVX2__) || defined(__AVX512F__)
/**
 * @brief Scalar multiplication - AVX2 implementation (256-bit + FMA)
 *
 * Same as AVX for scalar multiplication, but included for completeness
 *
 * @param data Input/output array
 * @param n Array length
 * @param scalar Scalar multiplier
 */
inline void scalar_multiply_avx2(double* data, int n, double scalar) {
    // AVX2 same as AVX for scalar mul
    scalar_multiply_avx(data, n, scalar);
}
#endif

#if defined(__AVX512F__)
/**
 * @brief Scalar multiplication - AVX-512 implementation (512-bit, 8 doubles)
 *
 * @param data Input/output array
 * @param n Array length
 * @param scalar Scalar multiplier
 */
inline void scalar_multiply_avx512(double* data, int n, double scalar) {
    __m512d scalar_vec = _mm512_set1_pd(scalar);
    int i = 0;

    // Main loop: process 8 doubles at a time
    for (; i + 8 <= n; i += 8) {
        __m512d vec = _mm512_loadu_pd(&data[i]);
        vec = _mm512_mul_pd(vec, scalar_vec);
        _mm512_storeu_pd(&data[i], vec);
    }

    // Handle remaining elements
    for (; i < n; ++i) {
        data[i] *= scalar;
    }
}
#endif

// ============================================================================
// Vector Add Operations
// ============================================================================

/**
 * @brief Vector addition - reference scalar implementation
 *
 * @param dst Destination array (also first operand)
 * @param src Source array (second operand)
 * @param n Array length
 */
inline void vector_add_scalar(double* dst, const double* src, int n) {
    for (int i = 0; i < n; ++i) {
        dst[i] += src[i];
    }
}

#if defined(__SSE2__)
/**
 * @brief Vector addition - SSE implementation (128-bit, 2 doubles)
 *
 * @param dst Destination array (also first operand)
 * @param src Source array (second operand)
 * @param n Array length
 */
inline void vector_add_sse(double* dst, const double* src, int n) {
    int i = 0;

    // Main loop: process 2 doubles at a time
    for (; i + 2 <= n; i += 2) {
        __m128d a = _mm_loadu_pd(&dst[i]);
        __m128d b = _mm_loadu_pd(&src[i]);
        __m128d sum = _mm_add_pd(a, b);
        _mm_storeu_pd(&dst[i], sum);
    }

    // Handle remaining elements
    for (; i < n; ++i) {
        dst[i] += src[i];
    }
}
#endif

#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
/**
 * @brief Vector addition - AVX implementation (256-bit, 4 doubles)
 *
 * @param dst Destination array (also first operand)
 * @param src Source array (second operand)
 * @param n Array length
 */
inline void vector_add_avx(double* dst, const double* src, int n) {
    int i = 0;

    // Main loop: process 4 doubles at a time
    for (; i + 4 <= n; i += 4) {
        __m256d a = _mm256_loadu_pd(&dst[i]);
        __m256d b = _mm256_loadu_pd(&src[i]);
        __m256d sum = _mm256_add_pd(a, b);
        _mm256_storeu_pd(&dst[i], sum);
    }

    // Handle remaining elements
    for (; i < n; ++i) {
        dst[i] += src[i];
    }
}
#endif

#if defined(__AVX2__) || defined(__AVX512F__)
/**
 * @brief Vector addition - AVX2 implementation (same as AVX)
 *
 * @param dst Destination array (also first operand)
 * @param src Source array (second operand)
 * @param n Array length
 */
inline void vector_add_avx2(double* dst, const double* src, int n) {
    vector_add_avx(dst, src, n);
}
#endif

#if defined(__AVX512F__)
/**
 * @brief Vector addition - AVX-512 implementation (512-bit, 8 doubles)
 *
 * @param dst Destination array (also first operand)
 * @param src Source array (second operand)
 * @param n Array length
 */
inline void vector_add_avx512(double* dst, const double* src, int n) {
    int i = 0;

    // Main loop: process 8 doubles at a time
    for (; i + 8 <= n; i += 8) {
        __m512d a = _mm512_loadu_pd(&dst[i]);
        __m512d b = _mm512_loadu_pd(&src[i]);
        __m512d sum = _mm512_add_pd(a, b);
        _mm512_storeu_pd(&dst[i], sum);
    }

    // Handle remaining elements
    for (; i < n; ++i) {
        dst[i] += src[i];
    }
}
#endif

// ============================================================================
// Vector Subtract Operations
// ============================================================================

/**
 * @brief Vector subtraction - reference scalar implementation
 *
 * @param dst Destination array (also first operand)
 * @param src Source array (second operand)
 * @param n Array length
 */
inline void vector_subtract_scalar(double* dst, const double* src, int n) {
    for (int i = 0; i < n; ++i) {
        dst[i] -= src[i];
    }
}

#if defined(__SSE2__)
/**
 * @brief Vector subtraction - SSE implementation (128-bit, 2 doubles)
 *
 * @param dst Destination array (also first operand)
 * @param src Source array (second operand)
 * @param n Array length
 */
inline void vector_subtract_sse(double* dst, const double* src, int n) {
    int i = 0;

    // Main loop: process 2 doubles at a time
    for (; i + 2 <= n; i += 2) {
        __m128d a = _mm_loadu_pd(&dst[i]);
        __m128d b = _mm_loadu_pd(&src[i]);
        __m128d diff = _mm_sub_pd(a, b);
        _mm_storeu_pd(&dst[i], diff);
    }

    // Handle remaining elements
    for (; i < n; ++i) {
        dst[i] -= src[i];
    }
}
#endif

#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
/**
 * @brief Vector subtraction - AVX implementation (256-bit, 4 doubles)
 *
 * @param dst Destination array (also first operand)
 * @param src Source array (second operand)
 * @param n Array length
 */
inline void vector_subtract_avx(double* dst, const double* src, int n) {
    int i = 0;

    // Main loop: process 4 doubles at a time
    for (; i + 4 <= n; i += 4) {
        __m256d a = _mm256_loadu_pd(&dst[i]);
        __m256d b = _mm256_loadu_pd(&src[i]);
        __m256d diff = _mm256_sub_pd(a, b);
        _mm256_storeu_pd(&dst[i], diff);
    }

    // Handle remaining elements
    for (; i < n; ++i) {
        dst[i] -= src[i];
    }
}
#endif

#if defined(__AVX2__) || defined(__AVX512F__)
/**
 * @brief Vector subtraction - AVX2 implementation (same as AVX)
 *
 * @param dst Destination array (also first operand)
 * @param src Source array (second operand)
 * @param n Array length
 */
inline void vector_subtract_avx2(double* dst, const double* src, int n) {
    vector_subtract_avx(dst, src, n);
}
#endif

#if defined(__AVX512F__)
/**
 * @brief Vector subtraction - AVX-512 implementation (512-bit, 8 doubles)
 *
 * @param dst Destination array (also first operand)
 * @param src Source array (second operand)
 * @param n Array length
 */
inline void vector_subtract_avx512(double* dst, const double* src, int n) {
    int i = 0;

    // Main loop: process 8 doubles at a time
    for (; i + 8 <= n; i += 8) {
        __m512d a = _mm512_loadu_pd(&dst[i]);
        __m512d b = _mm512_loadu_pd(&src[i]);
        __m512d diff = _mm512_sub_pd(a, b);
        _mm512_storeu_pd(&dst[i], diff);
    }

    // Handle remaining elements
    for (; i < n; ++i) {
        dst[i] -= src[i];
    }
}
#endif

// ============================================================================
// Dense Matrix Multiply Operations
// ============================================================================

/**
 * @brief Dense matrix multiplication - reference scalar implementation
 *
 * C = A * B where A is m×k, B is k×n, C is m×n
 * Uses column-major storage: C[j*m + i] = C(i,j)
 *
 * @param C Result matrix (m×n)
 * @param A Left matrix (m×k)
 * @param B Right matrix (k×n)
 * @param m Number of rows in A and C
 * @param k Number of columns in A and rows in B
 * @param n Number of columns in B and C
 */
inline void matrix_multiply_scalar(double* C, const double* A,
                                   const double* B, int m, int k, int n) {
    // Initialize C to zero
    std::memset(C, 0, m * n * sizeof(double));

    // Standard matrix multiplication
    for (int i = 0; i < m; ++i) {
        for (int l = 0; l < k; ++l) {
            double a_il = A[l * m + i];  // A(i,l) in column-major

            for (int j = 0; j < n; ++j) {
                C[j * m + i] += a_il * B[j * k + l];  // B(l,j) in column-major
            }
        }
    }
}

#if defined(__SSE2__)
/**
 * @brief Dense matrix multiplication - SSE implementation (128-bit, 2 doubles)
 *
 * Uses vectorization on the inner loop for better performance
 *
 * @param C Result matrix (m×n)
 * @param A Left matrix (m×k)
 * @param B Right matrix (k×n)
 * @param m Number of rows in A and C
 * @param k Number of columns in A and rows in B
 * @param n Number of columns in B and C
 */
inline void matrix_multiply_sse(double* C, const double* A,
                                const double* B, int m, int k, int n) {
    // Initialize C to zero
    std::memset(C, 0, m * n * sizeof(double));

    for (int i = 0; i < m; ++i) {
        for (int l = 0; l < k; ++l) {
            double a_il = A[l * m + i];

            int j = 0;
            // Vectorized loop: process 2 elements at a time
            for (; j + 2 <= n; j += 2) {
                __m128d c_vec = _mm_loadu_pd(&C[j * m + i]);
                __m128d b_vec = _mm_loadu_pd(&B[j * k + l]);
                __m128d a_vec = _mm_set1_pd(a_il);
                __m128d prod = _mm_mul_pd(a_vec, b_vec);
                c_vec = _mm_add_pd(c_vec, prod);
                _mm_storeu_pd(&C[j * m + i], c_vec);
            }

            // Handle remaining elements
            for (; j < n; ++j) {
                C[j * m + i] += a_il * B[j * k + l];
            }
        }
    }
}
#endif

#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
/**
 * @brief Dense matrix multiplication - AVX implementation (256-bit, 4 doubles)
 *
 * Uses vectorization on the inner loop for better performance
 *
 * @param C Result matrix (m×n)
 * @param A Left matrix (m×k)
 * @param B Right matrix (k×n)
 * @param m Number of rows in A and C
 * @param k Number of columns in A and rows in B
 * @param n Number of columns in B and C
 */
inline void matrix_multiply_avx(double* C, const double* A,
                                const double* B, int m, int k, int n) {
    // Initialize C to zero
    std::memset(C, 0, m * n * sizeof(double));

    for (int i = 0; i < m; ++i) {
        for (int l = 0; l < k; ++l) {
            double a_il = A[l * m + i];

            int j = 0;
            // Vectorized loop: process 4 elements at a time
            for (; j + 4 <= n; j += 4) {
                __m256d c_vec = _mm256_loadu_pd(&C[j * m + i]);
                __m256d b_vec = _mm256_loadu_pd(&B[j * k + l]);
                __m256d a_vec = _mm256_set1_pd(a_il);
                __m256d prod = _mm256_mul_pd(a_vec, b_vec);
                c_vec = _mm256_add_pd(c_vec, prod);
                _mm256_storeu_pd(&C[j * m + i], c_vec);
            }

            // Handle remaining elements
            for (; j < n; ++j) {
                C[j * m + i] += a_il * B[j * k + l];
            }
        }
    }
}
#endif

#if defined(__AVX2__) || defined(__AVX512F__)
/**
 * @brief Dense matrix multiplication - AVX2 implementation with FMA
 *
 * Uses FMA (Fused Multiply-Add) for better performance:
 * C += A * B in a single instruction
 *
 * @param C Result matrix (m×n)
 * @param A Left matrix (m×k)
 * @param B Right matrix (k×n)
 * @param m Number of rows in A and C
 * @param k Number of columns in A and rows in B
 * @param n Number of columns in B and C
 */
inline void matrix_multiply_avx2(double* C, const double* A,
                                 const double* B, int m, int k, int n) {
    // Initialize C to zero
    std::memset(C, 0, m * n * sizeof(double));

    for (int i = 0; i < m; ++i) {
        for (int l = 0; l < k; ++l) {
            double a_il = A[l * m + i];

            int j = 0;
            // Vectorized loop with FMA: process 4 elements at a time
            for (; j + 4 <= n; j += 4) {
                __m256d c_vec = _mm256_loadu_pd(&C[j * m + i]);
                __m256d b_vec = _mm256_loadu_pd(&B[j * k + l]);
                __m256d a_vec = _mm256_set1_pd(a_il);
                // FMA: C += A * B in one instruction
                c_vec = _mm256_fmadd_pd(a_vec, b_vec, c_vec);
                _mm256_storeu_pd(&C[j * m + i], c_vec);
            }

            // Handle remaining elements
            for (; j < n; ++j) {
                C[j * m + i] += a_il * B[j * k + l];
            }
        }
    }
}
#endif

#if defined(__AVX512F__)
/**
 * @brief Dense matrix multiplication - AVX-512 implementation (512-bit, 8 doubles)
 *
 * Uses AVX-512 vectorization for maximum performance
 *
 * @param C Result matrix (m×n)
 * @param A Left matrix (m×k)
 * @param B Right matrix (k×n)
 * @param m Number of rows in A and C
 * @param k Number of columns in A and rows in B
 * @param n Number of columns in B and C
 */
inline void matrix_multiply_avx512(double* C, const double* A,
                                   const double* B, int m, int k, int n) {
    // Initialize C to zero
    std::memset(C, 0, m * n * sizeof(double));

    for (int i = 0; i < m; ++i) {
        for (int l = 0; l < k; ++l) {
            double a_il = A[l * m + i];

            int j = 0;
            // Vectorized loop: process 8 elements at a time
            for (; j + 8 <= n; j += 8) {
                __m512d c_vec = _mm512_loadu_pd(&C[j * m + i]);
                __m512d b_vec = _mm512_loadu_pd(&B[j * k + l]);
                __m512d a_vec = _mm512_set1_pd(a_il);
                // FMA: C += A * B in one instruction
                c_vec = _mm512_fmadd_pd(a_vec, b_vec, c_vec);
                _mm512_storeu_pd(&C[j * m + i], c_vec);
            }

            // Handle remaining elements
            for (; j < n; ++j) {
                C[j * m + i] += a_il * B[j * k + l];
            }
        }
    }
}
#endif

// ============================================================================
// Sparse Matrix Operations (Dot Product)
// ============================================================================

/**
 * @brief Sparse dot product with SIMD optimization
 *
 * Computes dot product of two sparse vectors represented by indices and values
 * This is the core operation in sparse matrix multiplication
 *
 * @param result_values Array of values from first matrix row
 * @param result_indices Array of column indices from first matrix row
 * @param result_nnz Number of non-zeros in first matrix row
 * @param other_values Array of values from second matrix column
 * @param other_indices Array of row indices from second matrix column
 * @param other_nnz Number of non-zeros in second matrix column
 * @param tolerance Threshold for considering a value as non-zero
 * @return Dot product value
 */
inline double sparse_dot_product_scalar(
    const double* result_values, const int* result_indices, int result_nnz,
    const double* other_values, const int* other_indices, int other_nnz,
    double tolerance = 1e-12) {

    double sum = 0.0;
    int i = 0, j = 0;

    // Merge-like iteration through both sparse arrays
    while (i < result_nnz && j < other_nnz) {
        if (result_indices[i] == other_indices[j]) {
            if (std::abs(result_values[i]) > tolerance &&
                std::abs(other_values[j]) > tolerance) {
                sum += result_values[i] * other_values[j];
            }
            i++;
            j++;
        } else if (result_indices[i] < other_indices[j]) {
            i++;
        } else {
            j++;
        }
    }

    return sum;
}

#if defined(__SSE2__)
/**
 * @brief Sparse dot product - SSE implementation
 *
 * Accumulates multiple products in parallel using SSE
 */
inline double sparse_dot_product_sse(
    const double* result_values, const int* result_indices, int result_nnz,
    const double* other_values, const int* other_indices, int other_nnz,
    double tolerance = 1e-12) {

    // For sparse operations, SIMD helps when we have multiple matching indices
    // Use SSE to accumulate products in parallel when possible

    __m128d sum_vec = _mm_setzero_pd();
    double scalar_sum = 0.0;
    int match_count = 0;
    double matches[2];

    int i = 0, j = 0;
    while (i < result_nnz && j < other_nnz) {
        if (result_indices[i] == other_indices[j]) {
            if (std::abs(result_values[i]) > tolerance &&
                std::abs(other_values[j]) > tolerance) {

                if (match_count < 2) {
                    matches[match_count++] = result_values[i] * other_values[j];
                } else {
                    // Flush matches to SIMD register
                    if (match_count == 2) {
                        __m128d match_vec = _mm_loadu_pd(matches);
                        sum_vec = _mm_add_pd(sum_vec, match_vec);
                        match_count = 0;
                    }
                    scalar_sum += result_values[i] * other_values[j];
                }
            }
            i++;
            j++;
        } else if (result_indices[i] < other_indices[j]) {
            i++;
        } else {
            j++;
        }
    }

    // Flush remaining matches
    if (match_count > 0) {
        __m128d match_vec = _mm_loadu_pd(matches);
        sum_vec = _mm_add_pd(sum_vec, match_vec);
    }

    // Extract final sum
    double sum_array[2];
    _mm_storeu_pd(sum_array, sum_vec);
    return sum_array[0] + sum_array[1] + scalar_sum;
}
#endif

#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
/**
 * @brief Sparse dot product - AVX implementation
 *
 * Accumulates multiple products in parallel using AVX
 */
inline double sparse_dot_product_avx(
    const double* result_values, const int* result_indices, int result_nnz,
    const double* other_values, const int* other_indices, int other_nnz,
    double tolerance = 1e-12) {

    __m256d sum_vec = _mm256_setzero_pd();
    double scalar_sum = 0.0;
    int match_count = 0;
    double matches[4];

    int i = 0, j = 0;
    while (i < result_nnz && j < other_nnz) {
        if (result_indices[i] == other_indices[j]) {
            if (std::abs(result_values[i]) > tolerance &&
                std::abs(other_values[j]) > tolerance) {

                if (match_count < 4) {
                    matches[match_count++] = result_values[i] * other_values[j];
                } else {
                    // Flush matches to SIMD register
                    __m256d match_vec = _mm256_loadu_pd(matches);
                    sum_vec = _mm256_add_pd(sum_vec, match_vec);
                    match_count = 0;
                }
            }
            i++;
            j++;
        } else if (result_indices[i] < other_indices[j]) {
            i++;
        } else {
            j++;
        }
    }

    // Flush remaining matches
    if (match_count > 0) {
        // Zero-pad remaining elements
        for (int k = match_count; k < 4; ++k) {
            matches[k] = 0.0;
        }
        __m256d match_vec = _mm256_loadu_pd(matches);
        sum_vec = _mm256_add_pd(sum_vec, match_vec);
    }

    // Extract final sum (horizontal add)
    double sum_array[4];
    _mm256_storeu_pd(sum_array, sum_vec);
    return sum_array[0] + sum_array[1] + sum_array[2] + sum_array[3] + scalar_sum;
}
#endif

#if defined(__AVX2__) || defined(__AVX512F__)
/**
 * @brief Sparse dot product - AVX2 implementation (same as AVX)
 *
 * For sparse operations, AVX2 doesn't provide significant advantage over AVX
 * The main benefit would be in dense operations with FMA
 */
inline double sparse_dot_product_avx2(
    const double* result_values, const int* result_indices, int result_nnz,
    const double* other_values, const int* other_indices, int other_nnz,
    double tolerance = 1e-12) {
    return sparse_dot_product_avx(result_values, result_indices, result_nnz,
                                  other_values, other_indices, other_nnz,
                                  tolerance);
}
#endif

#if defined(__AVX512F__)
/**
 * @brief Sparse dot product - AVX-512 implementation
 *
 * Accumulates multiple products in parallel using AVX-512
 */
inline double sparse_dot_product_avx512(
    const double* result_values, const int* result_indices, int result_nnz,
    const double* other_values, const int* other_indices, int other_nnz,
    double tolerance = 1e-12) {

    __m512d sum_vec = _mm512_setzero_pd();
    double scalar_sum = 0.0;
    int match_count = 0;
    double matches[8];

    int i = 0, j = 0;
    while (i < result_nnz && j < other_nnz) {
        if (result_indices[i] == other_indices[j]) {
            if (std::abs(result_values[i]) > tolerance &&
                std::abs(other_values[j]) > tolerance) {

                if (match_count < 8) {
                    matches[match_count++] = result_values[i] * other_values[j];
                } else {
                    // Flush matches to SIMD register
                    __m512d match_vec = _mm512_loadu_pd(matches);
                    sum_vec = _mm512_add_pd(sum_vec, match_vec);
                    match_count = 0;
                }
            }
            i++;
            j++;
        } else if (result_indices[i] < other_indices[j]) {
            i++;
        } else {
            j++;
        }
    }

    // Flush remaining matches
    if (match_count > 0) {
        // Zero-pad remaining elements
        for (int k = match_count; k < 8; ++k) {
            matches[k] = 0.0;
        }
        __m512d match_vec = _mm512_loadu_pd(matches);
        sum_vec = _mm512_add_pd(sum_vec, match_vec);
    }

    // Extract final sum (horizontal add)
    return _mm512_reduce_add_pd(sum_vec) + scalar_sum;
}
#endif

} // namespace MatrixOps_SIMD

#endif // MATRIX_OPS_SIMD_H
