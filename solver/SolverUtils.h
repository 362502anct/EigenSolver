#ifndef SOLVERUTILS_H
#define SOLVERUTILS_H

#include "../matrix/Matrix.h"
#include "../matrix/SymmetricMatrix.h"
#include <cmath>

/**
 * Utility functions for eigenvalue solver implementations
 * Shared helper functions used across multiple solver algorithms
 */
namespace SolverUtils {

    /**
     * Check if a matrix is symmetric
     * @param matrix Input matrix
     * @param tolerance Tolerance for symmetry check (default: 1e-10)
     * @return True if matrix is symmetric within tolerance
     */
    inline bool isSymmetric(const Matrix& matrix, double tolerance = 1e-10) {
        if (matrix.getRows() != matrix.getCols()) {
            return false;
        }

        int n = matrix.getRows();
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                if (std::abs(matrix(i, j) - matrix(j, i)) > tolerance) {
                    return false;
                }
            }
        }
        return true;
    }

    /**
     * Check if matrix is of type SymmetricMatrix
     * Uses dynamic_cast to check actual type
     * @param matrix Input matrix
     * @return True if matrix is a SymmetricMatrix instance
     */
    inline bool isSymmetricMatrix(const Matrix& matrix) {
        return dynamic_cast<const SymmetricMatrix*>(&matrix) != nullptr;
    }

    /**
     * Get SymmetricMatrix pointer if possible
     * @param matrix Input matrix
     * @return Pointer to SymmetricMatrix if type matches, nullptr otherwise
     */
    inline const SymmetricMatrix* asSymmetricMatrix(const Matrix& matrix) {
        return dynamic_cast<const SymmetricMatrix*>(&matrix);
    }

    inline SymmetricMatrix* asSymmetricMatrix(Matrix& matrix) {
        return dynamic_cast<SymmetricMatrix*>(&matrix);
    }

    /**
     * Check if a matrix is tridiagonal
     * @param matrix Input matrix
     * @param tolerance Tolerance for zero check (default: 1e-10)
     * @return True if matrix is tridiagonal within tolerance
     */
    inline bool isTridiagonal(const Matrix& matrix, double tolerance = 1e-10) {
        if (matrix.getRows() != matrix.getCols()) {
            return false;
        }

        int n = matrix.getRows();
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                // Check if element is outside the three diagonals
                if (std::abs(i - j) > 1 && std::abs(matrix(i, j)) > tolerance) {
                    return false;
                }
            }
        }
        return true;
    }

    /**
     * Perform QR decomposition using Gram-Schmidt process
     * @param matrix Input matrix (A)
     * @param Q Output orthogonal matrix
     * @param R Output upper triangular matrix
     */
    inline void qrDecomposition(const Matrix& matrix, Matrix& Q, Matrix& R) {
        int n = matrix.getRows();

        // Initialize Q and R as zero matrices
        Q = Matrix::zeros(n, n);
        R = Matrix::zeros(n, n);

        // Gram-Schmidt orthogonalization
        for (int j = 0; j < n; ++j) {
            // Copy column j of A to column j of Q
            for (int i = 0; i < n; ++i) {
                Q.set(i, j, matrix(i, j));
            }

            // Orthogonalize against previous columns
            for (int k = 0; k < j; ++k) {
                // Compute dot product
                double dot = 0.0;
                for (int i = 0; i < n; ++i) {
                    dot += Q(i, k) * matrix(i, j);
                }

                // Store in R
                R.set(k, j, dot);

                // Subtract projection
                for (int i = 0; i < n; ++i) {
                    Q.set(i, j, Q(i, j) - dot * Q(i, k));
                }
            }

            // Compute norm of column j
            double norm = 0.0;
            for (int i = 0; i < n; ++i) {
                norm += Q(i, j) * Q(i, j);
            }
            norm = std::sqrt(norm);

            // Store in R
            R.set(j, j, norm);

            // Normalize column j of Q
            if (norm > 1e-15) {
                for (int i = 0; i < n; ++i) {
                    Q.set(i, j, Q(i, j) / norm);
                }
            }
        }
    }

    /**
     * Perform parallel QR decomposition using Gram-Schmidt process with OpenMP
     * @param matrix Input matrix (A)
     * @param Q Output orthogonal matrix
     * @param R Output upper triangular matrix
     */
    inline void qrDecompositionParallel(const Matrix& matrix, Matrix& Q, Matrix& R) {
        int n = matrix.getRows();

        // Initialize Q and R as zero matrices
        Q = Matrix::zeros(n, n);
        R = Matrix::zeros(n, n);

        // Gram-Schmidt orthogonalization with OpenMP
        for (int j = 0; j < n; ++j) {
            // Copy column j of A to column j of Q
            #pragma omp parallel for if(n > 100)
            for (int i = 0; i < n; ++i) {
                Q.set(i, j, matrix(i, j));
            }

            // Orthogonalize against previous columns
            for (int k = 0; k < j; ++k) {
                // Compute dot product
                double dot = 0.0;
                #pragma omp parallel for reduction(+:dot) if(n > 100)
                for (int i = 0; i < n; ++i) {
                    dot += Q(i, k) * matrix(i, j);
                }

                // Store in R
                R.set(k, j, dot);

                // Subtract projection
                #pragma omp parallel for if(n > 100)
                for (int i = 0; i < n; ++i) {
                    Q.set(i, j, Q(i, j) - dot * Q(i, k));
                }
            }

            // Compute norm of column j
            double norm = 0.0;
            #pragma omp parallel for reduction(+:norm) if(n > 100)
            for (int i = 0; i < n; ++i) {
                norm += Q(i, j) * Q(i, j);
            }
            norm = std::sqrt(norm);

            // Store in R
            R.set(j, j, norm);

            // Normalize column j of Q
            if (norm > 1e-15) {
                #pragma omp parallel for if(n > 100)
                for (int i = 0; i < n; ++i) {
                    Q.set(i, j, Q(i, j) / norm);
                }
            }
        }
    }

    /**
     * Apply Givens rotation to matrix
     * @param matrix Input/output matrix
     * @param i Row index
     * @param j Column index
     * @param cos_val Cosine value
     * @param sin_val Sine value
     */
    inline void givensRotate(Matrix& matrix, int i, int j, double cos_val, double sin_val) {
        int n = matrix.getRows();

        for (int k = 0; k < n; ++k) {
            double temp_i = matrix(i, k);
            double temp_j = matrix(j, k);
            matrix.set(i, k, cos_val * temp_i - sin_val * temp_j);
            matrix.set(j, k, sin_val * temp_i + cos_val * temp_j);
        }
    }

} // namespace SolverUtils

#endif // SOLVERUTILS_H
