#include "JacobiSolver.h"
#include "../matrix/SymmetricMatrix.h"
#include <stdexcept>
#include <cmath>

double* JacobiSolver::compute(const Matrix& matrix, int& eigenvalue_count) {
    if (matrix.getRows() != matrix.getCols()) {
        throw std::invalid_argument("Matrix must be square for eigenvalue computation");
    }

    // Check if matrix is SymmetricMatrix type for optimization
    bool is_explicitly_symmetric = SolverUtils::isSymmetricMatrix(matrix);
    bool is_mathematically_symmetric = SolverUtils::isSymmetric(matrix, tolerance);

    if (!is_mathematically_symmetric) {
        throw std::invalid_argument("Jacobi method requires a symmetric matrix");
    }

    // Log optimization info if explicitly using SymmetricMatrix
    if (is_explicitly_symmetric) {
        // Using SymmetricMatrix type enables optimizations
        // Future: Could use half-storage representation
    }

    if (parallel_mode) {
        return computeParallel(matrix, eigenvalue_count);
    } else {
        return computeSerial(matrix, eigenvalue_count);
    }
}

std::string JacobiSolver::getName() const {
    return parallel_mode ? "Jacobi Method (Parallel)" : "Jacobi Method";
}

double* JacobiSolver::computeSerial(const Matrix& matrix, int& eigenvalue_count) {
    Matrix A(matrix);
    int n = A.getRows();
    eigenvalue_count = n;
    double* eigenvalues = new double[n];

    for (int iter = 0; iter < max_iterations; ++iter) {
        // Find the largest off-diagonal element
        int p = 0, q = 1;
        double max_val = std::abs(A(p, q));

        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                if (std::abs(A(i, j)) > max_val) {
                    max_val = std::abs(A(i, j));
                    p = i;
                    q = j;
                }
            }
        }

        // Check for convergence
        if (max_val < tolerance) {
            break;
        }

        // Calculate rotation angle
        double theta = 0.5 * std::atan2(2 * A(p, q), A(q, q) - A(p, p));
        double c = std::cos(theta);
        double s = std::sin(theta);

        // Apply Jacobi rotation: A = J^T * A * J
        Matrix A_new = A;

        // Zero out the p,q and q,p elements
        A_new.set(p, q, 0.0);
        A_new.set(q, p, 0.0);

        // Update the p-th and q-th rows and columns
        for (int i = 0; i < n; ++i) {
            if (i != p && i != q) {
                double temp_p = c * A(i, p) - s * A(i, q);
                double temp_q = s * A(i, p) + c * A(i, q);
                A_new.set(i, p, temp_p);
                A_new.set(p, i, temp_p);  // Maintain symmetry
                A_new.set(i, q, temp_q);
                A_new.set(q, i, temp_q);  // Maintain symmetry
            }
        }

        // Update diagonal elements
        A_new.set(p, p, c*c*A(p, p) - 2*s*c*A(p, q) + s*s*A(q, q));
        A_new.set(q, q, s*s*A(p, p) + 2*s*c*A(p, q) + c*c*A(q, q));

        A = A_new;
    }

    // Extract eigenvalues from diagonal
    for (int i = 0; i < n; ++i) {
        eigenvalues[i] = A(i, i);
    }

    return eigenvalues;
}

double* JacobiSolver::computeParallel(const Matrix& matrix, int& eigenvalue_count) {
    Matrix A(matrix);
    int n = A.getRows();
    eigenvalue_count = n;
    double* eigenvalues = new double[n];

    for (int iter = 0; iter < max_iterations; ++iter) {
        // Find the largest off-diagonal element
        int p = 0, q = 1;
        double max_val = 0.0;

        // Serial search for maximum element (not a performance bottleneck)
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                double abs_val = std::abs(A(i, j));
                if (abs_val > max_val) {
                    max_val = abs_val;
                    p = i;
                    q = j;
                }
            }
        }

        // Check for convergence
        if (max_val < tolerance) {
            break;
        }

        // Calculate rotation angle
        double theta = 0.5 * std::atan2(2 * A(p, q), A(q, q) - A(p, p));
        double c = std::cos(theta);
        double s = std::sin(theta);

        // Apply Jacobi rotation in parallel
        Matrix A_new = A;

        // Zero out the p,q and q,p elements
        A_new.set(p, q, 0.0);
        A_new.set(q, p, 0.0);

        // Update the p-th and q-th rows and columns in parallel
        #pragma omp parallel sections
        {
            #pragma omp section
            {
                // Update row/column p
                for (int i = 0; i < n; ++i) {
                    if (i != p && i != q) {
                        double temp_p = c * A(i, p) - s * A(i, q);
                        A_new.set(i, p, temp_p);
                        A_new.set(p, i, temp_p);  // Maintain symmetry
                    }
                }
            }
            #pragma omp section
            {
                // Update row/column q
                for (int i = 0; i < n; ++i) {
                    if (i != p && i != q) {
                        double temp_q = s * A(i, p) + c * A(i, q);
                        A_new.set(i, q, temp_q);
                        A_new.set(q, i, temp_q);  // Maintain symmetry
                    }
                }
            }
            #pragma omp section
            {
                // Update diagonal elements
                A_new.set(p, p, c*c*A(p, p) - 2*s*c*A(p, q) + s*s*A(q, q));
                A_new.set(q, q, s*s*A(p, p) + 2*s*c*A(p, q) + c*c*A(q, q));
            }
        }

        A = A_new;
    }

    // Extract eigenvalues from diagonal
    for (int i = 0; i < n; ++i) {
        eigenvalues[i] = A(i, i);
    }

    return eigenvalues;
}
