#include "QRSolver.h"
#include <stdexcept>
#include <cmath>

double* QRSolver::compute(const Matrix& matrix, int& eigenvalue_count) {
    if (matrix.getRows() != matrix.getCols()) {
        throw std::invalid_argument("Matrix must be square for eigenvalue computation");
    }

    if (parallel_mode) {
        return computeParallel(matrix, eigenvalue_count);
    } else {
        return computeSerial(matrix, eigenvalue_count);
    }
}

std::string QRSolver::getName() const {
    return parallel_mode ? "QR Algorithm (Parallel)" : "QR Algorithm";
}

double* QRSolver::computeSerial(const Matrix& matrix, int& eigenvalue_count) {
    Matrix A(matrix);
    int n = A.getRows();
    eigenvalue_count = n;
    double* eigenvalues = new double[n];

    for (int iter = 0; iter < max_iterations; ++iter) {
        // Check for deflation
        if (iter > 0 && std::abs(A(n-1, n-2)) < tolerance) {
            eigenvalues[n-1] = A(n-1, n-1);
            n--;
            if (n <= 1) break;
            continue;
        }

        // Perform QR decomposition
        Matrix Q(n, n), R(n, n);
        SolverUtils::qrDecomposition(A, Q, R);
        A = R.multiply(Q);  // Update A = R * Q

        // Check for convergence
        bool converged = true;
        for (int i = 1; i < n; ++i) {
            if (std::abs(A(i, i-1)) > tolerance) {
                converged = false;
                break;
            }
        }

        if (converged) {
            for (int i = 0; i < n; ++i) {
                eigenvalues[i] = A(i, i);
            }
            break;
        }
    }

    // Fill in any remaining eigenvalues
    for (int i = 0; i < n; ++i) {
        eigenvalues[i] = A(i, i);
    }

    return eigenvalues;
}

double* QRSolver::computeParallel(const Matrix& matrix, int& eigenvalue_count) {
    Matrix A(matrix);
    int n = A.getRows();
    eigenvalue_count = n;
    double* eigenvalues = new double[n];

    for (int iter = 0; iter < max_iterations; ++iter) {
        // Check for deflation
        if (iter > 0 && std::abs(A(n-1, n-2)) < tolerance) {
            eigenvalues[n-1] = A(n-1, n-1);
            n--;
            if (n <= 1) break;
            continue;
        }

        // Perform QR decomposition in parallel
        Matrix Q(n, n), R(n, n);
        SolverUtils::qrDecompositionParallel(A, Q, R);
        A = R.multiply_parallel(Q);  // Update A = R * Q

        // Check for convergence using parallel reduction
        bool converged = true;
        #pragma omp parallel
        {
            bool local_converged = true;
            #pragma omp for nowait
            for (int i = 1; i < n; ++i) {
                if (std::abs(A(i, i-1)) > tolerance) {
                    local_converged = false;
                }
            }
            #pragma omp critical
            {
                if (!local_converged) {
                    converged = false;
                }
            }
        }

        if (converged) {
            #pragma omp parallel for
            for (int i = 0; i < n; ++i) {
                eigenvalues[i] = A(i, i);
            }
            break;
        }
    }

    // Fill in any remaining eigenvalues
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        eigenvalues[i] = A(i, i);
    }

    return eigenvalues;
}
