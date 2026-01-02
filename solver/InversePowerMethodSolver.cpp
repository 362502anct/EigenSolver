#include "InversePowerMethodSolver.h"
#include <stdexcept>
#include <cmath>
#include <random>

double* InversePowerMethodSolver::compute(const Matrix& matrix, int& eigenvalue_count) {
    if (matrix.getRows() != matrix.getCols()) {
        throw std::invalid_argument("Matrix must be square for eigenvalue computation");
    }

    int n = matrix.getRows();
    eigenvalue_count = 1;
    double* eigenvalues = new double[1];

    // Initialize random vector
    Matrix v(n, 1);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);

    for (int i = 0; i < n; ++i) {
        v.set(i, 0, dis(gen));
    }

    // Normalize initial vector
    double norm = 0.0;
    for (int i = 0; i < n; ++i) {
        norm += v(i, 0) * v(i, 0);
    }
    norm = std::sqrt(norm);

    if (norm > 0) {
        v = v * (1.0 / norm);
    }

    double eigenvalue = 0.0;

    // Note: This is a simplified implementation
    // In practice, we would solve A * w = v using LU decomposition
    // rather than explicitly computing the inverse
    for (int iter = 0; iter < max_iterations; ++iter) {
        // Simplified approach: solve A * w = v
        // In production code, use LU decomposition for efficiency
        Matrix A_dense = matrix;  // Create a copy for manipulation

        // Simple Gaussian elimination to solve Aw = v
        Matrix w = solveLinearSystem(A_dense, v);

        // Calculate Rayleigh quotient for the inverse
        double numerator = 0.0, denominator = 0.0;
        for (int i = 0; i < n; ++i) {
            numerator += v(i, 0) * w(i, 0);
            denominator += v(i, 0) * v(i, 0);
        }

        double inv_eigenvalue = (std::abs(denominator) > tolerance) ?
                               numerator / denominator : 0.0;

        // Take reciprocal to get eigenvalue of A
        double new_eigenvalue = (std::abs(inv_eigenvalue) > tolerance) ?
                               1.0 / inv_eigenvalue : 0.0;

        // Check for convergence
        if (iter > 0 && std::abs(new_eigenvalue - eigenvalue) < tolerance) {
            eigenvalue = new_eigenvalue;
            break;
        }

        eigenvalue = new_eigenvalue;

        // Normalize w
        norm = 0.0;
        for (int i = 0; i < n; ++i) {
            norm += w(i, 0) * w(i, 0);
        }
        norm = std::sqrt(norm);

        if (norm > tolerance) {
            v = w * (1.0 / norm);
        } else {
            break;
        }
    }

    eigenvalues[0] = eigenvalue;
    return eigenvalues;
}

std::string InversePowerMethodSolver::getName() const {
    return "Inverse Power Method";
}

Matrix InversePowerMethodSolver::solveLinearSystem(const Matrix& A, const Matrix& b) {
    int n = A.getRows();
    Matrix augmented(n, n + 1);

    // Create augmented matrix [A | b]
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            augmented.set(i, j, A(i, j));
        }
        augmented.set(i, n, b(i, 0));
    }

    // Gaussian elimination with partial pivoting
    for (int col = 0; col < n; ++col) {
        // Find pivot
        int pivot_row = col;
        double max_val = std::abs(augmented(col, col));
        for (int row = col + 1; row < n; ++row) {
            if (std::abs(augmented(row, col)) > max_val) {
                max_val = std::abs(augmented(row, col));
                pivot_row = row;
            }
        }

        // Swap rows
        if (pivot_row != col) {
            for (int j = col; j <= n; ++j) {
                double temp = augmented(col, j);
                augmented.set(col, j, augmented(pivot_row, j));
                augmented.set(pivot_row, j, temp);
            }
        }

        // Eliminate column
        for (int row = col + 1; row < n; ++row) {
            double factor = augmented(row, col) / augmented(col, col);
            for (int j = col; j <= n; ++j) {
                augmented.set(row, j, augmented(row, j) - factor * augmented(col, j));
            }
        }
    }

    // Back substitution
    Matrix x(n, 1);
    for (int i = n - 1; i >= 0; --i) {
        double sum = augmented(i, n);
        for (int j = i + 1; j < n; ++j) {
            sum -= augmented(i, j) * x(j, 0);
        }
        x.set(i, 0, sum / augmented(i, i));
    }

    return x;
}
