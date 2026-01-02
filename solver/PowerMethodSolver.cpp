#include "PowerMethodSolver.h"
#include <stdexcept>
#include <cmath>
#include <random>

double* PowerMethodSolver::compute(const Matrix& matrix, int& eigenvalue_count) {
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

    for (int iter = 0; iter < max_iterations; ++iter) {
        // Compute Av
        Matrix Av = matrix.multiply(v);

        // Calculate Rayleigh quotient: lambda = (v^T * Av) / (v^T * v)
        double numerator = 0.0, denominator = 0.0;
        for (int i = 0; i < n; ++i) {
            numerator += v(i, 0) * Av(i, 0);
            denominator += v(i, 0) * v(i, 0);
        }

        double new_eigenvalue = (std::abs(denominator) > tolerance) ?
                               numerator / denominator : 0.0;

        // Check for convergence
        if (iter > 0 && std::abs(new_eigenvalue - eigenvalue) < tolerance) {
            eigenvalue = new_eigenvalue;
            break;
        }

        eigenvalue = new_eigenvalue;

        // Normalize Av to get new v
        norm = 0.0;
        for (int i = 0; i < n; ++i) {
            norm += Av(i, 0) * Av(i, 0);
        }
        norm = std::sqrt(norm);

        if (norm > tolerance) {
            v = Av * (1.0 / norm);
        } else {
            break;
        }
    }

    eigenvalues[0] = eigenvalue;
    return eigenvalues;
}

std::string PowerMethodSolver::getName() const {
    return "Power Method";
}
