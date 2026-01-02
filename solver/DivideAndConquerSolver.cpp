#include "DivideAndConquerSolver.h"
#include "JacobiSolver.h"
#include <stdexcept>

double* DivideAndConquerSolver::compute(const Matrix& matrix, int& eigenvalue_count) {
    if (matrix.getRows() != matrix.getCols()) {
        throw std::invalid_argument("Matrix must be square for eigenvalue computation");
    }

    if (!SolverUtils::isSymmetric(matrix, tolerance)) {
        throw std::invalid_argument("Divide and Conquer method requires a symmetric matrix");
    }

    // Placeholder: fall back to Jacobi method
    // A full divide and conquer implementation would:
    // 1. Tridiagonalize the matrix (if not already)
    // 2. Recursively divide the matrix
    // 3. Solve the secular equation
    // 4. Conquer by merging results

    JacobiSolver jacobi;
    jacobi.setMaxIterations(max_iterations);
    jacobi.setTolerance(tolerance);

    return jacobi.compute(matrix, eigenvalue_count);
}

std::string DivideAndConquerSolver::getName() const {
    return "Divide and Conquer";
}
