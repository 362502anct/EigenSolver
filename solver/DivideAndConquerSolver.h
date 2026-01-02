#ifndef DIVIDEANDCONQUERSOLVER_H
#define DIVIDEANDCONQUERSOLVER_H

#include "IEigenSolver.h"
#include "SolverUtils.h"

/**
 * Divide and Conquer solver for symmetric tridiagonal matrices
 * This is a placeholder implementation that falls back to Jacobi method
 * A full implementation would use the divide and conquer algorithm
 * which is O(n^2) for tridiagonal matrices
 */
class DivideAndConquerSolver : public IEigenSolver {
public:
    DivideAndConquerSolver() = default;

    /**
     * Compute eigenvalues using divide and conquer method
     * Note: Current implementation falls back to Jacobi method
     * A production implementation would use the proper divide and conquer algorithm
     *
     * @param matrix Input matrix (should be symmetric tridiagonal)
     * @param eigenvalue_count Output parameter for number of eigenvalues
     * @return Heap-allocated array of eigenvalues (caller must delete[])
     */
    double* compute(const Matrix& matrix, int& eigenvalue_count) override;

    /**
     * Get algorithm name
     * @return "Divide and Conquer"
     */
    std::string getName() const override;
};

#endif // DIVIDEANDCONQUERSOLVER_H
