#ifndef INVERSEPOWERMETHODSOLVER_H
#define INVERSEPOWERMETHODSOLVER_H

#include "IEigenSolver.h"

/**
 * Inverse power method solver for finding the smallest magnitude eigenvalue
 * Returns a single eigenvalue in an array of size 1 for interface consistency
 */
class InversePowerMethodSolver : public IEigenSolver {
public:
    InversePowerMethodSolver() = default;

    /**
     * Compute smallest eigenvalue using inverse power method
     * @param matrix Input matrix (must be square and invertible)
     * @param eigenvalue_count Output parameter (will be set to 1)
     * @return Heap-allocated array with single eigenvalue (caller must delete[])
     */
    double* compute(const Matrix& matrix, int& eigenvalue_count) override;

    /**
     * Get algorithm name
     * @return "Inverse Power Method"
     */
    std::string getName() const override;

private:
    /**
     * Simple Gaussian elimination solver for Aw = v
     * This is a basic implementation for demonstration
     * In production, use LAPACK's LU decomposition
     */
    Matrix solveLinearSystem(const Matrix& A, const Matrix& b);
};

#endif // INVERSEPOWERMETHODSOLVER_H
