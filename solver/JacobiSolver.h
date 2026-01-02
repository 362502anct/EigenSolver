#ifndef JACOBISOLVER_H
#define JACOBISOLVER_H

#include "IEigenSolver.h"
#include "SolverUtils.h"

/**
 * Jacobi method solver for symmetric matrices
 * Uses Givens rotations to diagonalize the matrix
 * Supports both serial and parallel execution modes
 */
class JacobiSolver : public IEigenSolver {
public:
    JacobiSolver() = default;

    /**
     * Compute eigenvalues using Jacobi method
     * @param matrix Input matrix (must be symmetric)
     * @param eigenvalue_count Output parameter for number of eigenvalues
     * @return Heap-allocated array of eigenvalues (caller must delete[])
     */
    double* compute(const Matrix& matrix, int& eigenvalue_count) override;

    /**
     * Get algorithm name
     * @return "Jacobi Method" (serial) or "Jacobi Method (Parallel)" (parallel mode)
     */
    std::string getName() const override;

private:
    /**
     * Serial Jacobi method implementation
     */
    double* computeSerial(const Matrix& matrix, int& eigenvalue_count);

    /**
     * Parallel Jacobi method implementation with OpenMP
     */
    double* computeParallel(const Matrix& matrix, int& eigenvalue_count);
};

#endif // JACOBISOLVER_H
