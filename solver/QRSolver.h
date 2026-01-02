#ifndef QRSOLVER_H
#define QRSOLVER_H

#include "IEigenSolver.h"
#include "SolverUtils.h"

/**
 * QR Algorithm solver for eigenvalue computation
 * Supports both serial and parallel execution modes
 */
class QRSolver : public IEigenSolver {
public:
    QRSolver() = default;

    /**
     * Compute eigenvalues using QR algorithm
     * @param matrix Input matrix (must be square)
     * @param eigenvalue_count Output parameter for number of eigenvalues
     * @return Heap-allocated array of eigenvalues (caller must delete[])
     */
    double* compute(const Matrix& matrix, int& eigenvalue_count) override;

    /**
     * Get algorithm name
     * @return "QR Algorithm" (serial) or "QR Algorithm (Parallel)" (parallel mode)
     */
    std::string getName() const override;

private:
    /**
     * Serial QR algorithm implementation
     */
    double* computeSerial(const Matrix& matrix, int& eigenvalue_count);

    /**
     * Parallel QR algorithm implementation with OpenMP
     */
    double* computeParallel(const Matrix& matrix, int& eigenvalue_count);
};

#endif // QRSOLVER_H
