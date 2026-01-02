#ifndef POWERMETHODSOLVER_H
#define POWERMETHODSOLVER_H

#include "IEigenSolver.h"

/**
 * Power method solver for finding the dominant (largest magnitude) eigenvalue
 * Returns a single eigenvalue in an array of size 1 for interface consistency
 */
class PowerMethodSolver : public IEigenSolver {
public:
    PowerMethodSolver() = default;

    /**
     * Compute dominant eigenvalue using power method
     * @param matrix Input matrix (must be square)
     * @param eigenvalue_count Output parameter (will be set to 1)
     * @return Heap-allocated array with single eigenvalue (caller must delete[])
     */
    double* compute(const Matrix& matrix, int& eigenvalue_count) override;

    /**
     * Get algorithm name
     * @return "Power Method"
     */
    std::string getName() const override;
};

#endif // POWERMETHODSOLVER_H
