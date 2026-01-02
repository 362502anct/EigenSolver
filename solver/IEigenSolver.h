#ifndef IEIGENSOLVER_H
#define IEIGENSOLVER_H

#include "../matrix/Matrix.h"
#include <string>

/**
 * Abstract base class for eigenvalue solvers
 * All concrete algorithm implementations must inherit from this interface
 */
class IEigenSolver {
public:
    virtual ~IEigenSolver() = default;

    /**
     * Compute eigenvalues of the given matrix
     * @param matrix Input matrix (must be square)
     * @param eigenvalue_count Output parameter for number of eigenvalues computed
     * @return Heap-allocated array of eigenvalues (caller must delete[])
     */
    virtual double* compute(const Matrix& matrix, int& eigenvalue_count) = 0;

    /**
     * Get the name of this algorithm
     * @return Algorithm name as string
     */
    virtual std::string getName() const = 0;

    /**
     * Configure parallel mode for algorithms that support it
     * @param parallel True to enable parallelization, false for serial execution
     */
    void setParallelMode(bool parallel) { parallel_mode = parallel; }

    /**
     * Set maximum iterations for iterative algorithms
     * @param max_iter Maximum number of iterations
     */
    void setMaxIterations(int max_iter) { max_iterations = max_iter; }

    /**
     * Set convergence tolerance
     * @param tol Tolerance value (default: 1e-10)
     */
    void setTolerance(double tol) { tolerance = tol; }

    /**
     * Check if parallel mode is enabled
     * @return True if parallel mode is active
     */
    bool isParallelMode() const { return parallel_mode; }

    /**
     * Get configured maximum iterations
     * @return Maximum iterations
     */
    int getMaxIterations() const { return max_iterations; }

    /**
     * Get configured tolerance
     * @return Tolerance value
     */
    double getTolerance() const { return tolerance; }

protected:
    bool parallel_mode = false;
    int max_iterations = 1000;
    double tolerance = 1e-10;
};

#endif // IEIGENSOLVER_H
