#ifndef EIGENSOLVER_H
#define EIGENSOLVER_H

#include "../matrix/Matrix.h"
#include "../util/Utils.h"
#include "SolverFactory.h"
#include "IEigenSolver.h"
#include <memory>
#include <vector>
#include <functional>

/**
 * EigenSolver - Compatibility layer for legacy static method interface
 * This class maintains backward compatibility while using the new OOP architecture
 */
class EigenSolver {
public:
    // ========== Legacy Static Methods (Backward Compatibility) ==========

    // QR Algorithm with shifts for eigenvalue computation
    static double* qrAlgorithm(const Matrix& matrix, int& eigenvalue_count, int maxIterations = 1000, double tolerance = 1e-10);

    // Parallel QR Algorithm implementation
    static double* qrAlgorithmParallel(const Matrix& matrix, int& eigenvalue_count, int maxIterations = 1000, double tolerance = 1e-10);

    // Power method for finding dominant eigenvalue (now returns array for consistency)
    static double* powerMethod(const Matrix& matrix, int& eigenvalue_count, int maxIterations = 1000, double tolerance = 1e-10);

    // Inverse power method for finding smallest eigenvalue (now returns array for consistency)
    static double* inversePowerMethod(const Matrix& matrix, int& eigenvalue_count, int maxIterations = 1000, double tolerance = 1e-10);

    // Jacobi method for symmetric matrices
    static double* jacobiMethod(const Matrix& matrix, int& eigenvalue_count, int maxIterations = 1000, double tolerance = 1e-10);

    // Parallel Jacobi method
    static double* jacobiMethodParallel(const Matrix& matrix, int& eigenvalue_count, int maxIterations = 1000, double tolerance = 1e-10);

    // Divide and Conquer method for symmetric tridiagonal matrices
    static double* divideAndConquer(const Matrix& matrix, int& eigenvalue_count, double tolerance = 1e-10);

    // Compute all eigenvalues using the best method based on matrix properties
    static double* solve(const Matrix& matrix, int& eigenvalue_count, const std::string& method = "auto",
                        int maxIterations = 1000, double tolerance = 1e-10);

    // Compute eigenvalues and eigenvectors
    static std::pair<double*, Matrix> solveWithEigenvectors(const Matrix& matrix,
                                                           int& eigenvalue_count,
                                                           const std::string& method = "auto",
                                                           int maxIterations = 1000,
                                                           double tolerance = 1e-10);

    // ========== New OOP Interface ==========

    /**
     * Create a solver instance using the factory
     * @param method Algorithm name (qr, jacobi, power, inverse_power, divide_conquer, auto)
     * @return Unique pointer to solver instance
     */
    static std::unique_ptr<IEigenSolver> createSolver(const std::string& method = "auto") {
        if (method == "auto") {
            return SolverFactory::createAuto(Matrix(1, 1));  // Dummy matrix, will be replaced in compute
        }
        return SolverFactory::create(method);
    }

    /**
     * Create a solver with automatic algorithm selection based on matrix
     * @param matrix Input matrix for analysis
     * @param tolerance Tolerance for matrix property checks
     * @return Unique pointer to solver instance
     */
    static std::unique_ptr<IEigenSolver> createAutoSolver(const Matrix& matrix, double tolerance = 1e-10) {
        return SolverFactory::createAuto(matrix, tolerance);
    }

private:
    // Helper functions are now in SolverUtils namespace
    // Kept here for backward compatibility with any external code that might use them
    static bool isSymmetric(const Matrix& matrix, double tolerance = 1e-10);
    static bool isTridiagonal(const Matrix& matrix, double tolerance = 1e-10);
};

#endif // EIGENSOLVER_H