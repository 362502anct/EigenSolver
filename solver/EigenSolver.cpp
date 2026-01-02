#include "EigenSolver.h"
#include "QRSolver.h"
#include "JacobiSolver.h"
#include "PowerMethodSolver.h"
#include "InversePowerMethodSolver.h"
#include "DivideAndConquerSolver.h"
#include "SolverUtils.h"
#include <stdexcept>

// ========== Legacy Static Method Implementations ==========

// QR Algorithm with shifts for eigenvalue computation
double* EigenSolver::qrAlgorithm(const Matrix& matrix, int& eigenvalue_count, int maxIterations, double tolerance) {
    QRSolver solver;
    solver.setMaxIterations(maxIterations);
    solver.setTolerance(tolerance);
    solver.setParallelMode(false);
    return solver.compute(matrix, eigenvalue_count);
}

// Parallel QR Algorithm implementation
double* EigenSolver::qrAlgorithmParallel(const Matrix& matrix, int& eigenvalue_count, int maxIterations, double tolerance) {
    QRSolver solver;
    solver.setMaxIterations(maxIterations);
    solver.setTolerance(tolerance);
    solver.setParallelMode(true);
    return solver.compute(matrix, eigenvalue_count);
}

// Power method for finding dominant eigenvalue (now returns array)
double* EigenSolver::powerMethod(const Matrix& matrix, int& eigenvalue_count, int maxIterations, double tolerance) {
    PowerMethodSolver solver;
    solver.setMaxIterations(maxIterations);
    solver.setTolerance(tolerance);
    return solver.compute(matrix, eigenvalue_count);
}

// Inverse power method for finding smallest eigenvalue (now returns array)
double* EigenSolver::inversePowerMethod(const Matrix& matrix, int& eigenvalue_count, int maxIterations, double tolerance) {
    InversePowerMethodSolver solver;
    solver.setMaxIterations(maxIterations);
    solver.setTolerance(tolerance);
    return solver.compute(matrix, eigenvalue_count);
}

// Jacobi method for symmetric matrices
double* EigenSolver::jacobiMethod(const Matrix& matrix, int& eigenvalue_count, int maxIterations, double tolerance) {
    JacobiSolver solver;
    solver.setMaxIterations(maxIterations);
    solver.setTolerance(tolerance);
    solver.setParallelMode(false);
    return solver.compute(matrix, eigenvalue_count);
}

// Parallel Jacobi method
double* EigenSolver::jacobiMethodParallel(const Matrix& matrix, int& eigenvalue_count, int maxIterations, double tolerance) {
    JacobiSolver solver;
    solver.setMaxIterations(maxIterations);
    solver.setTolerance(tolerance);
    solver.setParallelMode(true);
    return solver.compute(matrix, eigenvalue_count);
}

// Divide and Conquer method for symmetric tridiagonal matrices
double* EigenSolver::divideAndConquer(const Matrix& matrix, int& eigenvalue_count, double tolerance) {
    DivideAndConquerSolver solver;
    solver.setTolerance(tolerance);
    return solver.compute(matrix, eigenvalue_count);
}

// Compute all eigenvalues using the best method based on matrix properties
double* EigenSolver::solve(const Matrix& matrix, int& eigenvalue_count, const std::string& method,
                          int maxIterations, double tolerance) {
    // Use factory to create solver
    std::unique_ptr<IEigenSolver> solver;

    if (method == "auto") {
        // Automatic selection based on matrix properties
        solver = SolverFactory::createAuto(matrix, tolerance);
    } else if (method == "qr") {
        solver = std::make_unique<QRSolver>();
    } else if (method == "qr_parallel") {
        solver = std::make_unique<QRSolver>();
        solver->setParallelMode(true);
    } else if (method == "jacobi") {
        solver = std::make_unique<JacobiSolver>();
    } else if (method == "jacobi_parallel") {
        solver = std::make_unique<JacobiSolver>();
        solver->setParallelMode(true);
    } else if (method == "power") {
        solver = std::make_unique<PowerMethodSolver>();
    } else if (method == "inverse_power") {
        solver = std::make_unique<InversePowerMethodSolver>();
    } else if (method == "divide_conquer") {
        solver = std::make_unique<DivideAndConquerSolver>();
    } else {
        throw std::invalid_argument("Unknown method: " + method);
    }

    // Configure solver
    solver->setMaxIterations(maxIterations);
    solver->setTolerance(tolerance);

    // Compute eigenvalues
    return solver->compute(matrix, eigenvalue_count);
}

// Compute eigenvalues and eigenvectors
std::pair<double*, Matrix> EigenSolver::solveWithEigenvectors(const Matrix& matrix,
                                                             int& eigenvalue_count,
                                                             const std::string& method,
                                                             int maxIterations,
                                                             double tolerance) {
    // For now, just return eigenvalues and an identity matrix for eigenvectors
    // A full implementation would track the transformation matrices during the algorithm
    double* eigenvals = solve(matrix, eigenvalue_count, method, maxIterations, tolerance);
    Matrix eigenvectors = Matrix::identity(matrix.getRows());

    return std::make_pair(eigenvals, eigenvectors);
}

// ========== Helper Functions (Forward to SolverUtils) ==========

bool EigenSolver::isSymmetric(const Matrix& matrix, double tolerance) {
    return SolverUtils::isSymmetric(matrix, tolerance);
}

bool EigenSolver::isTridiagonal(const Matrix& matrix, double tolerance) {
    return SolverUtils::isTridiagonal(matrix, tolerance);
}
