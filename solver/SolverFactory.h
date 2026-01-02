#ifndef SOLVERFACTORY_H
#define SOLVERFACTORY_H

#include "IEigenSolver.h"
#include "QRSolver.h"
#include "JacobiSolver.h"
#include "PowerMethodSolver.h"
#include "InversePowerMethodSolver.h"
#include "DivideAndConquerSolver.h"
#include "SolverUtils.h"
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <functional>

/**
 * Factory class for creating eigenvalue solver instances
 * Supports algorithm selection by name and automatic algorithm selection
 */
class SolverFactory {
public:
    /**
     * Create a solver instance by algorithm name
     * @param method Algorithm name (qr, jacobi, power, inverse_power, divide_conquer, auto)
     * @return Unique pointer to solver instance
     * @throws std::invalid_argument if method name is unknown
     */
    static std::unique_ptr<IEigenSolver> create(const std::string& method) {
        if (method == "qr" || method == "qr_algorithm") {
            return std::make_unique<QRSolver>();
        } else if (method == "jacobi" || method == "jacobi_method") {
            return std::make_unique<JacobiSolver>();
        } else if (method == "power" || method == "power_method") {
            return std::make_unique<PowerMethodSolver>();
        } else if (method == "inverse_power" || method == "inverse_power_method") {
            return std::make_unique<InversePowerMethodSolver>();
        } else if (method == "divide_conquer" || method == "divide_and_conquer") {
            return std::make_unique<DivideAndConquerSolver>();
        } else if (method == "auto") {
            throw std::invalid_argument("Use createAuto() for automatic method selection");
        } else {
            throw std::invalid_argument("Unknown method: " + method +
                                     "\nSupported methods: qr, jacobi, power, inverse_power, divide_conquer, auto");
        }
    }

    /**
     * Create a solver instance with automatic algorithm selection based on matrix properties
     * Selection strategy:
     * - Symmetric tridiagonal -> Divide and Conquer (fastest)
     * - Symmetric -> Jacobi Method (parallel)
     * - General square -> QR Algorithm (parallel)
     *
     * @param matrix Input matrix for analysis
     * @param tolerance Tolerance for matrix property checks (default: 1e-10)
     * @return Unique pointer to solver instance
     */
    static std::unique_ptr<IEigenSolver> createAuto(const Matrix& matrix, double tolerance = 1e-10) {
        std::unique_ptr<IEigenSolver> solver;

        // Check matrix properties
        bool symmetric = SolverUtils::isSymmetric(matrix, tolerance);
        bool tridiagonal = symmetric && SolverUtils::isTridiagonal(matrix, tolerance);

        // Select algorithm based on properties
        if (tridiagonal) {
            // Best for symmetric tridiagonal matrices
            solver = std::make_unique<DivideAndConquerSolver>();
        } else if (symmetric) {
            // Jacobi is highly parallelizable for symmetric matrices
            solver = std::make_unique<JacobiSolver>();
            solver->setParallelMode(true);  // Enable parallel mode
        } else {
            // QR algorithm is general-purpose
            solver = std::make_unique<QRSolver>();
            solver->setParallelMode(true);  // Enable parallel mode
        }

        return solver;
    }

    /**
     * Create a solver instance by method name with automatic parallel mode selection
     * Parallel variants are automatically selected if "parallel" is in the method name
     *
     * @param method Algorithm name (supports qr, jacobi, qr_parallel, jacobi_parallel, etc.)
     * @return Unique pointer to solver instance
     */
    static std::unique_ptr<IEigenSolver> createWithParallel(const std::string& method) {
        // Check if parallel mode is requested
        bool parallel = (method.find("parallel") != std::string::npos ||
                        method.find("_parallel") != std::string::npos);

        // Extract base method name
        std::string base_method = method;
        size_t parallel_pos = base_method.find("_parallel");
        if (parallel_pos != std::string::npos) {
            base_method = base_method.substr(0, parallel_pos);
        }

        // Create solver
        auto solver = create(base_method);

        // Set parallel mode if requested
        if (parallel) {
            solver->setParallelMode(true);
        }

        return solver;
    }

    /**
     * Get list of supported algorithm names
     * @return Vector of algorithm names
     */
    static std::vector<std::string> getSupportedMethods() {
        return {
            "qr",
            "jacobi",
            "power",
            "inverse_power",
            "divide_conquer",
            "qr_parallel",
            "jacobi_parallel",
            "auto"
        };
    }

    /**
     * Check if a method name is supported
     * @param method Algorithm name to check
     * @return True if method is supported
     */
    static bool isMethodSupported(const std::string& method) {
        static const std::vector<std::string> methods = getSupportedMethods();
        return std::find(methods.begin(), methods.end(), method) != methods.end();
    }
};

#endif // SOLVERFACTORY_H
