#include "../matrix/Matrix.h"
#include "../io/MatrixMarketIO.h"
#include "../solver/EigenSolver.h"
#include "../util/Utils.h"
#include <iostream>
#include <string>
#include <algorithm>
#include <omp.h>

void printUsage() {
    std::cout << "Usage: eigensolver [options]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  -f <filename>    : Input matrix file in Matrix Market format" << std::endl;
    std::cout << "  -s <size>        : Generate random matrix of given size" << std::endl;
    std::cout << "  -m <method>      : Eigenvalue computation method (qr, jacobi, auto)" << std::endl;
    std::cout << "  -t <tolerance>   : Convergence tolerance (default: 1e-10)" << std::endl;
    std::cout << "  -i <iterations>  : Maximum iterations (default: 1000)" << std::endl;
    std::cout << "  -h               : Show this help message" << std::endl;
}

int main(int argc, char* argv[]) {
    std::string filename = "";
    int matrix_size = 0;
    std::string method = "auto";
    double tolerance = 1e-10;
    int max_iterations = 1000;
    bool use_random = false;
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "-f" && i + 1 < argc) {
            filename = argv[++i];
        } else if (arg == "-s" && i + 1 < argc) {
            matrix_size = std::stoi(argv[++i]);
            use_random = true;
        } else if (arg == "-m" && i + 1 < argc) {
            method = argv[++i];
        } else if (arg == "-t" && i + 1 < argc) {
            tolerance = std::stod(argv[++i]);
        } else if (arg == "-i" && i + 1 < argc) {
            max_iterations = std::stoi(argv[++i]);
        } else if (arg == "-h") {
            printUsage();
            return 0;
        }
    }
    
    // Initialize OpenMP
    int num_threads = omp_get_max_threads();
    std::cout << "EigenSolver - Parallel Eigenvalue Solver" << std::endl;
    std::cout << "Using " << num_threads << " threads" << std::endl;
    std::cout << "OpenMP version: " << _OPENMP << std::endl;
    
    try {
        Matrix matrix;
        
        if (!filename.empty()) {
            // Read matrix from file
            std::cout << "Reading matrix from file: " << filename << std::endl;
            
            if (MatrixMarketIO::isValidMatrixMarketFile(filename)) {
                matrix = MatrixMarketIO::readMatrix(filename);
            } else {
                std::cerr << "Error: Invalid Matrix Market file format" << std::endl;
                return 1;
            }
        } else if (use_random && matrix_size > 0) {
            // Generate random symmetric matrix
            std::cout << "Generating random symmetric matrix of size " << matrix_size << "x" << matrix_size << std::endl;
            matrix = Utils::randomSymmetricMatrix(matrix_size);
        } else {
            // Default: create a small test matrix
            std::cout << "Using default 3x3 test matrix" << std::endl;
            matrix = Matrix(3, 3);
            matrix(0, 0) = 4.0; matrix(0, 1) = 1.0; matrix(0, 2) = 1.0;
            matrix(1, 0) = 1.0; matrix(1, 1) = 3.0; matrix(1, 2) = 2.0;
            matrix(2, 0) = 1.0; matrix(2, 1) = 2.0; matrix(2, 2) = 3.0;
        }
        
        std::cout << "Matrix size: " << matrix.getRows() << "x" << matrix.getCols() << std::endl;
        
        // Display matrix properties
        std::cout << "Matrix is symmetric: " << (Utils::isSymmetric(matrix) ? "Yes" : "No") << std::endl;
        std::cout << "Matrix Frobenius norm: " << Utils::frobeniusNorm(matrix) << std::endl;
        
        // Solve for eigenvalues
        std::cout << "Computing eigenvalues using method: " << method << std::endl;
        
        auto start_time = Utils::startTimer();
        int eigenvalue_count;
        double* eigenvalues = EigenSolver::solve(matrix, eigenvalue_count, method, max_iterations, tolerance);
        double computation_time = Utils::stopTimer(start_time);
        
        // Sort eigenvalues in descending order
        std::sort(eigenvalues, eigenvalues + eigenvalue_count, [](double a, double b) { return a > b; });
        
        // Display results
        std::cout << "\nComputed eigenvalues:" << std::endl;
        for (int i = 0; i < eigenvalue_count; ++i) {
            std::cout << "λ" << i+1 << " = " << eigenvalues[i] << std::endl;
        }
        
        // Verify results
        bool verification = Utils::verifyEigenvalues(matrix, eigenvalues, eigenvalue_count);
        std::cout << "\nEigenvalue verification (trace check): " << (verification ? "PASSED" : "FAILED") << std::endl;
        
        // Performance statistics
        Utils::printPerformanceStats(computation_time, "EigenSolver (" + method + ")", matrix);
        
        // Save results to file if needed
        if (!filename.empty()) {
            std::string output_filename = filename + ".eigenvals";
            std::ofstream output_file(output_filename);
            if (output_file.is_open()) {
                output_file << "# Eigenvalues computed by EigenSolver" << std::endl;
                output_file << "# Method: " << method << std::endl;
                output_file << "# Computation time: " << computation_time << " seconds" << std::endl;
                output_file << "# Number of eigenvalues: " << eigenvalue_count << std::endl;
                
                for (int i = 0; i < eigenvalue_count; ++i) {
                    output_file << eigenvalues[i] << std::endl;
                }
                
                output_file.close();
                std::cout << "Eigenvalues saved to: " << output_filename << std::endl;
            }
        }
        
        // Clean up allocated memory
        delete[] eigenvalues;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}