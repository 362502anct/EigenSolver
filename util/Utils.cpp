#include "Utils.h"
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <sstream>

// Timing utilities
std::chrono::high_resolution_clock::time_point Utils::startTimer() {
    return std::chrono::high_resolution_clock::now();
}

double Utils::stopTimer(const std::chrono::high_resolution_clock::time_point& start) {
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    return duration.count() / 1000000.0;  // Convert to seconds
}

// Matrix analysis utilities
double Utils::matrixNorm(const Matrix& matrix) {
    // Calculate the Frobenius norm by default
    return frobeniusNorm(matrix);
}

double Utils::frobeniusNorm(const Matrix& matrix) {
    // For sparse matrix, directly access the stored non-zero values
    double sum = 0.0;
    const double* values = matrix.getValues();
    int nnz = matrix.getNNZ();

    // Parallel computation for sparse matrices
    #pragma omp parallel for reduction(+:sum)
    for (int idx = 0; idx < nnz; ++idx) {
        sum += values[idx] * values[idx];
    }

    return std::sqrt(sum);
}

double Utils::spectralNorm(const Matrix& matrix) {
    // The spectral norm is the largest singular value
    // For this implementation, we'll use the largest eigenvalue of A^T * A
    // This is a simplified approach - a full SVD would be more accurate
    Matrix AT = matrix.transpose();
    Matrix ATA = AT.multiply(matrix);

    // Get eigenvalues of A^T * A
    double* eigenvals = ATA.eigenvalues();
    int n = ATA.getRows();

    // Find the maximum eigenvalue
    double max_eigenval = 0.0;
    for (int i = 0; i < n; ++i) {
        if (eigenvals[i] > max_eigenval) {
            max_eigenval = eigenvals[i];
        }
    }

    delete[] eigenvals;
    return std::sqrt(max_eigenval);
}

double Utils::conditionNumber(const Matrix& matrix) {
    // Condition number is the ratio of largest to smallest singular values
    // For this simplified implementation, we'll use the ratio of largest
    // to smallest eigenvalues (for symmetric matrices)
    double* eigenvals = matrix.eigenvalues();
    int eigenvalue_count = matrix.getRows();

    if (eigenvalue_count == 0) {
        return 0.0;
    }

    double max_val = std::abs(eigenvals[0]);
    double min_val = std::abs(eigenvals[0]);

    for (int i = 0; i < eigenvalue_count; ++i) {
        double abs_val = std::abs(eigenvals[i]);
        if (abs_val > max_val) max_val = abs_val;
        if (abs_val < min_val) min_val = abs_val;
    }

    if (min_val < 1e-15) {
        delete[] eigenvals;
        return 1e15;  // Matrix is nearly singular
    }

    delete[] eigenvals;
    return max_val / min_val;
}

// Verification utilities
bool Utils::verifyEigenvalues(const Matrix& matrix, const double* eigenvalues, int count) {
    if (matrix.getRows() != matrix.getCols()) {
        return false;
    }
    
    if (count != matrix.getRows()) {
        return false;
    }
    
    // A simple verification: trace of matrix should equal sum of eigenvalues
    double trace = 0.0;
    for (int i = 0; i < matrix.getRows(); ++i) {
        trace += matrix(i, i);
    }
    
    double eigen_sum = 0.0;
    for (int i = 0; i < count; ++i) {
        eigen_sum += eigenvalues[i];
    }
    
    return std::abs(trace - eigen_sum) < 1e-8;
}

bool Utils::verifyEigenvector(const Matrix& matrix, double eigenvalue, const Matrix& eigenvector, double tolerance) {
    if (matrix.getRows() != matrix.getCols() || eigenvector.getCols() != 1) {
        return false;
    }
    
    if (matrix.getRows() != eigenvector.getRows()) {
        return false;
    }
    
    // Check if A*v = lambda*v, or equivalently A*v - lambda*v = 0
    Matrix Av = matrix.multiply(eigenvector);
    Matrix lambda_v = eigenvector * eigenvalue;
    Matrix diff = Av - lambda_v;
    
    // Check if the difference is close to zero
    for (int i = 0; i < diff.getRows(); ++i) {
        if (std::abs(diff(i, 0)) > tolerance) {
            return false;
        }
    }
    
    return true;
}

// Matrix properties
bool Utils::isSymmetric(const Matrix& matrix, double tolerance) {
    if (matrix.getRows() != matrix.getCols()) {
        return false;
    }
    
    int n = matrix.getRows();
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            if (std::abs(matrix(i, j) - matrix(j, i)) > tolerance) {
                return false;
            }
        }
    }
    return true;
}

bool Utils::isPositiveDefinite(const Matrix& matrix) {
    if (!isSymmetric(matrix)) {
        return false;
    }

    // A matrix is positive definite if all eigenvalues are positive
    double* eigenvals = matrix.eigenvalues();
    int eigenvalue_count = matrix.getRows();

    for (int i = 0; i < eigenvalue_count; ++i) {
        if (eigenvals[i] <= 0) {
            delete[] eigenvals;
            return false;
        }
    }
    
    delete[] eigenvals;
    return true;
}

bool Utils::isOrthogonal(const Matrix& matrix, double tolerance) {
    if (matrix.getRows() != matrix.getCols()) {
        return false;
    }
    
    int n = matrix.getRows();
    
    // Check if A * A^T = I
    Matrix AT = matrix.transpose();
    Matrix product = matrix.multiply(AT);
    Matrix identity = Matrix::identity(n);
    
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (std::abs(product(i, j) - identity(i, j)) > tolerance) {
                return false;
            }
        }
    }
    
    return true;
}

int Utils::matrixRank(const Matrix& matrix, double tolerance) {
    // This is a simplified implementation
    // A full implementation would use SVD to determine rank
    // For now, we'll return a conservative estimate
    double* eigenvals = matrix.eigenvalues();
    int eigenvalue_count = matrix.getRows();

    int rank = 0;
    for (int i = 0; i < eigenvalue_count; ++i) {
        if (std::abs(eigenvals[i]) > tolerance) {
            rank++;
        }
    }

    delete[] eigenvals;
    return rank;
}

// Error computation
double Utils::relativeError(const Matrix& original, const Matrix& approximation) {
    if (original.getRows() != approximation.getRows() || 
        original.getCols() != approximation.getCols()) {
        return -1.0;  // Error: matrices have different dimensions
    }
    
    Matrix diff = original - approximation;
    
    double norm_original = frobeniusNorm(original);
    double norm_diff = frobeniusNorm(diff);
    
    if (norm_original < 1e-15) {
        return norm_diff;  // Original matrix is nearly zero
    }
    
    return norm_diff / norm_original;
}

double Utils::residualError(const Matrix& matrix, const double* eigenvalues, int count, const Matrix& eigenvectors) {
    // Calculate residual error for eigenvalue decomposition
    // For each eigenvalue-eigenvector pair, compute ||A*v - lambda*v||
    int n = matrix.getRows();
    if (count != n || eigenvectors.getRows() != n || eigenvectors.getCols() != n) {
        return -1.0;  // Error: incompatible dimensions
    }
    
    double max_residual = 0.0;
    
    for (int i = 0; i < n; ++i) {
        // Extract i-th eigenvector
        Matrix v(n, 1);
        for (int j = 0; j < n; ++j) {
            v.set(j, 0, eigenvectors(j, i));
        }
        
        // Compute A*v
        Matrix Av = matrix.multiply(v);
        
        // Compute lambda*v
        Matrix lambda_v = v * eigenvalues[i];
        
        // Compute residual
        Matrix residual = Av - lambda_v;
        double residual_norm = frobeniusNorm(residual);
        
        if (residual_norm > max_residual) {
            max_residual = residual_norm;
        }
    }
    
    return max_residual;
}

// Performance utilities
void Utils::printPerformanceStats(double time_taken, const std::string& algorithm_name, const Matrix& matrix) {
    std::cout << "=== Performance Report ===" << std::endl;
    std::cout << "Algorithm: " << algorithm_name << std::endl;
    std::cout << "Matrix size: " << matrix.getRows() << "x" << matrix.getCols() << std::endl;
    std::cout << "Time taken: " << formatTime(time_taken) << std::endl;
    std::cout << "=========================" << std::endl;
}

// Random utilities
Matrix Utils::randomSymmetricMatrix(int size, double min, double max) {
    Matrix matrix = Matrix::random(size, size, min, max);

    // Make it symmetric by averaging with its transpose
    Matrix AT = matrix.transpose();
    Matrix symmetric = (matrix + AT) * 0.5;
    return symmetric;
}

Matrix Utils::randomOrthogonalMatrix(int size) {
    // Create a random matrix and perform QR decomposition
    // The Q matrix from QR decomposition is orthogonal
    Matrix random_matrix = Matrix::random(size, size, -1.0, 1.0);
    
    // Perform QR decomposition (simplified approach)
    // In a full implementation, we would use a proper QR decomposition
    // For now, we'll use Gram-Schmidt process
    Matrix Q(size, size);
    
    for (int j = 0; j < size; ++j) {
        // Copy j-th column
        for (int i = 0; i < size; ++i) {
            Q.set(i, j, random_matrix(i, j));
        }

        // Orthogonalize against previous columns
        for (int i = 0; i < j; ++i) {
            double dot_product = 0.0;
            for (int k = 0; k < size; ++k) {
                dot_product += Q(k, j) * Q(k, i);
            }

            for (int k = 0; k < size; ++k) {
                Q.set(k, j, Q(k, j) - dot_product * Q(k, i));
            }
        }

        // Normalize
        double norm = 0.0;
        for (int i = 0; i < size; ++i) {
            norm += Q(i, j) * Q(i, j);
        }
        norm = std::sqrt(norm);

        if (norm > 1e-15) {
            for (int i = 0; i < size; ++i) {
                Q.set(i, j, Q(i, j) / norm);
            }
        }
    }
    
    return Q;
}

// String utilities for output
std::string Utils::formatTime(double seconds) {
    if (seconds < 1e-3) {
        return std::to_string(seconds * 1e6) + " μs";
    } else if (seconds < 1.0) {
        return std::to_string(seconds * 1e3) + " ms";
    } else {
        return std::to_string(seconds) + " s";
    }
}

std::string Utils::formatNumber(double number) {
    std::ostringstream oss;
    oss << std::scientific << std::setprecision(6) << number;
    return oss.str();
}