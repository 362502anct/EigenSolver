#include "EigenSolver.h"
#include <cmath>
#include <algorithm>
#include <random>
#include <omp.h>

// QR Algorithm with shifts for eigenvalue computation
double* EigenSolver::qrAlgorithm(const Matrix& matrix, int& eigenvalue_count, int maxIterations, double tolerance) {
    if (matrix.getRows() != matrix.getCols()) {
        throw std::invalid_argument("Matrix must be square for eigenvalue computation");
    }
    
    Matrix A = matrix.toDenseMatrix();  // Convert to dense for algorithm
    int n = A.getRows();
    eigenvalue_count = n;
    double* eigenvalues = new double[n];
    
    for (int iter = 0; iter < maxIterations; ++iter) {
        // Find shift (using Wilkinson shift for better convergence)
        // Note: Shift variable reserved for future implementation
        // For now, using basic QR without shift
        [[maybe_unused]] double shift = 0.0;
        if (iter > 0 && std::abs(A(n-1, n-2)) < tolerance) {
            // Deflation: last eigenvalue found
            eigenvalues[n-1] = A(n-1, n-1);
            // For simplicity, we'll just continue with the same size matrix
            // A proper implementation would reduce the matrix size
            n--;
            if (n <= 1) break;
            continue;
        }
        
        // Perform QR decomposition with shift
        Matrix Q(n, n), R(n, n);
        qrDecomposition(A, Q, R);
        A = R.multiply(Q);  // Update A = R * Q
        
        // Check for convergence (superdiagonal elements should be small)
        bool converged = true;
        for (int i = 1; i < n; ++i) {
            if (std::abs(A(i, i-1)) > tolerance) {
                converged = false;
                break;
            }
        }
        
        if (converged) {
            for (int i = 0; i < n; ++i) {
                eigenvalues[i] = A(i, i);
            }
            break;
        }
    }
    
    // Fill in any remaining eigenvalues if algorithm didn't fully converge
    for (int i = 0; i < n; ++i) {
        eigenvalues[i] = A(i, i);
    }
    
    return eigenvalues;
}

// Parallel QR Algorithm implementation
double* EigenSolver::qrAlgorithmParallel(const Matrix& matrix, int& eigenvalue_count, int maxIterations, double tolerance) {
    if (matrix.getRows() != matrix.getCols()) {
        throw std::invalid_argument("Matrix must be square for eigenvalue computation");
    }
    
    Matrix A = matrix.toDenseMatrix();  // Convert to dense for algorithm
    int n = A.getRows();
    eigenvalue_count = n;
    double* eigenvalues = new double[n];
    
    for (int iter = 0; iter < maxIterations; ++iter) {
        // Perform QR decomposition with shift in parallel
        Matrix Q(n, n), R(n, n);
        qrDecompositionParallel(A, Q, R);
        A = R.multiply_parallel(Q);  // Update A = R * Q
        
        // Check for convergence using parallel reduction
        bool converged = true;
        #pragma omp parallel
        {
            bool local_converged = true;
            #pragma omp for nowait
            for (int i = 1; i < n; ++i) {
                if (std::abs(A(i, i-1)) > tolerance) {
                    local_converged = false;
                }
            }
            #pragma omp critical
            {
                if (!local_converged) {
                    converged = false;
                }
            }
        }
        
        if (converged) {
            #pragma omp parallel for
            for (int i = 0; i < n; ++i) {
                eigenvalues[i] = A(i, i);
            }
            break;
        }
    }
    
    // Fill in any remaining eigenvalues if algorithm didn't fully converge
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        eigenvalues[i] = A(i, i);
    }
    
    return eigenvalues;
}

// Power method for finding dominant eigenvalue
double EigenSolver::powerMethod(const Matrix& matrix, int maxIterations, double tolerance) {
    if (matrix.getRows() != matrix.getCols()) {
        throw std::invalid_argument("Matrix must be square for eigenvalue computation");
    }
    
    int n = matrix.getRows();
    
    // Initialize random vector
    Matrix v(n, 1);
    v.randomize(-1.0, 1.0);
    
    // Normalize initial vector
    double norm = 0.0;
    for (int i = 0; i < n; ++i) {
        norm += v(i, 0) * v(i, 0);
    }
    norm = std::sqrt(norm);
    if (norm > 0) {
        v = v * (1.0 / norm);
    }
    
    double eigenvalue = 0.0;
    double prev_eigenvalue = 0.0;
    
    for (int iter = 0; iter < maxIterations; ++iter) {
        Matrix w = matrix.multiply(v);  // Matrix-vector multiplication
        
        // Calculate Rayleigh quotient (eigenvalue estimate)
        double numerator = 0.0, denominator = 0.0;
        for (int i = 0; i < n; ++i) {
            numerator += v(i, 0) * w(i, 0);
            denominator += v(i, 0) * v(i, 0);
        }
        
        if (std::abs(denominator) > tolerance) {
            eigenvalue = numerator / denominator;
        }
        
        // Normalize w
        norm = 0.0;
        for (int i = 0; i < n; ++i) {
            norm += w(i, 0) * w(i, 0);
        }
        norm = std::sqrt(norm);
        if (norm > tolerance) {
            v = w * (1.0 / norm);
        }
        
        // Check for convergence
        if (iter > 0 && std::abs(eigenvalue - prev_eigenvalue) < tolerance) {
            break;
        }
        prev_eigenvalue = eigenvalue;
    }
    
    return eigenvalue;
}

// Inverse power method for finding smallest eigenvalue
double EigenSolver::inversePowerMethod(const Matrix& matrix, int maxIterations, double tolerance) {
    if (matrix.getRows() != matrix.getCols()) {
        throw std::invalid_argument("Matrix must be square for eigenvalue computation");
    }
    
    // For inverse power method, we need to solve (A - shift*I)x = b at each iteration
    // This is a simplified version - in practice, you'd want to factorize A once
    Matrix A = matrix.toDenseMatrix();
    int n = A.getRows();
    
    // Initialize random vector
    Matrix v(n, 1);
    v.randomize(-1.0, 1.0);
    
    // Normalize initial vector
    double norm = 0.0;
    for (int i = 0; i < n; ++i) {
        norm += v(i, 0) * v(i, 0);
    }
    norm = std::sqrt(norm);
    if (norm > 0) {
        v = v * (1.0 / norm);
    }
    
    double eigenvalue = 0.0;
    
    for (int iter = 0; iter < maxIterations; ++iter) {
        // Solve A * w = v (equivalent to w = A^(-1) * v)
        // This is a simplified implementation - in practice, use LU decomposition
        Matrix A_inv = A.toDenseMatrix(); // In a real implementation, you'd compute the inverse
        Matrix w = A_inv.multiply(v);
        
        // Calculate Rayleigh quotient
        double numerator = 0.0, denominator = 0.0;
        for (int i = 0; i < n; ++i) {
            numerator += v(i, 0) * w(i, 0);
            denominator += v(i, 0) * v(i, 0);
        }
        
        if (std::abs(denominator) > tolerance) {
            eigenvalue = numerator / denominator;
        }
        
        // Take reciprocal since we're finding eigenvalue of A^(-1)
        if (std::abs(eigenvalue) > tolerance) {
            eigenvalue = 1.0 / eigenvalue;
        }
        
        // Normalize w
        norm = 0.0;
        for (int i = 0; i < n; ++i) {
            norm += w(i, 0) * w(i, 0);
        }
        norm = std::sqrt(norm);
        if (norm > tolerance) {
            v = w * (1.0 / norm);
        }
    }
    
    return eigenvalue;
}

// Jacobi method for symmetric matrices
double* EigenSolver::jacobiMethod(const Matrix& matrix, int& eigenvalue_count, int maxIterations, double tolerance) {
    if (matrix.getRows() != matrix.getCols()) {
        throw std::invalid_argument("Matrix must be square for eigenvalue computation");
    }
    
    if (!isSymmetric(matrix, tolerance)) {
        throw std::invalid_argument("Jacobi method requires a symmetric matrix");
    }
    
    Matrix A = matrix.toDenseMatrix();  // Work with a copy
    int n = A.getRows();
    eigenvalue_count = n;
    double* eigenvalues = new double[n];
    
    for (int iter = 0; iter < maxIterations; ++iter) {
        // Find the largest off-diagonal element
        int p = 0, q = 1;
        double max_val = std::abs(A(p, q));
        
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                if (std::abs(A(i, j)) > max_val) {
                    max_val = std::abs(A(i, j));
                    p = i;
                    q = j;
                }
            }
        }
        
        // Check for convergence
        if (max_val < tolerance) {
            break;
        }
        
        // Calculate rotation angle
        double theta = 0.5 * std::atan2(2 * A(p, q), A(q, q) - A(p, p));
        double c = std::cos(theta);
        double s = std::sin(theta);
        
        // Apply Jacobi rotation: A = J^T * A * J
        // Create rotation matrix and apply transformation
        Matrix A_new = A;
        
        // Zero out the p,q and q,p elements
        A_new(p, q) = 0.0;
        A_new(q, p) = 0.0;
        
        // Update the p-th and q-th rows and columns
        for (int i = 0; i < n; ++i) {
            if (i != p && i != q) {
                double temp_p = c * A(i, p) - s * A(i, q);
                double temp_q = s * A(i, p) + c * A(i, q);
                A_new(i, p) = temp_p;
                A_new(p, i) = temp_p;  // Maintain symmetry
                A_new(i, q) = temp_q;
                A_new(q, i) = temp_q;  // Maintain symmetry
            }
        }
        
        // Update diagonal elements
        A_new(p, p) = c*c*A(p, p) - 2*s*c*A(p, q) + s*s*A(q, q);
        A_new(q, q) = s*s*A(p, p) + 2*s*c*A(p, q) + c*c*A(q, q);
        
        A = A_new;
    }
    
    // Extract eigenvalues from diagonal
    for (int i = 0; i < n; ++i) {
        eigenvalues[i] = A(i, i);
    }
    
    return eigenvalues;
}

// Parallel Jacobi method
double* EigenSolver::jacobiMethodParallel(const Matrix& matrix, int& eigenvalue_count, int maxIterations, double tolerance) {
    if (matrix.getRows() != matrix.getCols()) {
        throw std::invalid_argument("Matrix must be square for eigenvalue computation");
    }
    
    if (!isSymmetric(matrix, tolerance)) {
        throw std::invalid_argument("Jacobi method requires a symmetric matrix");
    }
    
    Matrix A = matrix.toDenseMatrix();  // Work with a copy
    int n = A.getRows();
    eigenvalue_count = n;
    double* eigenvalues = new double[n];
    
    for (int iter = 0; iter < maxIterations; ++iter) {
        // Find the largest off-diagonal element
        int p = 0, q = 1;
        double max_val = 0.0;

        // Serial search for maximum element (not a performance bottleneck)
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                double abs_val = std::abs(A(i, j));
                if (abs_val > max_val) {
                    max_val = abs_val;
                    p = i;
                    q = j;
                }
            }
        }

        // Check for convergence
        if (max_val < tolerance) {
            break;
        }
        
        // Calculate rotation angle
        double theta = 0.5 * std::atan2(2 * A(p, q), A(q, q) - A(p, p));
        double c = std::cos(theta);
        double s = std::sin(theta);
        
        // Apply Jacobi rotation in parallel
        Matrix A_new = A;
        
        // Zero out the p,q and q,p elements
        A_new(p, q) = 0.0;
        A_new(q, p) = 0.0;
        
        // Update the p-th and q-th rows and columns in parallel
        #pragma omp parallel sections
        {
            #pragma omp section
            {
                // Update row/column p
                for (int i = 0; i < n; ++i) {
                    if (i != p && i != q) {
                        double temp_p = c * A(i, p) - s * A(i, q);
                        A_new(i, p) = temp_p;
                        A_new(p, i) = temp_p;  // Maintain symmetry
                    }
                }
            }
            #pragma omp section
            {
                // Update row/column q
                for (int i = 0; i < n; ++i) {
                    if (i != p && i != q) {
                        double temp_q = s * A(i, p) + c * A(i, q);
                        A_new(i, q) = temp_q;
                        A_new(q, i) = temp_q;  // Maintain symmetry
                    }
                }
            }
            #pragma omp section
            {
                // Update diagonal elements
                A_new(p, p) = c*c*A(p, p) - 2*s*c*A(p, q) + s*s*A(q, q);
                A_new(q, q) = s*s*A(p, p) + 2*s*c*A(p, q) + c*c*A(q, q);
            }
        }
        
        A = A_new;
    }
    
    // Extract eigenvalues from diagonal in parallel
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        eigenvalues[i] = A(i, i);
    }
    
    return eigenvalues;
}

// Helper function: QR Decomposition
Matrix EigenSolver::qrDecomposition(const Matrix& matrix, Matrix& Q, Matrix& R) {
    int m = matrix.getRows();
    int n = matrix.getCols();
    
    Q = Matrix::zeros(m, n);
    R = Matrix::zeros(n, n);
    
    // Gram-Schmidt process
    Matrix A = matrix.toDenseMatrix();
    
    for (int j = 0; j < n; ++j) {
        // Start with the j-th column of A
        for (int i = 0; i < m; ++i) {
            Q(i, j) = A(i, j);
        }
        
        // Subtract projections onto previous vectors
        for (int i = 0; i < j; ++i) {
            double dot_product = 0.0;
            for (int k = 0; k < m; ++k) {
                dot_product += Q(k, j) * Q(k, i);
            }
            
            for (int k = 0; k < m; ++k) {
                Q(k, j) -= dot_product * Q(k, i);
            }
        }
        
        // Calculate norm and normalize
        double norm = 0.0;
        for (int i = 0; i < m; ++i) {
            norm += Q(i, j) * Q(i, j);
        }
        norm = std::sqrt(norm);
        
        R(j, j) = norm;
        
        if (norm > 1e-15) {
            for (int i = 0; i < m; ++i) {
                Q(i, j) /= norm;
            }
        }
        
        // Calculate R entries
        for (int i = 0; i < j; ++i) {
            double dot_product = 0.0;
            for (int k = 0; k < m; ++k) {
                dot_product += A(k, j) * Q(k, i);
            }
            R(i, j) = dot_product;
        }
    }
    
    return Q;
}

// Parallel QR Decomposition
Matrix EigenSolver::qrDecompositionParallel(const Matrix& matrix, Matrix& Q, Matrix& R) {
    int m = matrix.getRows();
    int n = matrix.getCols();
    
    Q = Matrix::zeros(m, n);
    R = Matrix::zeros(n, n);
    
    Matrix A = matrix.toDenseMatrix();
    
    for (int j = 0; j < n; ++j) {
        // Start with the j-th column of A
        #pragma omp parallel for
        for (int i = 0; i < m; ++i) {
            Q(i, j) = A(i, j);
        }
        
        // Subtract projections onto previous vectors
        for (int i = 0; i < j; ++i) {
            double dot_product = 0.0;
            #pragma omp parallel for reduction(+:dot_product)
            for (int k = 0; k < m; ++k) {
                dot_product += Q(k, j) * Q(k, i);
            }
            
            #pragma omp parallel for
            for (int k = 0; k < m; ++k) {
                Q(k, j) -= dot_product * Q(k, i);
            }
        }
        
        // Calculate norm and normalize
        double norm = 0.0;
        #pragma omp parallel for reduction(+:norm)
        for (int i = 0; i < m; ++i) {
            norm += Q(i, j) * Q(i, j);
        }
        norm = std::sqrt(norm);
        
        R(j, j) = norm;
        
        if (norm > 1e-15) {
            #pragma omp parallel for
            for (int i = 0; i < m; ++i) {
                Q(i, j) /= norm;
            }
        }
        
        // Calculate R entries
        #pragma omp parallel for
        for (int i = 0; i < j; ++i) {
            double dot_product = 0.0;
            #pragma omp parallel for reduction(+:dot_product)
            for (int k = 0; k < m; ++k) {
                dot_product += A(k, j) * Q(k, i);
            }
            R(i, j) = dot_product;
        }
    }
    
    return Q;
}

// Helper function: Check if matrix is symmetric
bool EigenSolver::isSymmetric(const Matrix& matrix, double tolerance) {
    int n = matrix.getRows();
    if (n != matrix.getCols()) {
        return false;
    }
    
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            if (std::abs(matrix(i, j) - matrix(j, i)) > tolerance) {
                return false;
            }
        }
    }
    return true;
}

// Helper function: Check if matrix is tridiagonal
bool EigenSolver::isTridiagonal(const Matrix& matrix, double tolerance) {
    int n = matrix.getRows();
    if (n != matrix.getCols()) {
        return false;
    }
    
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (std::abs(i - j) > 1 && std::abs(matrix(i, j)) > tolerance) {
                return false;
            }
        }
    }
    return true;
}

// Compute all eigenvalues using the best method based on matrix properties
double* EigenSolver::solve(const Matrix& matrix, int& eigenvalue_count, const std::string& method, 
                          int maxIterations, double tolerance) {
    if (method == "qr") {
        return qrAlgorithm(matrix, eigenvalue_count, maxIterations, tolerance);
    } else if (method == "qr_parallel") {
        return qrAlgorithmParallel(matrix, eigenvalue_count, maxIterations, tolerance);
    } else if (method == "jacobi") {
        return jacobiMethod(matrix, eigenvalue_count, maxIterations, tolerance);
    } else if (method == "jacobi_parallel") {
        return jacobiMethodParallel(matrix, eigenvalue_count, maxIterations, tolerance);
    } else if (method == "auto") {
        // Choose the best method based on matrix properties
        if (isSymmetric(matrix, tolerance)) {
            if (isTridiagonal(matrix, tolerance)) {
                return divideAndConquer(matrix, eigenvalue_count, tolerance);
            } else {
                return jacobiMethodParallel(matrix, eigenvalue_count, maxIterations, tolerance);
            }
        } else {
            return qrAlgorithmParallel(matrix, eigenvalue_count, maxIterations, tolerance);
        }
    } else {
        throw std::invalid_argument("Unknown method: " + method);
    }
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

// Divide and Conquer method for symmetric tridiagonal matrices
double* EigenSolver::divideAndConquer(const Matrix& matrix, int& eigenvalue_count, double tolerance) {
    // This is a simplified placeholder - a full implementation would be more complex
    // For now, we'll fall back to the Jacobi method for symmetric matrices
    return jacobiMethod(matrix, eigenvalue_count, 1000, tolerance);
}