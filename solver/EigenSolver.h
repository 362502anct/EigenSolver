#ifndef EIGENSOLVER_H
#define EIGENSOLVER_H

#include "../matrix/Matrix.h"
#include "../util/Utils.h"
#include <vector>
#include <functional>

class EigenSolver {
public:
    // QR Algorithm with shifts for eigenvalue computation
    static double* qrAlgorithm(const Matrix& matrix, int& eigenvalue_count, int maxIterations = 1000, double tolerance = 1e-10);
    
    // Parallel QR Algorithm implementation
    static double* qrAlgorithmParallel(const Matrix& matrix, int& eigenvalue_count, int maxIterations = 1000, double tolerance = 1e-10);
    
    // Power method for finding dominant eigenvalue
    static double powerMethod(const Matrix& matrix, int maxIterations = 1000, double tolerance = 1e-10);
    
    // Inverse power method for finding smallest eigenvalue
    static double inversePowerMethod(const Matrix& matrix, int maxIterations = 1000, double tolerance = 1e-10);
    
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
    
private:
    // Helper functions for QR decomposition
    static Matrix qrDecomposition(const Matrix& matrix, Matrix& Q, Matrix& R);
    static Matrix qrDecompositionParallel(const Matrix& matrix, Matrix& Q, Matrix& R);
    
    // Helper function to check if matrix is symmetric
    static bool isSymmetric(const Matrix& matrix, double tolerance = 1e-10);
    
    // Helper function to check if matrix is tridiagonal
    static bool isTridiagonal(const Matrix& matrix, double tolerance = 1e-10);
    
    // Householder transformation for tridiagonalization
    static std::pair<Matrix, Matrix> householderTridiagonalize(const Matrix& matrix);
    
    // Givens rotation for QR decomposition
    static void givensRotate(Matrix& matrix, int i, int j, double cos_val, double sin_val);
};

#endif // EIGENSOLVER_H