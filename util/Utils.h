#ifndef UTILS_H
#define UTILS_H

#include "../matrix/Matrix.h"
#include <string>
#include <chrono>

class Utils {
public:
    // Timing utilities
    static std::chrono::high_resolution_clock::time_point startTimer();
    static double stopTimer(const std::chrono::high_resolution_clock::time_point& start);
    
    // Matrix analysis utilities
    static double matrixNorm(const Matrix& matrix);
    static double frobeniusNorm(const Matrix& matrix);
    static double spectralNorm(const Matrix& matrix);
    static double conditionNumber(const Matrix& matrix);
    
    // Verification utilities
    static bool verifyEigenvalues(const Matrix& matrix, const double* eigenvalues, int count);
    static bool verifyEigenvector(const Matrix& matrix, double eigenvalue, const Matrix& eigenvector, double tolerance = 1e-10);
    
    // Matrix properties
    static bool isSymmetric(const Matrix& matrix, double tolerance = 1e-10);
    static bool isPositiveDefinite(const Matrix& matrix);
    static bool isOrthogonal(const Matrix& matrix, double tolerance = 1e-10);
    static int matrixRank(const Matrix& matrix, double tolerance = 1e-10);
    
    // Error computation
    static double relativeError(const Matrix& original, const Matrix& approximation);
    static double residualError(const Matrix& matrix, const double* eigenvalues, int count, const Matrix& eigenvectors);
    
    // Performance utilities
    static void printPerformanceStats(double time_taken, const std::string& algorithm_name, const Matrix& matrix);
    
    // Random utilities
    static Matrix randomSymmetricMatrix(int size, double min = -1.0, double max = 1.0);
    static Matrix randomOrthogonalMatrix(int size);
    
    // String utilities for output
    static std::string formatTime(double seconds);
    static std::string formatNumber(double number);
};

#endif // UTILS_H