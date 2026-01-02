#ifndef MATRIX_H
#define MATRIX_H

#include <iostream>
#include <utility>
#include <cblas.h>
#include <lapacke.h>
#include <omp.h>

// Forward declaration for SIMD dispatcher
class SIMDDispatcher;

class Matrix {

public:
    // Constructors
    Matrix();
    Matrix(int rows, int cols);
    Matrix(int rows, int cols, double* values, int* row_indices, int* col_ptrs, int nnz);  // From CSC data
    Matrix(const Matrix& other);
    Matrix(Matrix&& other) noexcept;  // Move constructor

    // Destructor (virtual for inheritance support)
    virtual ~Matrix();

    // Assignment operators
    Matrix& operator=(const Matrix& other);
    Matrix& operator=(Matrix&& other) noexcept;  // Move assignment

    // Swap function for copy-and-swap idiom
    void swap(Matrix& other) noexcept;
    // Getters
    int getRows() const { return rows; }
    int getCols() const { return cols; }
    int getNNZ() const { return nnz; }

    // Symmetric matrix support
    bool isSymmetric() const { return is_symmetric; }
    void setSymmetric(bool symmetric) { is_symmetric = symmetric; }
    bool checkSymmetry(double tolerance = 1e-10) const;

    // Access sparse matrix data (for efficient operations)
    const double* getValues() const { return values; }
    const int* getRowIndices() const { return row_indices; }
    const int* getColPtrs() const { return col_ptrs; }

    // Access elements (read-only for sparse matrices)
    virtual double operator()(int row, int col) const;
    virtual void set(int row, int col, double value);  // Set element value (virtual for inheritance)
    
    // Basic operations
    Matrix operator+(const Matrix& other) const;
    Matrix operator-(const Matrix& other) const;
    Matrix operator*(const Matrix& other) const;
    Matrix operator*(double scalar) const;
    
    // Matrix operations
    Matrix transpose() const;
    Matrix multiply(const Matrix& other) const;  // Using BLAS/Sparse operations
    Matrix multiply_parallel(const Matrix& other) const;  // Parallel multiplication
    Matrix multiply_dense_simd(const Matrix& other) const;  // SIMD-optimized dense multiplication
    
    // Convert to dense format (for LAPACK operations)
    virtual void toDense(double* result) const;
    
    // LAPACK operations (converts to dense internally)
    double* eigenvalues() const;  // Compute eigenvalues using LAPACK
    std::pair<double*, Matrix> eigensystem() const;  // Compute eigenvalues and eigenvectors
    
    // Utility functions
    void print() const;
    void fill(double value);
    void randomize(double min = -1.0, double max = 1.0);
    
    // Static utility functions
    static Matrix identity(int size);
    static Matrix zeros(int rows, int cols);
    static Matrix ones(int rows, int cols);
    static Matrix random(int rows, int cols, double min = -1.0, double max = 1.0);
    static Matrix randomSymmetric(int size, double min = -1.0, double max = 1.0);
    static Matrix randomSparseSymmetric(int size, double density = 0.3, double min = -1.0, double max = 1.0);
    
    // BLAS/LAPACK wrapper functions (for dense matrices)
    void syev_wrapper(double* eigenvalues, Matrix& eigenvectors) const;
    void geev_wrapper(double* real_eigenvals, double* imag_eigenvals,
                     Matrix& left_eigenvectors, Matrix& right_eigenvectors) const;

private:
    // Compressed Sparse Column (CSC) format
    double* values;      // Non-zero values
    int* row_indices;    // Row indices of non-zero values
    int* col_ptrs;       // Column pointers (size = cols + 1)
    int rows;
    int cols;
    int nnz;             // Number of non-zeros
    bool is_symmetric = false;  // Symmetric matrix flag

};

// Scalar multiplication (scalar * matrix)
Matrix operator*(double scalar, const Matrix& matrix);

#endif // MATRIX_H