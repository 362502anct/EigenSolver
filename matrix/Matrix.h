#ifndef MATRIX_H
#define MATRIX_H

#include <iostream>
#include <utility>
#include <cblas.h>
#include <lapacke.h>
#include <omp.h>

class Matrix {
private:
    // Compressed Sparse Column (CSC) format
    double* values;      // Non-zero values
    int* row_indices;    // Row indices of non-zero values
    int* col_ptrs;       // Column pointers (size = cols + 1)
    int rows;
    int cols;
    int nnz;             // Number of non-zeros

    // Dense format (for compatibility with some operations)
    bool is_dense;
    double* dense_data;  // Only used when is_dense is true

public:
    // Constructors
    Matrix();
    Matrix(int rows, int cols);
    Matrix(int rows, int cols, const double* dense_data);  // From dense data
    Matrix(int rows, int cols, double* values, int* row_indices, int* col_ptrs, int nnz);  // From CSC data
    Matrix(const Matrix& other);
    
    // Destructor
    ~Matrix();
    
    // Assignment operator
    Matrix& operator=(const Matrix& other);

    // Swap function for copy-and-swap idiom
    void swap(Matrix& other) noexcept;
    // Getters
    int getRows() const { return rows; }
    int getCols() const { return cols; }
    int getNNZ() const { return nnz; }
    bool isDense() const { return is_dense; }

    // Access sparse matrix data (for efficient operations)
    const double* getValues() const { return values; }
    const int* getRowIndices() const { return row_indices; }
    const int* getColPtrs() const { return col_ptrs; }
    
    // Access elements (for dense matrices)
    double& operator()(int row, int col);
    double operator()(int row, int col) const;
    
    // Basic operations
    Matrix operator+(const Matrix& other) const;
    Matrix operator-(const Matrix& other) const;
    Matrix operator*(const Matrix& other) const;
    Matrix operator*(double scalar) const;
    
    // Matrix operations
    Matrix transpose() const;
    Matrix multiply(const Matrix& other) const;  // Using BLAS/Sparse operations
    Matrix multiply_parallel(const Matrix& other) const;  // Parallel multiplication
    
    // Convert between dense and sparse
    void toDense(double* result) const;
    void toSparse(const double* dense_data, int rows, int cols);
    Matrix toDenseMatrix() const;
    Matrix toSparseMatrix() const;
    
    // LAPACK operations (for dense matrices)
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
    
    // BLAS/LAPACK wrapper functions (for dense matrices)
    void syev_wrapper(double* eigenvalues, Matrix& eigenvectors) const;
    void geev_wrapper(double* real_eigenvals, double* imag_eigenvals,
                     Matrix& left_eigenvectors, Matrix& right_eigenvectors) const;
};

// Scalar multiplication (scalar * matrix)
Matrix operator*(double scalar, const Matrix& matrix);

#endif // MATRIX_H