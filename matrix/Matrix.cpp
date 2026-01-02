#include "Matrix.h"
#include <stdexcept>
#include <random>
#include <algorithm>
#include <cmath>
#include <cstring>

// Constructors
Matrix::Matrix() : values(nullptr), row_indices(nullptr), col_ptrs(nullptr), rows(0), cols(0), nnz(0), is_dense(true), dense_data(nullptr) {}

Matrix::Matrix(int r, int c) : values(nullptr), row_indices(nullptr), col_ptrs(nullptr), rows(r), cols(c), nnz(r*c), is_dense(true), dense_data(nullptr) {
    if (r < 0 || c < 0) {
        throw std::invalid_argument("Matrix dimensions must be non-negative");
    }

    dense_data = new double[rows * cols];

    // Initialize to zero
    for (int i = 0; i < rows * cols; ++i) {
        dense_data[i] = 0.0;
    }
}

Matrix::Matrix(int rows, int cols, const double* input_data) : values(nullptr), row_indices(nullptr), col_ptrs(nullptr), rows(rows), cols(cols), nnz(rows*cols), is_dense(true), dense_data(nullptr) {
    if (rows < 0 || cols < 0) {
        throw std::invalid_argument("Matrix dimensions must be non-negative");
    }
    
    dense_data = new double[rows * cols];
    nnz = rows * cols;
    
    // Copy the input data
    std::memcpy(dense_data, input_data, rows * cols * sizeof(double));
}

Matrix::Matrix(int rows, int cols, double* vals, int* row_idxs, int* col_ptrs_data, int non_zeros)
    : values(nullptr), row_indices(nullptr), col_ptrs(nullptr), rows(rows), cols(cols), nnz(non_zeros), is_dense(false), dense_data(nullptr) {
    values = new double[nnz];
    row_indices = new int[nnz];
    col_ptrs = new int[cols + 1];

    std::memcpy(values, vals, nnz * sizeof(double));
    std::memcpy(row_indices, row_idxs, nnz * sizeof(int));
    std::memcpy(col_ptrs, col_ptrs_data, (cols + 1) * sizeof(int));
}

Matrix::Matrix(const Matrix& other) : values(nullptr), row_indices(nullptr), col_ptrs(nullptr), rows(other.rows), cols(other.cols), nnz(other.nnz), is_dense(other.is_dense), dense_data(nullptr) {
    if (other.is_dense) {
        dense_data = new double[rows * cols];
        std::memcpy(dense_data, other.dense_data, rows * cols * sizeof(double));
        values = nullptr;
        row_indices = nullptr;
        col_ptrs = nullptr;
    } else {
        values = new double[nnz];
        row_indices = new int[nnz];
        col_ptrs = new int[cols + 1];
        
        std::memcpy(values, other.values, nnz * sizeof(double));
        std::memcpy(row_indices, other.row_indices, nnz * sizeof(int));
        std::memcpy(col_ptrs, other.col_ptrs, (cols + 1) * sizeof(int));
        
        dense_data = nullptr;
    }
}

// Destructor
Matrix::~Matrix() {
    if (is_dense && dense_data) {
        delete[] dense_data;
    } else {
        if (values) delete[] values;
        if (row_indices) delete[] row_indices;
        if (col_ptrs) delete[] col_ptrs;
    }
}

// Swap function for copy-and-swap idiom
void Matrix::swap(Matrix& other) noexcept {
    std::swap(rows, other.rows);
    std::swap(cols, other.cols);
    std::swap(nnz, other.nnz);
    std::swap(is_dense, other.is_dense);
    std::swap(values, other.values);
    std::swap(row_indices, other.row_indices);
    std::swap(col_ptrs, other.col_ptrs);
    std::swap(dense_data, other.dense_data);
}

// Assignment operator using copy-and-swap idiom
Matrix& Matrix::operator=(const Matrix& other) {
    if (this != &other) {
        Matrix temp(other);  // Create a copy (may throw, but *this is unchanged)
        swap(temp);          // Swap with temp (noexcept)
        // temp destructor cleans up old data
    }
    return *this;
}

// Access elements
double& Matrix::operator()(int row, int col) {
    if (row < 0 || row >= rows || col < 0 || col >= cols) {
        throw std::out_of_range("Matrix indices out of bounds");
    }
    
    if (is_dense) {
        return dense_data[row * cols + col];
    } else {
        // For sparse matrix, need to search in CSC format
        // This is inefficient but necessary for element access
        int start = col_ptrs[col];
        int end = col_ptrs[col + 1];
        
        for (int idx = start; idx < end; ++idx) {
            if (row_indices[idx] == row) {
                return values[idx];
            }
        }
        
        // If not found, this is a zero element - we can't return a reference to zero
        throw std::runtime_error("Cannot get reference to zero element in sparse matrix");
    }
}

double Matrix::operator()(int row, int col) const {
    if (row < 0 || row >= rows || col < 0 || col >= cols) {
        throw std::out_of_range("Matrix indices out of bounds");
    }
    
    if (is_dense) {
        return dense_data[row * cols + col];
    } else {
        // For sparse matrix, search in CSC format
        int start = col_ptrs[col];
        int end = col_ptrs[col + 1];
        
        for (int idx = start; idx < end; ++idx) {
            if (row_indices[idx] == row) {
                return values[idx];
            }
        }
        
        // Element not found, so it's zero
        return 0.0;
    }
}

// Basic operations
Matrix Matrix::operator+(const Matrix& other) const {
    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument("Matrix dimensions must match for addition");
    }
    
    if (is_dense && other.is_dense) {
        Matrix result(rows, cols);
        for (int i = 0; i < rows * cols; ++i) {
            result.dense_data[i] = dense_data[i] + other.dense_data[i];
        }
        return result;
    } else {
        // Convert to dense for simplicity (in a full implementation, sparse operations would be more efficient)
        Matrix this_dense = toDenseMatrix();
        Matrix other_dense = other.toDenseMatrix();
        return this_dense + other_dense;
    }
}

Matrix Matrix::operator-(const Matrix& other) const {
    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument("Matrix dimensions must match for subtraction");
    }
    
    if (is_dense && other.is_dense) {
        Matrix result(rows, cols);
        for (int i = 0; i < rows * cols; ++i) {
            result.dense_data[i] = dense_data[i] - other.dense_data[i];
        }
        return result;
    } else {
        // Convert to dense for simplicity
        Matrix this_dense = toDenseMatrix();
        Matrix other_dense = other.toDenseMatrix();
        return this_dense - other_dense;
    }
}

Matrix Matrix::operator*(const Matrix& other) const {
    return multiply(other);
}

Matrix Matrix::operator*(double scalar) const {
    if (is_dense) {
        Matrix result(rows, cols);
        for (int i = 0; i < rows * cols; ++i) {
            result.dense_data[i] = dense_data[i] * scalar;
        }
        return result;
    } else {
        // For sparse matrix, create a new sparse matrix with scaled values
        Matrix result(rows, cols, values, row_indices, col_ptrs, nnz);
        result.is_dense = false;

        // Scale all non-zero values
        for (int i = 0; i < nnz; ++i) {
            result.values[i] = values[i] * scalar;
        }

        return result;
    }
}

// Matrix operations
Matrix Matrix::transpose() const {
    if (is_dense) {
        Matrix result(cols, rows);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                result(j, i) = (*this)(i, j);
            }
        }
        return result;
    } else {
        // For sparse matrix, transpose in CSC format
        Matrix result(cols, rows);
        result.is_dense = false;
        result.nnz = nnz;
        
        // Allocate memory for transposed matrix (CSR format)
        result.values = new double[nnz];
        result.row_indices = new int[nnz];
        result.col_ptrs = new int[rows + 1];
        
        // Transpose: CSC -> CSR
        // First, count non-zeros per row in the transposed matrix (which are columns in original)
        int* row_counts = new int[rows];
        for (int i = 0; i < rows; ++i) row_counts[i] = 0;
        
        for (int j = 0; j < cols; ++j) {
            for (int idx = col_ptrs[j]; idx < col_ptrs[j + 1]; ++idx) {
                int row_idx = row_indices[idx];
                row_counts[row_idx]++;
            }
        }
        
        // Create row pointers for the transposed matrix
        result.col_ptrs[0] = 0;
        for (int i = 1; i <= rows; ++i) {
            result.col_ptrs[i] = result.col_ptrs[i-1] + row_counts[i-1];
        }
        
        // Reset row_counts to use as current position tracker
        for (int i = 0; i < rows; ++i) row_counts[i] = result.col_ptrs[i];
        
        // Fill the transposed matrix
        for (int j = 0; j < cols; ++j) {
            for (int idx = col_ptrs[j]; idx < col_ptrs[j + 1]; ++idx) {
                int row_idx = row_indices[idx];
                int pos = row_counts[row_idx]++;
                
                result.values[pos] = values[idx];
                result.row_indices[pos] = j;  // j becomes the new row in transposed matrix
            }
        }
        
        delete[] row_counts;
        return result;
    }
}

Matrix Matrix::multiply(const Matrix& other) const {
    if (cols != other.rows) {
        throw std::invalid_argument("Number of columns in first matrix must match number of rows in second matrix");
    }
    
    if (is_dense && other.is_dense) {
        Matrix result(rows, other.cols);
        
        // Dense matrix multiplication
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < other.cols; ++j) {
                double sum = 0.0;
                for (int k = 0; k < cols; ++k) {
                    sum += (*this)(i, k) * other(k, j);
                }
                result(i, j) = sum;
            }
        }
        
        return result;
    } else {
        // For sparse matrices, convert to dense for simplicity (a full implementation would use sparse algorithms)
        Matrix this_dense = toDenseMatrix();
        Matrix other_dense = other.toDenseMatrix();
        return this_dense.multiply(other_dense);
    }
}

Matrix Matrix::multiply_parallel(const Matrix& other) const {
    if (cols != other.rows) {
        throw std::invalid_argument("Number of columns in first matrix must match number of rows in second matrix");
    }
    
    if (is_dense && other.is_dense) {
        Matrix result(rows, other.cols);
        
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < other.cols; ++j) {
                double sum = 0.0;
                for (int k = 0; k < cols; ++k) {
                    sum += (*this)(i, k) * other(k, j);
                }
                result(i, j) = sum;
            }
        }
        
        return result;
    } else {
        // For sparse matrices, convert to dense for simplicity
        Matrix this_dense = toDenseMatrix();
        Matrix other_dense = other.toDenseMatrix();
        return this_dense.multiply_parallel(other_dense);
    }
}

// Convert between dense and sparse
void Matrix::toDense(double* result) const {
    if (is_dense) {
        std::memcpy(result, dense_data, rows * cols * sizeof(double));
    } else {
        // Initialize result to zero
        for (int i = 0; i < rows * cols; ++i) {
            result[i] = 0.0;
        }
        
        // Fill in non-zero values from sparse representation
        for (int j = 0; j < cols; ++j) {
            for (int idx = col_ptrs[j]; idx < col_ptrs[j + 1]; ++idx) {
                int i = row_indices[idx];
                result[i * cols + j] = values[idx];
            }
        }
    }
}

void Matrix::toSparse(const double* dense_data, int rows, int cols) {
    // Clean up existing sparse data if any
    if (!is_dense) {
        if (values) delete[] values;
        if (row_indices) delete[] row_indices;
        if (col_ptrs) delete[] col_ptrs;
    }
    
    // Count non-zero elements
    int count = 0;
    for (int i = 0; i < rows * cols; ++i) {
        if (std::abs(dense_data[i]) > 1e-12) {
            count++;
        }
    }
    
    // Allocate memory for sparse representation
    this->values = new double[count];
    this->row_indices = new int[count];
    this->col_ptrs = new int[cols + 1];
    this->nnz = count;
    this->is_dense = false;
    
    // Fill sparse representation in CSC format
    int idx = 0;
    col_ptrs[0] = 0;
    
    for (int j = 0; j < cols; ++j) {
        for (int i = 0; i < rows; ++i) {
            if (std::abs(dense_data[i * cols + j]) > 1e-12) {
                values[idx] = dense_data[i * cols + j];
                row_indices[idx] = i;
                idx++;
            }
        }
        col_ptrs[j + 1] = idx;
    }
}

Matrix Matrix::toDenseMatrix() const {
    if (is_dense) {
        return *this;
    } else {
        Matrix result(rows, cols);
        toDense(result.dense_data);
        return result;
    }
}

Matrix Matrix::toSparseMatrix() const {
    if (!is_dense) {
        return *this;
    } else {
        Matrix result(rows, cols);
        result.toSparse(dense_data, rows, cols);
        return result;
    }
}

// LAPACK operations
double* Matrix::eigenvalues() const {
    if (rows != cols) {
        throw std::invalid_argument("Matrix must be square to compute eigenvalues");
    }
    
    // Convert to dense if needed
    Matrix A = toDenseMatrix();
    double* eigenvals = new double[rows];
    Matrix eigenvectors(rows, cols);
    
    A.syev_wrapper(eigenvals, eigenvectors);
    return eigenvals;
}

std::pair<double*, Matrix> Matrix::eigensystem() const {
    if (rows != cols) {
        throw std::invalid_argument("Matrix must be square to compute eigensystem");
    }
    
    Matrix A = toDenseMatrix();
    double* eigenvals = new double[rows];
    Matrix eigenvectors(rows, cols);
    
    A.syev_wrapper(eigenvals, eigenvectors);
    return std::make_pair(eigenvals, eigenvectors);
}

// LAPACK wrapper functions
void Matrix::syev_wrapper(double* eigenvalues, Matrix& eigenvectors) const {
    if (rows != cols) {
        throw std::invalid_argument("Matrix must be square for syev");
    }
    
    int n = rows;
    int info;
    
    // Create a copy of the matrix data for LAPACK (LAPACK uses column-major order)
    double* A_copy = new double[n * n];
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            A_copy[j * n + i] = (*this)(i, j);  // Convert from row-major to column-major
        }
    }
    
    double* work = new double[3 * n]; // Workspace
    
    // Call LAPACK function to compute eigenvalues and eigenvectors of symmetric matrix
    info = LAPACKE_dsyev(LAPACK_COL_MAJOR, 'V', 'U', n, A_copy, n, eigenvalues);
    
    if (info != 0) {
        delete[] A_copy;
        delete[] work;
        throw std::runtime_error("LAPACK syev failed with info = " + std::to_string(info));
    }
    
    // Copy eigenvectors back to the result matrix (convert from column-major to row-major)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            eigenvectors(i, j) = A_copy[j * n + i];
        }
    }
    
    delete[] A_copy;
    delete[] work;
}

void Matrix::geev_wrapper(double* real_eigenvals, double* imag_eigenvals,
                         [[maybe_unused]] Matrix& left_eigenvectors, Matrix& right_eigenvectors) const {
    if (rows != cols) {
        throw std::invalid_argument("Matrix must be square for geev");
    }
    
    int n = rows;
    int info;
    
    // Create a copy of the matrix data for LAPACK (column-major order)
    double* A_copy = new double[n * n];
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            A_copy[j * n + i] = (*this)(i, j);
        }
    }
    
    double* work = new double[4 * n]; // Workspace
    
    // Call LAPACK function for general eigenvalue problem
    info = LAPACKE_dgeev(LAPACK_COL_MAJOR, 'N', 'V', n,
                         A_copy, n,
                         real_eigenvals, imag_eigenvals,
                         nullptr, n,  // Don't compute left eigenvectors
                         right_eigenvectors.dense_data, n);
    
    if (info != 0) {
        delete[] A_copy;
        delete[] work;
        throw std::runtime_error("LAPACK geev failed with info = " + std::to_string(info));
    }
    
    delete[] A_copy;
    delete[] work;
}

// Utility functions
void Matrix::print() const {
    if (is_dense) {
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                std::cout << (*this)(i, j) << " ";
            }
            std::cout << std::endl;
        }
    } else {
        std::cout << "Sparse matrix (" << nnz << " non-zeros):" << std::endl;
        for (int j = 0; j < cols; ++j) {
            for (int idx = col_ptrs[j]; idx < col_ptrs[j + 1]; ++idx) {
                int i = row_indices[idx];
                std::cout << "(" << i << "," << j << "): " << values[idx] << std::endl;
            }
        }
    }
}

void Matrix::fill(double value) {
    if (is_dense) {
        for (int i = 0; i < rows * cols; ++i) {
            dense_data[i] = value;
        }
    } else {
        // Convert to dense to fill, then back to sparse if needed
        Matrix dense_version = toDenseMatrix();
        dense_version.fill(value);
        *this = dense_version.toSparseMatrix();
    }
}

void Matrix::randomize(double min, double max) {
    if (is_dense) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dis(min, max);
        
        for (int i = 0; i < rows * cols; ++i) {
            dense_data[i] = dis(gen);
        }
    } else {
        // Convert to dense to randomize, then back to sparse if needed
        Matrix dense_version = toDenseMatrix();
        dense_version.randomize(min, max);
        *this = dense_version.toSparseMatrix();
    }
}

// Static utility functions
Matrix Matrix::identity(int size) {
    Matrix result(size, size);
    for (int i = 0; i < size; ++i) {
        result(i, i) = 1.0;
    }
    return result;
}

Matrix Matrix::zeros(int rows, int cols) {
    return Matrix(rows, cols); // Constructor already initializes to zero
}

Matrix Matrix::ones(int rows, int cols) {
    Matrix result(rows, cols);
    result.fill(1.0);
    return result;
}

Matrix Matrix::random(int rows, int cols, double min, double max) {
    Matrix result(rows, cols);
    result.randomize(min, max);
    return result;
}

// Scalar multiplication (scalar * matrix)
Matrix operator*(double scalar, const Matrix& matrix) {
    return matrix * scalar;
}