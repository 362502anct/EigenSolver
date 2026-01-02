#include "Matrix.h"
#include "../util/SIMDDispatcher.h"
#include <stdexcept>
#include <random>
#include <algorithm>
#include <cmath>
#include <cstring>

// Constructors
Matrix::Matrix()
    : values(nullptr), row_indices(nullptr), col_ptrs(nullptr),
      rows(0), cols(0), nnz(0) {}

Matrix::Matrix(int r, int c)
    : values(nullptr), row_indices(nullptr), col_ptrs(nullptr),
      rows(r), cols(c), nnz(0) {
    if (r < 0 || c < 0) {
        throw std::invalid_argument("Matrix dimensions must be non-negative");
    }

    // Allocate column pointers (all zeros, meaning no non-zeros in any column)
    col_ptrs = new int[cols + 1];
    for (int i = 0; i <= cols; ++i) {
        col_ptrs[i] = 0;
    }
}

Matrix::Matrix(int rows, int cols, double* vals, int* row_idxs, int* col_ptrs_data, int non_zeros)
    : values(nullptr), row_indices(nullptr), col_ptrs(nullptr),
      rows(rows), cols(cols), nnz(non_zeros) {
    values = new double[nnz];
    row_indices = new int[nnz];
    col_ptrs = new int[cols + 1];

    std::memcpy(values, vals, nnz * sizeof(double));
    std::memcpy(row_indices, row_idxs, nnz * sizeof(int));
    std::memcpy(col_ptrs, col_ptrs_data, (cols + 1) * sizeof(int));
}

Matrix::Matrix(const Matrix& other)
    : values(nullptr), row_indices(nullptr), col_ptrs(nullptr),
      rows(other.rows), cols(other.cols), nnz(other.nnz) {
    values = new double[nnz];
    row_indices = new int[nnz];
    col_ptrs = new int[cols + 1];

    std::memcpy(values, other.values, nnz * sizeof(double));
    std::memcpy(row_indices, other.row_indices, nnz * sizeof(int));
    std::memcpy(col_ptrs, other.col_ptrs, (cols + 1) * sizeof(int));
}

// Move constructor
Matrix::Matrix(Matrix&& other) noexcept
    : values(other.values), row_indices(other.row_indices), col_ptrs(other.col_ptrs),
      rows(other.rows), cols(other.cols), nnz(other.nnz) {
    // Leave the source object in a valid but empty state
    other.rows = 0;
    other.cols = 0;
    other.nnz = 0;
    other.values = nullptr;
    other.row_indices = nullptr;
    other.col_ptrs = nullptr;
}

// Destructor
Matrix::~Matrix() {
    if (values) delete[] values;
    if (row_indices) delete[] row_indices;
    if (col_ptrs) delete[] col_ptrs;
}

// Swap function for copy-and-swap idiom
void Matrix::swap(Matrix& other) noexcept {
    std::swap(rows, other.rows);
    std::swap(cols, other.cols);
    std::swap(nnz, other.nnz);
    std::swap(values, other.values);
    std::swap(row_indices, other.row_indices);
    std::swap(col_ptrs, other.col_ptrs);
}

// Assignment operator
Matrix& Matrix::operator=(const Matrix& other) {
    if (this != &other) {
        // Clean up existing data
        if (values) delete[] values;
        if (row_indices) delete[] row_indices;
        if (col_ptrs) delete[] col_ptrs;

        rows = other.rows;
        cols = other.cols;
        nnz = other.nnz;

        values = new double[nnz];
        row_indices = new int[nnz];
        col_ptrs = new int[cols + 1];

        std::memcpy(values, other.values, nnz * sizeof(double));
        std::memcpy(row_indices, other.row_indices, nnz * sizeof(int));
        std::memcpy(col_ptrs, other.col_ptrs, (cols + 1) * sizeof(int));
    }
    return *this;
}

// Move assignment operator
Matrix& Matrix::operator=(Matrix&& other) noexcept {
    if (this != &other) {
        // Clean up existing data
        if (values) delete[] values;
        if (row_indices) delete[] row_indices;
        if (col_ptrs) delete[] col_ptrs;

        // Steal resources from other
        rows = other.rows;
        cols = other.cols;
        nnz = other.nnz;
        values = other.values;
        row_indices = other.row_indices;
        col_ptrs = other.col_ptrs;

        // Leave the source object in a valid but empty state
        other.rows = 0;
        other.cols = 0;
        other.nnz = 0;
        other.values = nullptr;
        other.row_indices = nullptr;
        other.col_ptrs = nullptr;
    }
    return *this;
}

// Access elements (read-only)
double Matrix::operator()(int row, int col) const {
    if (row < 0 || row >= rows || col < 0 || col >= cols) {
        throw std::out_of_range("Matrix indices out of bounds");
    }

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

// Set element value
void Matrix::set(int row, int col, double value) {
    if (row < 0 || row >= rows || col < 0 || col >= cols) {
        throw std::out_of_range("Matrix indices out of bounds");
    }

    // For sparse matrix, search in CSC format
    int start = col_ptrs[col];
    int end = col_ptrs[col + 1];

    // First, check if element already exists (only if values is not null)
    if (values != nullptr) {
        for (int idx = start; idx < end; ++idx) {
            if (row_indices[idx] == row) {
                // Found existing element, update it
                values[idx] = value;
                return;
            }
        }
    }

    // Element not found, need to insert it (only if non-zero)
    if (std::abs(value) > 1e-12) {
        // Allocate new arrays with increased size
        int new_nnz = nnz + 1;
        double* new_values = new double[new_nnz];
        int* new_row_indices = new int[new_nnz];

        // Find the correct insertion position to maintain row order
        int insert_pos = start;

        // Copy elements before the insertion point
        for (int idx = 0; idx < insert_pos; ++idx) {
            new_values[idx] = values ? values[idx] : 0.0;
            new_row_indices[idx] = row_indices[idx];
        }

        // Insert new element
        new_values[insert_pos] = value;
        new_row_indices[insert_pos] = row;

        // Copy elements after the insertion point
        for (int idx = insert_pos; idx < nnz; ++idx) {
            new_values[idx + 1] = values ? values[idx] : 0.0;
            new_row_indices[idx + 1] = row_indices[idx];
        }

        // Free old arrays
        if (values) delete[] values;
        if (row_indices) delete[] row_indices;

        // Update pointers
        values = new_values;
        row_indices = new_row_indices;
        nnz = new_nnz;

        // Update column pointers
        for (int j = col + 1; j <= cols; ++j) {
            col_ptrs[j]++;
        }
    }
}

// Basic operations
Matrix Matrix::operator+(const Matrix& other) const {
    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument("Matrix dimensions must match for addition");
    }

    // Convert to dense for simplicity (for sparse addition, we'd need merge algorithm)
    Matrix result(rows, cols);

    // Count non-zeros in result
    int max_nnz = nnz + other.nnz;
    double* temp_values = new double[max_nnz];
    int* temp_row_indices = new int[max_nnz];
    int* temp_col_ptrs = new int[cols + 1];

    temp_col_ptrs[0] = 0;
    int current_nnz = 0;

    for (int j = 0; j < cols; ++j) {
        int idx1 = col_ptrs[j];
        int idx2 = other.col_ptrs[j];
        int end1 = col_ptrs[j + 1];
        int end2 = other.col_ptrs[j + 1];

        while (idx1 < end1 || idx2 < end2) {
            int row1 = (idx1 < end1) ? row_indices[idx1] : rows;
            int row2 = (idx2 < end2) ? other.row_indices[idx2] : rows;

            if (row1 < row2) {
                temp_values[current_nnz] = values[idx1];
                temp_row_indices[current_nnz] = row1;
                current_nnz++;
                idx1++;
            } else if (row2 < row1) {
                temp_values[current_nnz] = other.values[idx2];
                temp_row_indices[current_nnz] = row2;
                current_nnz++;
                idx2++;
            } else {
                double sum = values[idx1] + other.values[idx2];
                if (std::abs(sum) > 1e-12) {
                    temp_values[current_nnz] = sum;
                    temp_row_indices[current_nnz] = row1;
                    current_nnz++;
                }
                idx1++;
                idx2++;
            }
        }

        temp_col_ptrs[j + 1] = current_nnz;
    }

    // Allocate result with exact size
    result.values = new double[current_nnz];
    result.row_indices = new int[current_nnz];
    result.nnz = current_nnz;

    std::memcpy(result.values, temp_values, current_nnz * sizeof(double));
    std::memcpy(result.row_indices, temp_row_indices, current_nnz * sizeof(int));
    std::memcpy(result.col_ptrs, temp_col_ptrs, (cols + 1) * sizeof(int));

    delete[] temp_values;
    delete[] temp_row_indices;
    delete[] temp_col_ptrs;

    return result;
}

Matrix Matrix::operator-(const Matrix& other) const {
    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument("Matrix dimensions must match for subtraction");
    }

    // Convert to dense for simplicity
    Matrix result(rows, cols);

    // Count non-zeros in result
    int max_nnz = nnz + other.nnz;
    double* temp_values = new double[max_nnz];
    int* temp_row_indices = new int[max_nnz];
    int* temp_col_ptrs = new int[cols + 1];

    temp_col_ptrs[0] = 0;
    int current_nnz = 0;

    for (int j = 0; j < cols; ++j) {
        int idx1 = col_ptrs[j];
        int idx2 = other.col_ptrs[j];
        int end1 = col_ptrs[j + 1];
        int end2 = other.col_ptrs[j + 1];

        while (idx1 < end1 || idx2 < end2) {
            int row1 = (idx1 < end1) ? row_indices[idx1] : rows;
            int row2 = (idx2 < end2) ? other.row_indices[idx2] : rows;

            if (row1 < row2) {
                temp_values[current_nnz] = values[idx1];
                temp_row_indices[current_nnz] = row1;
                current_nnz++;
                idx1++;
            } else if (row2 < row1) {
                temp_values[current_nnz] = -other.values[idx2];
                temp_row_indices[current_nnz] = row2;
                current_nnz++;
                idx2++;
            } else {
                double diff = values[idx1] - other.values[idx2];
                if (std::abs(diff) > 1e-12) {
                    temp_values[current_nnz] = diff;
                    temp_row_indices[current_nnz] = row1;
                    current_nnz++;
                }
                idx1++;
                idx2++;
            }
        }

        temp_col_ptrs[j + 1] = current_nnz;
    }

    // Allocate result with exact size
    result.values = new double[current_nnz];
    result.row_indices = new int[current_nnz];
    result.nnz = current_nnz;

    std::memcpy(result.values, temp_values, current_nnz * sizeof(double));
    std::memcpy(result.row_indices, temp_row_indices, current_nnz * sizeof(int));
    std::memcpy(result.col_ptrs, temp_col_ptrs, (cols + 1) * sizeof(int));

    delete[] temp_values;
    delete[] temp_row_indices;
    delete[] temp_col_ptrs;

    return result;
}

Matrix Matrix::operator*(const Matrix& other) const {
    return multiply(other);
}

Matrix Matrix::operator*(double scalar) const {
    Matrix result(rows, cols, values, row_indices, col_ptrs, nnz);

    // Use SIMD-optimized scalar multiplication
    auto& dispatcher = SIMDDispatcher::getInstance();
    auto multiply_func = dispatcher.getScalarMultiplyFunc();
    multiply_func(result.values, nnz, scalar);

    return result;
}

// Matrix operations
Matrix Matrix::transpose() const {
    Matrix result(cols, rows);
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

Matrix Matrix::multiply(const Matrix& other) const {
    if (cols != other.rows) {
        throw std::invalid_argument("Number of columns in first matrix must match number of rows in second matrix");
    }

    // Use sparse matrix multiplication
    Matrix result(rows, other.cols);

    // Temporary arrays to build the result
    int max_nnz_per_col = nnz;
    double* temp_values = new double[max_nnz_per_col * other.cols];
    int* temp_row_indices = new int[max_nnz_per_col * other.cols];
    int* temp_col_ptrs = new int[other.cols + 1];

    temp_col_ptrs[0] = 0;
    int current_nnz = 0;

    for (int j = 0; j < other.cols; ++j) {
        // For each column in result, compute the dot products
        for (int i = 0; i < rows; ++i) {
            double sum = 0.0;

            // Compute dot product of row i of this matrix with column j of other matrix
            for (int k = 0; k < cols; ++k) {
                double a_ik = (*this)(i, k);
                double a_kj = other(k, j);

                if (std::abs(a_ik) > 1e-12 && std::abs(a_kj) > 1e-12) {
                    sum += a_ik * a_kj;
                }
            }

            if (std::abs(sum) > 1e-12) {
                temp_values[current_nnz] = sum;
                temp_row_indices[current_nnz] = i;
                current_nnz++;
            }
        }

        temp_col_ptrs[j + 1] = current_nnz;
    }

    // Allocate result with exact size
    result.values = new double[current_nnz];
    result.row_indices = new int[current_nnz];
    result.nnz = current_nnz;

    std::memcpy(result.values, temp_values, current_nnz * sizeof(double));
    std::memcpy(result.row_indices, temp_row_indices, current_nnz * sizeof(int));
    std::memcpy(result.col_ptrs, temp_col_ptrs, (other.cols + 1) * sizeof(int));

    delete[] temp_values;
    delete[] temp_row_indices;
    delete[] temp_col_ptrs;

    return result;
}

Matrix Matrix::multiply_parallel(const Matrix& other) const {
    if (cols != other.rows) {
        throw std::invalid_argument("Number of columns in first matrix must match number of rows in second matrix");
    }

    // Use sparse matrix multiplication with OpenMP
    Matrix result(rows, other.cols);

    // Temporary arrays to build the result
    int max_nnz_per_col = nnz;
    double* temp_values = new double[max_nnz_per_col * other.cols];
    int* temp_row_indices = new int[max_nnz_per_col * other.cols];
    int* temp_col_ptrs = new int[other.cols + 1];

    temp_col_ptrs[0] = 0;
    int current_nnz = 0;

    #pragma omp parallel for
    for (int j = 0; j < other.cols; ++j) {
        // For each column in result, compute the dot products
        for (int i = 0; i < rows; ++i) {
            double sum = 0.0;

            // Compute dot product of row i of this matrix with column j of other matrix
            for (int k = 0; k < cols; ++k) {
                double a_ik = (*this)(i, k);
                double a_kj = other(k, j);

                if (std::abs(a_ik) > 1e-12 && std::abs(a_kj) > 1e-12) {
                    sum += a_ik * a_kj;
                }
            }

            if (std::abs(sum) > 1e-12) {
                #pragma omp critical
                {
                    temp_values[current_nnz] = sum;
                    temp_row_indices[current_nnz] = i;
                    current_nnz++;
                }
            }
        }

        temp_col_ptrs[j + 1] = current_nnz;
    }

    // Allocate result with exact size
    result.values = new double[current_nnz];
    result.row_indices = new int[current_nnz];
    result.nnz = current_nnz;

    std::memcpy(result.values, temp_values, current_nnz * sizeof(double));
    std::memcpy(result.row_indices, temp_row_indices, current_nnz * sizeof(int));
    std::memcpy(result.col_ptrs, temp_col_ptrs, (other.cols + 1) * sizeof(int));

    delete[] temp_values;
    delete[] temp_row_indices;
    delete[] temp_col_ptrs;

    return result;
}

// SIMD-optimized dense matrix multiplication
Matrix Matrix::multiply_dense_simd(const Matrix& other) const {
    if (cols != other.rows) {
        throw std::invalid_argument("Number of columns in first matrix must match number of rows in second matrix");
    }

    // Convert both matrices to dense format for SIMD processing
    double* A_dense = new double[rows * cols];
    double* B_dense = new double[other.rows * other.cols];
    double* C_dense = new double[rows * other.cols];

    toDense(A_dense);
    other.toDense(B_dense);

    // Use SIMD-optimized matrix multiplication
    auto& dispatcher = SIMDDispatcher::getInstance();
    auto multiply_func = dispatcher.getMatrixMultiplyFunc();

    multiply_func(C_dense, A_dense, B_dense, rows, cols, other.cols);

    // Create result from dense multiplication
    Matrix result(rows, other.cols);
    result.is_dense = true;
    result.dense_data = C_dense;

    // Clean up temporary arrays
    delete[] A_dense;
    delete[] B_dense;

    return result;
}

// Convert to dense format
void Matrix::toDense(double* result) const {
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

// LAPACK operations
double* Matrix::eigenvalues() const {
    if (rows != cols) {
        throw std::invalid_argument("Matrix must be square to compute eigenvalues");
    }

    // Convert to dense for LAPACK
    double* dense_matrix = new double[rows * cols];
    toDense(dense_matrix);

    double* eigenvals = new double[rows];
    int info;

    // Create a copy for LAPACK (LAPACK uses column-major order)
    double* A_copy = new double[rows * cols];
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            A_copy[j * rows + i] = dense_matrix[i * cols + j];
        }
    }

    // Call LAPACK function to compute eigenvalues of symmetric matrix
    info = LAPACKE_dsyev(LAPACK_COL_MAJOR, 'N', 'U', rows, A_copy, rows, eigenvals);

    delete[] dense_matrix;
    delete[] A_copy;

    if (info != 0) {
        delete[] eigenvals;
        throw std::runtime_error("LAPACK dsyev failed with info = " + std::to_string(info));
    }

    return eigenvals;
}

std::pair<double*, Matrix> Matrix::eigensystem() const {
    if (rows != cols) {
        throw std::invalid_argument("Matrix must be square to compute eigensystem");
    }

    // Convert to dense for LAPACK
    double* dense_matrix = new double[rows * cols];
    toDense(dense_matrix);

    double* eigenvals = new double[rows];
    int info;

    // Create a copy for LAPACK (column-major order)
    double* A_copy = new double[rows * cols];
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            A_copy[j * rows + i] = dense_matrix[i * cols + j];
        }
    }

    // Call LAPACK function to compute eigenvalues and eigenvectors
    info = LAPACKE_dsyev(LAPACK_COL_MAJOR, 'V', 'U', rows, A_copy, rows, eigenvals);

    delete[] dense_matrix;

    if (info != 0) {
        delete[] eigenvals;
        delete[] A_copy;
        throw std::runtime_error("LAPACK dsyev failed with info = " + std::to_string(info));
    }

    // Convert eigenvectors to sparse matrix
    Matrix eigenvectors(rows, cols);
    int nnz_count = 0;

    // Count non-zeros
    for (int i = 0; i < rows * cols; ++i) {
        if (std::abs(A_copy[i]) > 1e-12) {
            nnz_count++;
        }
    }

    eigenvectors.values = new double[nnz_count];
    eigenvectors.row_indices = new int[nnz_count];
    eigenvectors.nnz = nnz_count;

    int idx = 0;
    for (int j = 0; j < cols; ++j) {
        eigenvectors.col_ptrs[j] = idx;
        for (int i = 0; i < rows; ++i) {
            double val = A_copy[j * rows + i];
            if (std::abs(val) > 1e-12) {
                eigenvectors.values[idx] = val;
                eigenvectors.row_indices[idx] = i;
                idx++;
            }
        }
    }
    eigenvectors.col_ptrs[cols] = idx;

    delete[] A_copy;

    return std::make_pair(eigenvals, eigenvectors);
}

// LAPACK wrapper functions
void Matrix::syev_wrapper(double* eigenvalues, Matrix& eigenvectors) const {
    if (rows != cols) {
        throw std::invalid_argument("Matrix must be square for syev");
    }

    int n = rows;
    int info;

    // Convert to dense for LAPACK
    double* dense_matrix = new double[rows * cols];
    toDense(dense_matrix);

    // Create a copy of the matrix data for LAPACK (LAPACK uses column-major order)
    double* A_copy = new double[n * n];
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            A_copy[j * n + i] = dense_matrix[i * n + j];
        }
    }

    delete[] dense_matrix;

    double* work = new double[3 * n];

    // Call LAPACK function to compute eigenvalues and eigenvectors of symmetric matrix
    info = LAPACKE_dsyev(LAPACK_COL_MAJOR, 'V', 'U', n, A_copy, n, eigenvalues);

    if (info != 0) {
        delete[] A_copy;
        delete[] work;
        throw std::runtime_error("LAPACK syev failed with info = " + std::to_string(info));
    }

    // Convert eigenvectors to sparse matrix
    int nnz_count = 0;
    for (int i = 0; i < n * n; ++i) {
        if (std::abs(A_copy[i]) > 1e-12) {
            nnz_count++;
        }
    }

    eigenvectors.values = new double[nnz_count];
    eigenvectors.row_indices = new int[nnz_count];
    eigenvectors.nnz = nnz_count;

    int idx = 0;
    for (int j = 0; j < n; ++j) {
        eigenvectors.col_ptrs[j] = idx;
        for (int i = 0; i < n; ++i) {
            double val = A_copy[j * n + i];
            if (std::abs(val) > 1e-12) {
                eigenvectors.values[idx] = val;
                eigenvectors.row_indices[idx] = i;
                idx++;
            }
        }
    }
    eigenvectors.col_ptrs[n] = idx;

    delete[] A_copy;
    delete[] work;
}

void Matrix::geev_wrapper(double* real_eigenvals, double* imag_eigenvals,
                         [[maybe_unused]] Matrix& left_eigenvectors,
                         Matrix& right_eigenvectors) const {
    if (rows != cols) {
        throw std::invalid_argument("Matrix must be square for geev");
    }

    int n = rows;
    int info;

    // Convert to dense for LAPACK
    double* dense_matrix = new double[rows * cols];
    toDense(dense_matrix);

    // Create a copy of the matrix data for LAPACK (column-major order)
    double* A_copy = new double[n * n];
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            A_copy[j * n + i] = dense_matrix[i * n + j];
        }
    }

    delete[] dense_matrix;

    double* work = new double[4 * n];
    double* vr = new double[n * n]; // Right eigenvectors

    // Call LAPACK function for general eigenvalue problem
    // Note: 'N' means don't compute left eigenvectors
    (void)left_eigenvectors;  // Suppress unused parameter warning
    info = LAPACKE_dgeev(LAPACK_COL_MAJOR, 'N', 'V', n,
                         A_copy, n,
                         real_eigenvals, imag_eigenvals,
                         nullptr, n,  // Don't compute left eigenvectors
                         vr, n);

    if (info != 0) {
        delete[] A_copy;
        delete[] work;
        delete[] vr;
        throw std::runtime_error("LAPACK geev failed with info = " + std::to_string(info));
    }

    // Convert right eigenvectors to sparse matrix
    int nnz_count = 0;
    for (int i = 0; i < n * n; ++i) {
        if (std::abs(vr[i]) > 1e-12) {
            nnz_count++;
        }
    }

    right_eigenvectors.values = new double[nnz_count];
    right_eigenvectors.row_indices = new int[nnz_count];
    right_eigenvectors.nnz = nnz_count;

    int idx = 0;
    for (int j = 0; j < n; ++j) {
        right_eigenvectors.col_ptrs[j] = idx;
        for (int i = 0; i < n; ++i) {
            double val = vr[j * n + i];
            if (std::abs(val) > 1e-12) {
                right_eigenvectors.values[idx] = val;
                right_eigenvectors.row_indices[idx] = i;
                idx++;
            }
        }
    }
    right_eigenvectors.col_ptrs[n] = idx;

    delete[] A_copy;
    delete[] work;
    delete[] vr;
}

// Utility functions
void Matrix::print() const {
    std::cout << "Sparse matrix (" << nnz << " non-zeros):" << std::endl;
    for (int j = 0; j < cols; ++j) {
        for (int idx = col_ptrs[j]; idx < col_ptrs[j + 1]; ++idx) {
            int i = row_indices[idx];
            std::cout << "(" << i << "," << j << "): " << values[idx] << std::endl;
        }
    }
}

void Matrix::fill(double value) {
    if (std::abs(value) < 1e-12) {
        // If filling with zeros, just deallocate non-zeros
        if (values) delete[] values;
        if (row_indices) delete[] row_indices;

        nnz = 0;
        values = nullptr;
        row_indices = nullptr;

        for (int i = 0; i <= cols; ++i) {
            col_ptrs[i] = 0;
        }
    } else {
        // Fill with non-zero value - convert to fully dense
        if (values) delete[] values;
        if (row_indices) delete[] row_indices;

        nnz = rows * cols;
        values = new double[nnz];
        row_indices = new int[nnz];

        int idx = 0;
        for (int j = 0; j < cols; ++j) {
            col_ptrs[j] = idx;
            for (int i = 0; i < rows; ++i) {
                values[idx] = value;
                row_indices[idx] = i;
                idx++;
            }
        }
        col_ptrs[cols] = idx;
    }
}

void Matrix::randomize(double min, double max) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(min, max);

    // Fill all current non-zeros with random values
    for (int i = 0; i < nnz; ++i) {
        values[i] = dis(gen);
    }
}

// Static utility functions
Matrix Matrix::identity(int size) {
    Matrix result(size, size);

    result.nnz = size;
    result.values = new double[size];
    result.row_indices = new int[size];

    for (int i = 0; i < size; ++i) {
        result.values[i] = 1.0;
        result.row_indices[i] = i;
        result.col_ptrs[i] = i;
    }
    result.col_ptrs[size] = size;

    return result;
}

Matrix Matrix::zeros(int rows, int cols) {
    return Matrix(rows, cols);
}

Matrix Matrix::ones(int rows, int cols) {
    Matrix result(rows, cols);
    result.fill(1.0);
    return result;
}

Matrix Matrix::random(int rows, int cols, double min, double max) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(min, max);

    // Randomly decide density (e.g., 10% non-zeros)
    double density = 0.1;
    int expected_nnz = static_cast<int>(rows * cols * density);

    // Allocate arrays directly
    double* vals = new double[expected_nnz];
    int* row_idxs = new int[expected_nnz];
    int* col_ptr_data = new int[cols + 1];

    int idx = 0;
    for (int j = 0; j < cols; ++j) {
        col_ptr_data[j] = idx;
        for (int i = 0; i < rows; ++i) {
            if (dis(gen) > 0.9) {  // 10% chance of non-zero
                vals[idx] = dis(gen);
                row_idxs[idx] = i;
                idx++;
            }
        }
    }
    col_ptr_data[cols] = idx;

    // Create matrix from CSC data
    return Matrix(rows, cols, vals, row_idxs, col_ptr_data, idx);
}

// Scalar multiplication (scalar * matrix)
Matrix operator*(double scalar, const Matrix& matrix) {
    return matrix * scalar;
}
