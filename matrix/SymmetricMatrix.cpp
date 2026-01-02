#include "SymmetricMatrix.h"
#include "../util/SIMDDispatcher.h"
#include "../matrix/MatrixOps_SIMD.h"
#include <stdexcept>
#include <random>
#include <cmath>
#include <algorithm>
#include <cstring>

// Default constructor
SymmetricMatrix::SymmetricMatrix()
    : Matrix(0, 0), use_half_storage(true) {
}

// Constructor for square symmetric matrix
SymmetricMatrix::SymmetricMatrix(int size)
    : Matrix(size, size), use_half_storage(true) {
    int n = size;
    half_data.resize(n * (n + 1) / 2, 0.0);
}

// Constructor from general matrix - enforces symmetry
SymmetricMatrix::SymmetricMatrix(const Matrix& other)
    : Matrix(other.getRows(), other.getCols()), use_half_storage(true) {

    if (other.getRows() != other.getCols()) {
        throw std::invalid_argument("Symmetric matrix must be square");
    }

    // Initialize half-storage from the matrix
    initHalfStorage(other);
}

// Constructor from half-storage data
SymmetricMatrix::SymmetricMatrix(int size, const double* upper_tri)
    : Matrix(size, size), use_half_storage(true) {

    int packed_size = size * (size + 1) / 2;
    half_data.assign(upper_tri, upper_tri + packed_size);
}

// Constructor from CSC sparse data
SymmetricMatrix::SymmetricMatrix(int size, double* vals, int* row_idxs, int* col_ptrs_data, int non_zeros)
    : Matrix(size, size, vals, row_idxs, col_ptrs_data, non_zeros), use_half_storage(true) {

    // Initialize half-storage from the CSC data
    // This will enforce symmetry by only storing upper triangular part
    initHalfStorage(*this);
}

// Destructor
SymmetricMatrix::~SymmetricMatrix() {
    // Vector cleanup is automatic
}

// Copy constructor
SymmetricMatrix::SymmetricMatrix(const SymmetricMatrix& other)
    : Matrix(other), use_half_storage(other.use_half_storage),
      half_data(other.half_data) {
}

// Assignment operator
SymmetricMatrix& SymmetricMatrix::operator=(const SymmetricMatrix& other) {
    if (this != &other) {
        // Copy base class parts
        Matrix::operator=(other);
        use_half_storage = other.use_half_storage;
        half_data = other.half_data;
    }
    return *this;
}

// Get element with coordinate mapping
double SymmetricMatrix::operator()(int row, int col) const {
    if (row < 0 || row >= getRows() || col < 0 || col >= getCols()) {
        throw std::out_of_range("Matrix index out of range");
    }

    if (use_half_storage) {
        // Access from half-storage (upper triangular packed)
        int idx = packIndex(row, col);
        return half_data[idx];
    } else {
        // Fall back to base class storage
        return Matrix::operator()(row, col);
    }
}

// Set element (maintains symmetry automatically)
void SymmetricMatrix::set(int row, int col, double value) {
    if (row < 0 || row >= getRows() || col < 0 || col >= getCols()) {
        throw std::out_of_range("Matrix index out of range");
    }

    if (use_half_storage) {
        // Store only in upper triangular part
        int idx = packIndex(row, col);
        half_data[idx] = value;

        // Also update base class storage for compatibility with Matrix operations
        this->Matrix::set(row, col, value);
        if (row != col) {
            this->Matrix::set(col, row, value);
        }
    } else {
        // Use base class with automatic mirroring
        this->Matrix::set(row, col, value);
        if (row != col) {
            this->Matrix::set(col, row, value);
        }
    }
}

// Convert to full Matrix representation
Matrix SymmetricMatrix::toFullMatrix() const {
    int n = getRows();
    Matrix result(n, n);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (use_half_storage) {
                int idx = packIndex(i, j);
                result.set(i, j, half_data[idx]);
            } else {
                result.set(i, j, (*this)(i, j));
            }
        }
    }

    return result;
}

// Convert to dense format (overrides base class)
void SymmetricMatrix::toDense(double* result) const {
    int n = getRows();

    if (use_half_storage) {
        // Expand from half-storage to full dense
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < n; ++i) {
                int idx = packIndex(i, j);
                result[j * n + i] = half_data[idx];  // Column-major
            }
        }
    } else {
        // Use base class implementation
        Matrix::toDense(result);
    }
}

// Get upper triangular packed data
std::vector<double> SymmetricMatrix::getUpperTriangular() const {
    return half_data;
}

// Efficient symmetric matrix-vector multiplication using BLAS
Matrix SymmetricMatrix::symmetricMultiply(const Matrix& vec) const {
    if (vec.getCols() != 1) {
        throw std::invalid_argument("Vector must be a column vector");
    }

    int n = getRows();
    if (n != vec.getRows()) {
        throw std::invalid_argument("Dimension mismatch");
    }

    // Convert to dense for BLAS (can be optimized further)
    double* A_dense = new double[n * n];
    toDense(A_dense);

    double* x = new double[n];
    double* y = new double[n];

    // Extract vector
    for (int i = 0; i < n; ++i) {
        x[i] = vec(i, 0);
    }

    // Use BLAS symmetric matrix-vector multiplication
    // y = alpha * A * x + beta * y
    cblas_dsymv(CblasColMajor, CblasUpper, n,
                1.0, A_dense, n, x, 1, 0.0, y, 1);

    // Create result matrix
    Matrix result(n, 1);
    for (int i = 0; i < n; ++i) {
        result.set(i, 0, y[i]);
    }

    delete[] A_dense;
    delete[] x;
    delete[] y;

    return result;
}

// Verify symmetry property
bool SymmetricMatrix::verifySymmetry(double tolerance) const {
    int n = getRows();
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            if (std::abs((*this)(i, j) - (*this)(j, i)) > tolerance) {
                return false;
            }
        }
    }
    return true;
}

// Initialize half-storage from full matrix
void SymmetricMatrix::initHalfStorage(const Matrix& matrix) {
    int n = matrix.getRows();
    half_data.resize(n * (n + 1) / 2);

    // Store only upper triangular part
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i <= j; ++i) {
            int idx = packIndex(i, j);
            // Average symmetric elements to enforce symmetry
            double val_ij = matrix(i, j);
            double val_ji = matrix(j, i);
            half_data[idx] = (val_ij + val_ji) / 2.0;
        }
    }
}

// Create random symmetric matrix
SymmetricMatrix SymmetricMatrix::randomSymmetric(int size, double min, double max) {
    SymmetricMatrix result(size);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(min, max);

    // Generate only upper triangular part
    for (int j = 0; j < size; ++j) {
        for (int i = 0; i <= j; ++i) {
            double val = dis(gen);
            int idx = SymmetricMatrix::packIndex(i, j);
            result.half_data[idx] = val;

            // Also update base class for compatibility
            result.set(i, j, val);
            if (i != j) {
                result.set(j, i, val);
            }
        }
    }

    return result;
}

// Create random sparse symmetric matrix
SymmetricMatrix SymmetricMatrix::randomSparseSymmetric(int size, double density, double min, double max) {
    SymmetricMatrix result(size);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis_val(min, max);
    std::uniform_real_distribution<> dis_prob(0.0, 1.0);

    // Generate only upper triangular part
    for (int j = 0; j < size; ++j) {
        // Diagonal elements (higher probability)
        if (dis_prob(gen) < density * 2.0) {
            double val = dis_val(gen);
            int idx = SymmetricMatrix::packIndex(j, j);
            result.half_data[idx] = val;
            result.set(j, j, val);
        }

        // Off-diagonal upper triangular elements
        for (int i = 0; i < j; ++i) {
            if (dis_prob(gen) < density) {
                double val = dis_val(gen);
                int idx = SymmetricMatrix::packIndex(i, j);
                result.half_data[idx] = val;

                // Mirror to lower triangular in base class
                result.set(i, j, val);
                result.set(j, i, val);
            }
        }
    }

    return result;
}

// Factory method to create symmetric matrix from general matrix
SymmetricMatrix SymmetricMatrix::fromMatrix(const Matrix& matrix) {
    return SymmetricMatrix(matrix);
}

// ============================================================================
// SIMD-Optimized Operations for SymmetricMatrix
// ============================================================================

/**
 * @brief Scalar multiplication for SymmetricMatrix (maintains symmetry)
 *
 * Uses SIMD to directly multiply the half-storage data
 */
SymmetricMatrix SymmetricMatrix::operator*(double scalar) const {
    SymmetricMatrix result(*this);

    if (use_half_storage) {
        // Use SIMD-optimized multiplication on half-storage
        auto& dispatcher = SIMDDispatcher::getInstance();
        auto multiply_func = dispatcher.getScalarMultiplyFunc();

        int total = getSize() * (getSize() + 1) / 2;
        multiply_func(result.half_data.data(), total, scalar);
    }

    return result;
}

/**
 * @brief Symmetric matrix addition
 *
 * C = A + B where A and B are symmetric
 * Result C is also symmetric: C(i,j) = A(i,j) + B(i,j)
 */
SymmetricMatrix SymmetricMatrix::operator+(const SymmetricMatrix& other) const {
    if (getSize() != other.getSize()) {
        throw std::invalid_argument("SymmetricMatrix dimensions must match for addition");
    }

    SymmetricMatrix result(getSize());

    if (use_half_storage && other.use_half_storage) {
        // Use SIMD-optimized addition on half-storage
        auto& dispatcher = SIMDDispatcher::getInstance();
        auto add_func = dispatcher.getVectorAddFunc();

        int total = getSize() * (getSize() + 1) / 2;

        // Copy first matrix to result
        std::memcpy(result.half_data.data(), half_data.data(), total * sizeof(double));

        // Add second matrix
        add_func(result.half_data.data(), other.half_data.data(), total);
    }

    return result;
}

/**
 * @brief Symmetric matrix subtraction
 *
 * C = A - B where A and B are symmetric
 * Result C is also symmetric: C(i,j) = A(i,j) - B(i,j)
 */
SymmetricMatrix SymmetricMatrix::operator-(const SymmetricMatrix& other) const {
    if (getSize() != other.getSize()) {
        throw std::invalid_argument("SymmetricMatrix dimensions must match for subtraction");
    }

    SymmetricMatrix result(getSize());

    if (use_half_storage && other.use_half_storage) {
        // Use SIMD-optimized subtraction on half-storage
        auto& dispatcher = SIMDDispatcher::getInstance();
        auto subtract_func = dispatcher.getVectorSubtractFunc();

        int total = getSize() * (getSize() + 1) / 2;

        // Copy first matrix to result
        std::memcpy(result.half_data.data(), half_data.data(), total * sizeof(double));

        // Subtract second matrix
        subtract_func(result.half_data.data(), other.half_data.data(), total);
    }

    return result;
}
