#ifndef SYMMETRICMATRIX_H
#define SYMMETRICMATRIX_H

#include "Matrix.h"
#include <memory>
#include <vector>

/**
 * SymmetricMatrix - Specialized matrix class for symmetric matrices
 * Uses half-storage optimization (only stores upper triangular part)
 * Storage format: Upper triangular packed (column-major)
 */
class SymmetricMatrix : public Matrix {
public:
    /**
     * Default constructor
     */
    SymmetricMatrix();

    /**
     * Create symmetric matrix with specified size
     */
    explicit SymmetricMatrix(int size);

    /**
     * Create from full matrix (enforces symmetry)
     */
    explicit SymmetricMatrix(const Matrix& other);

    /**
     * Create from half-storage data
     * @param size Matrix dimension (n x n)
     * @param upper_tri Upper triangular packed data (length = size*(size+1)/2)
     */
    SymmetricMatrix(int size, const double* upper_tri);

    /**
     * Create from CSC sparse data (ensures symmetry)
     */
    SymmetricMatrix(int size, double* values, int* row_indices, int* col_ptrs, int nnz);

    /**
     * Destructor
     */
    ~SymmetricMatrix();

    /**
     * Copy constructor
     */
    SymmetricMatrix(const SymmetricMatrix& other);

    /**
     * Assignment operator
     */
    SymmetricMatrix& operator=(const SymmetricMatrix& other);

    /**
     * Get element value (with coordinate mapping for half-storage)
     * Automatically handles A(i,j) = A(j,i) for i > j
     */
    double operator()(int row, int col) const override;

    /**
     * Set element value (maintains symmetry automatically)
     */
    void set(int row, int col, double value) override;

    /**
     * Get matrix size (symmetric matrices are square)
     */
    int getSize() const { return getRows(); }

    /**
     * Check if using half-storage optimization
     */
    bool isHalfStorage() const { return use_half_storage; }

    /**
     * Convert to full Matrix representation
     */
    Matrix toFullMatrix() const;

    /**
     * Get upper triangular packed data
     * Returns array of length n*(n+1)/2
     */
    std::vector<double> getUpperTriangular() const;

    /**
     * Efficient matrix-vector multiplication using BLAS dsymv
     */
    Matrix symmetricMultiply(const Matrix& vec) const;

    /**
     * Verify symmetry property
     */
    bool verifySymmetry(double tolerance = 1e-10) const;

    /**
     * Factory: Create random symmetric matrix
     */
    static SymmetricMatrix randomSymmetric(int size, double min = -1.0, double max = 1.0);

    /**
     * Factory: Create random sparse symmetric matrix
     */
    static SymmetricMatrix randomSparseSymmetric(int size, double density = 0.3,
                                                 double min = -1.0, double max = 1.0);

    /**
     * Factory: Create from general matrix
     */
    static SymmetricMatrix fromMatrix(const Matrix& matrix);

    /**
     * Convert to dense format (overrides base class for half-storage)
     */
    void toDense(double* result) const override;

    /**
     * SIMD-optimized scalar multiplication (returns SymmetricMatrix)
     */
    SymmetricMatrix operator*(double scalar) const;

    /**
     * Symmetric matrix addition (returns SymmetricMatrix)
     * Result is also symmetric: (A + B)^T = A^T + B^T = A + B
     */
    SymmetricMatrix operator+(const SymmetricMatrix& other) const;

    /**
     * Symmetric matrix subtraction (returns SymmetricMatrix)
     * Result is also symmetric: (A - B)^T = A^T - B^T = A - B
     */
    SymmetricMatrix operator-(const SymmetricMatrix& other) const;

    /**
     * Get as Matrix reference (for backward compatibility)
     */
    const Matrix& asMatrix() const { return *this; }
    Matrix& asMatrix() { return *this; }

private:
    bool use_half_storage = true;  // Enable half-storage optimization
    std::vector<double> half_data;  // Upper triangular packed data

    /**
     * Convert 2D indices to 1D packed index (upper triangular)
     * Maps A(i,j) where 0 <= i <= j < n to packed index
     * Formula: idx = j*(j+1)/2 + i
     */
    static int packIndex(int i, int j) {
        // Ensure i <= j (swap if necessary)
        if (i > j) std::swap(i, j);
        return j * (j + 1) / 2 + i;
    }

    /**
     * Initialize half-storage from full matrix
     */
    void initHalfStorage(const Matrix& matrix);
};

#endif // SYMMETRICMATRIX_H
