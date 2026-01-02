#ifndef GENERALMATRIX_H
#define GENERALMATRIX_H

#include "Matrix.h"

/**
 * GeneralMatrix - Wrapper class for general (non-symmetric) matrices
 * Provides clear type distinction from symmetric matrices
 * All functionality is inherited from Matrix base class
 */
class GeneralMatrix : public Matrix {
public:
    // Inherit all constructors
    using Matrix::Matrix;

    /**
     * Create general matrix from base Matrix class
     */
    explicit GeneralMatrix(const Matrix& other) : Matrix(other) {}

    /**
     * Create general matrix with specified dimensions
     */
    GeneralMatrix(int rows, int cols) : Matrix(rows, cols) {}

    /**
     * Create from CSC data
     */
    GeneralMatrix(int rows, int cols, double* values, int* row_indices, int* col_ptrs, int nnz)
        : Matrix(rows, cols, values, row_indices, col_ptrs, nnz) {}

    /**
     * Create a random general matrix
     */
    static GeneralMatrix randomGeneral(int rows, int cols, double min = -1.0, double max = 1.0);

    /**
     * Create a random sparse general matrix
     */
    static GeneralMatrix randomSparseGeneral(int rows, int cols, double density = 0.3,
                                             double min = -1.0, double max = 1.0);

    /**
     * Factory method to create general matrix from Matrix
     */
    static GeneralMatrix fromMatrix(const Matrix& matrix) {
        return GeneralMatrix(matrix);
    }

    /**
     * Get a reference to this as a Matrix (for backward compatibility)
     */
    const Matrix& asMatrix() const { return *this; }
    Matrix& asMatrix() { return *this; }
};

#endif // GENERALMATRIX_H
