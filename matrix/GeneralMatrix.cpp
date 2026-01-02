#include "GeneralMatrix.h"
#include <random>

// Create random general matrix
GeneralMatrix GeneralMatrix::randomGeneral(int rows, int cols, double min, double max) {
    Matrix base = Matrix::random(rows, cols, min, max);
    return GeneralMatrix(base);
}

// Create random sparse general matrix
GeneralMatrix GeneralMatrix::randomSparseGeneral(int rows, int cols, double density, double min, double max) {
    GeneralMatrix result(rows, cols);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis_val(min, max);
    std::uniform_real_distribution<> dis_prob(0.0, 1.0);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if (dis_prob(gen) < density) {
                result.set(i, j, dis_val(gen));
            }
        }
    }

    return result;
}
