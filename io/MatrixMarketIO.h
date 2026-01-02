#ifndef MATRIXMARKETIO_H
#define MATRIXMARKETIO_H

#include "../matrix/Matrix.h"
#include <string>
#include <fstream>
#include <sstream>
#include <vector>

class MatrixMarketIO {
public:
    // Read a matrix from Matrix Market format file
    static Matrix readMatrix(const std::string& filename);
    
    // Write a matrix to Matrix Market format file
    static void writeMatrix(const Matrix& matrix, const std::string& filename);
    
    // Read a sparse matrix from Matrix Market format and convert to dense
    static Matrix readSparseMatrix(const std::string& filename);
    
    // Helper function to check if a file is in Matrix Market format
    static bool isValidMatrixMarketFile(const std::string& filename);
    
    // Parse the header of a Matrix Market file
    static void parseHeader(const std::string& header_line, 
                           std::string& object, 
                           std::string& format, 
                           std::string& field, 
                           std::string& symmetry);
};

#endif // MATRIXMARKETIO_H