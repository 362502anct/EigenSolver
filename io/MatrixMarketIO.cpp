#include "MatrixMarketIO.h"
#include <iostream>
#include <stdexcept>
#include <cctype>

Matrix MatrixMarketIO::readMatrix(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    
    std::string line;
    
    // Read header
    std::getline(file, line);
    if (line.substr(0, 2) != "%%") {
        throw std::runtime_error("Invalid Matrix Market file format: missing header");
    }

    // Skip comment lines (lines starting with single %, not %%)
    while (std::getline(file, line) && !line.empty() && line[0] == '%' && line.substr(0, 2) != "%%") {
        // Skip comment lines
    }
    
    // Parse the size information
    std::istringstream iss(line);
    int rows, cols, entries;
    iss >> rows >> cols >> entries;
    
    // Create matrix
    Matrix matrix(rows, cols);
    
    // Read the matrix data
    for (int idx = 0; idx < entries; ++idx) {
        std::getline(file, line);
        if (line.empty()) continue;
        
        std::istringstream entry_iss(line);
        int i, j;
        double value;
        entry_iss >> i >> j >> value;
        
        // Matrix Market format uses 1-based indexing
        if (i >= 1 && i <= rows && j >= 1 && j <= cols) {
            matrix(i-1, j-1) = value;
        }
    }
    
    file.close();
    return matrix;
}

void MatrixMarketIO::writeMatrix(const Matrix& matrix, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot create file: " + filename);
    }
    
    // Write header
    file << "%%MatrixMarket matrix coordinate real general" << std::endl;
    file << "% Created by EigenSolver" << std::endl;
    
    // Count non-zero entries for coordinate format
    int non_zero_count = 0;
    int rows = matrix.getRows();
    int cols = matrix.getCols();
    
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if (matrix(i, j) != 0.0) {
                non_zero_count++;
            }
        }
    }
    
    // Write size information
    file << rows << " " << cols << " " << non_zero_count << std::endl;
    
    // Write non-zero entries
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if (matrix(i, j) != 0.0) {
                // Use 1-based indexing for Matrix Market format
                file << (i + 1) << " " << (j + 1) << " " << matrix(i, j) << std::endl;
            }
        }
    }
    
    file.close();
}

Matrix MatrixMarketIO::readSparseMatrix(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    
    std::string line;
    
    // Read header
    std::getline(file, line);
    if (line.substr(0, 2) != "%%") {
        throw std::runtime_error("Invalid Matrix Market file format: missing header");
    }

    // Skip comment lines (lines starting with single %, not %%)
    while (std::getline(file, line) && !line.empty() && line[0] == '%' && line.substr(0, 2) != "%%") {
        // Skip comment lines
    }
    
    // Parse the size information
    std::istringstream iss(line);
    int rows, cols, entries;
    iss >> rows >> cols >> entries;
    
    // Create matrix
    Matrix matrix(rows, cols);
    
    // Read the sparse matrix data
    for (int idx = 0; idx < entries; ++idx) {
        std::getline(file, line);
        if (line.empty()) continue;
        
        std::istringstream entry_iss(line);
        int i, j;
        double value;
        entry_iss >> i >> j >> value;
        
        // Matrix Market format uses 1-based indexing
        if (i >= 1 && i <= rows && j >= 1 && j <= cols) {
            matrix(i-1, j-1) = value;
        }
    }
    
    file.close();
    return matrix;
}

bool MatrixMarketIO::isValidMatrixMarketFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        return false;
    }
    
    std::string line;
    std::getline(file, line);
    
    // Check if the first line starts with %%MatrixMarket
    if (line.substr(0, 14) == "%%MatrixMarket") {
        file.close();
        return true;
    }
    
    file.close();
    return false;
}

void MatrixMarketIO::parseHeader(const std::string& header_line, 
                                std::string& object, 
                                std::string& format, 
                                std::string& field, 
                                std::string& symmetry) {
    // Expected format: %%MatrixMarket <object> <format> <field> <symmetry>
    std::istringstream iss(header_line.substr(2)); // Skip %%
    
    std::string matrix_market;
    iss >> matrix_market >> object >> format >> field >> symmetry;
}