#include "MatrixMarketIO.h"
#include "util/debug.h"
#include <iostream>
#include <stdexcept>
#include <cctype>
#include <fstream>
#include <sstream>
#include <cstdio>
#include <memory>
#include <vector>

// Helper function to check if file is gzip compressed
bool isGzipFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
#ifdef DEBUG
        std::cerr << "Debug isGzipFile: Cannot open file: " << filename << std::endl;
#endif
        return false;
    }
    unsigned char magic[2];
    file.read(reinterpret_cast<char*>(magic), 2);
    file.close();
    bool is_gzip = (magic[0] == 0x1f && magic[1] == 0x8b);
#ifdef DEBUG
    std::cerr << "Debug isGzipFile: magic[0]=" << std::hex << (int)magic[0]
              << " magic[1]=" << (int)magic[1] << std::dec
              << " is_gzip=" << is_gzip << std::endl;
#endif
    return is_gzip;
}

// Helper function to decompress gzip file to a temporary string
std::string decompressGzip(const std::string& filename) {
    std::string cmd = "gunzip -c '" + filename + "'";
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) {
        throw std::runtime_error("Failed to decompress file: " + filename);
    }

    std::string result;
    char buffer[4096];
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        result += buffer;
    }

    int status = pclose(pipe);
    if (status != 0) {
        throw std::runtime_error("Failed to decompress file: " + filename);
    }

    return result;
}

Matrix MatrixMarketIO::readMatrix(const std::string& filename) {
    std::istringstream file_stream;

    // Check if file is gzip compressed and decompress if needed
    bool is_gzip = isGzipFile(filename);
#ifdef DEBUG
    std::cerr << "Debug: isGzipFile(" << filename << ") = " << is_gzip << std::endl;
#endif

    if (is_gzip) {
        try {
            std::string decompressed = decompressGzip(filename);
#ifdef DEBUG
            std::cerr << "Debug: Decompressed size = " << decompressed.length() << " bytes" << std::endl;
            std::cerr << "Debug: First 100 chars: " << decompressed.substr(0, 100) << std::endl;
#endif
            file_stream.str(decompressed);
        } catch (const std::exception& e) {
#ifdef DEBUG
            std::cerr << "Debug: Decompression error: " << e.what() << std::endl;
#endif
            throw;
        }
    } else {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file: " + filename);
        }
        std::stringstream buffer;
        buffer << file.rdbuf();
        file.close();
        file_stream.str(buffer.str());
    }
    
    std::string line;

    // Read header from file_stream
    std::getline(file_stream, line);
    if (line.substr(0, 2) != "%%") {
        throw std::runtime_error("Invalid Matrix Market file format: missing header");
    }

    // Skip comment lines (lines starting with single %, not %%)
    while (std::getline(file_stream, line) && !line.empty() && line[0] == '%' && line.substr(0, 2) != "%%") {
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
        std::getline(file_stream, line);
        if (line.empty()) continue;

        std::istringstream entry_iss(line);
        int i, j;
        double value;
        entry_iss >> i >> j >> value;

        // Matrix Market format uses 1-based indexing
        if (i >= 1 && i <= rows && j >= 1 && j <= cols) {
            matrix.set(i-1, j-1, value);
        }
    }

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
            matrix.set(i-1, j-1, value);
        }
    }
    
    file.close();
    return matrix;
}

bool MatrixMarketIO::isValidMatrixMarketFile(const std::string& filename) {
    std::string line;

#ifdef DEBUG
    std::cerr << "Debug isValidMatrixMarketFile: Checking " << filename << std::endl;
#endif

    // Check if file is gzip compressed and read accordingly
    if (isGzipFile(filename)) {
#ifdef DEBUG
        std::cerr << "Debug isValidMatrixMarketFile: File is gzip compressed" << std::endl;
#endif
        try {
            std::string decompressed = decompressGzip(filename);
#ifdef DEBUG
            std::cerr << "Debug isValidMatrixMarketFile: Decompressed size = " << decompressed.length() << std::endl;
#endif
            std::istringstream stream(decompressed);
            std::getline(stream, line);
#ifdef DEBUG
            std::cerr << "Debug isValidMatrixMarketFile: First line = '" << line << "'" << std::endl;
            std::cerr << "Debug isValidMatrixMarketFile: First 20 chars = '" << line.substr(0, 20) << "'" << std::endl;
#endif
        } catch (const std::exception& e) {
#ifdef DEBUG
            std::cerr << "Debug isValidMatrixMarketFile: Exception - " << e.what() << std::endl;
#endif
            return false;
        }
    } else {
#ifdef DEBUG
        std::cerr << "Debug isValidMatrixMarketFile: File is NOT gzip compressed" << std::endl;
#endif
        std::ifstream file(filename);
        if (!file.is_open()) {
#ifdef DEBUG
            std::cerr << "Debug isValidMatrixMarketFile: Cannot open file" << std::endl;
#endif
            return false;
        }
        std::getline(file, line);
        file.close();
#ifdef DEBUG
        std::cerr << "Debug isValidMatrixMarketFile: First line = '" << line << "'" << std::endl;
#endif
    }

    // Check if the first line starts with %%MatrixMarket
    if (line.substr(0, 14) == "%%MatrixMarket") {
#ifdef DEBUG
        std::cerr << "Debug isValidMatrixMarketFile: RETURN TRUE" << std::endl;
#endif
        return true;
    }

#ifdef DEBUG
    std::cerr << "Debug isValidMatrixMarketFile: RETURN FALSE" << std::endl;
#endif
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