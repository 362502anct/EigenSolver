# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

EigenSolver is a high-performance C++17 library for computing eigenvalues of matrices using parallel algorithms. The codebase emphasizes performance optimization through OpenMP parallelization and LAPACK/BLAS integration.

## Build Commands

### Standard Build
```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### Clean Build
```bash
rm -rf build
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### Installing Dependencies

**Ubuntu/Debian:**
```bash
sudo apt-get install build-essential cmake libopenblas-dev liblapack-dev libomp-dev
```

**CentOS/RHEL/Fedora:**
```bash
sudo yum install cmake gcc gcc-c++ openblas-devel lapack-devel libgomp-devel
# or newer versions:
sudo dnf install cmake gcc gcc-c++ openblas-devel lapack-devel libgomp-devel
```

**macOS:**
```bash
brew install cmake openblas lapack llvm
```

## Running the Executable

The binary is located at `build/eigensolver`.

```bash
# Random symmetric matrix
./build/eigensolver -s 100

# From Matrix Market file
./build/eigensolver -f matrix.mtx

# Specific algorithm with custom parameters
./build/eigensolver -s 50 -m jacobi -t 1e-12 -i 2000

# Available methods: qr, jacobi, auto
```

## Architecture

### Layered Design

The codebase follows a strict layered architecture with clear dependency boundaries:

```
main (CLI orchestration)
  ↓
solver (algorithms) ← matrix (data structures)
  ↓                    ↓
util (utilities)   io (file I/O)
```

**Key principle:** Lower layers (io, util) must NOT depend on upper layers (solver, main). Dependencies flow downward only.

### Component Structure

**matrix/** - Core matrix data structures
- Dual storage: CSC (Compressed Sparse Column) format and dense arrays
- BLAS/LAPACK integration for optimized operations
- Matrix operations support both serial and OpenMP parallel execution
- Critical: All matrix operations must maintain consistency between sparse and dense representations

**solver/** - Eigenvalue computation algorithms
- Implements multiple algorithms: QR (serial/parallel), Jacobi (serial/parallel), Power Method, Inverse Power Method, Divide and Conquer
- Each algorithm is a static method that returns heap-allocated double arrays (caller owns the memory)
- Auto-selection logic in `solve()` method chooses algorithm based on matrix properties (symmetry, tridiagonal structure)
- Uses helper functions for QR decomposition, Householder transformations, Givens rotations

**io/** - Matrix Market format I/O
- Reads/writes Matrix Market format files (industry standard for sparse/dense matrices)
- Handles both coordinate (sparse) and array (dense) formats
- Performs 1-based to 0-based index conversion for Matrix Market compatibility

**util/** - Utility functions
- Performance timing and statistics
- Matrix property analysis (symmetry, definiteness, rank checks)
- Eigenvalue/eigenvector verification
- Random matrix generation
- Norm computations

**main/** - Command-line interface
- Argument parsing and validation
- Orchestrates the workflow: load/generate matrix → select algorithm → compute eigenvalues → verify results → report performance
- Initializes OpenMP and manages thread pools

### Memory Management

- Matrix class uses manual memory management with raw pointers (CSC format: `values`, `row_indices`, `col_ptrs`; dense format: `dense_data`)
- Eigenvalue solvers return `double*` arrays allocated on the heap - caller is responsible for deletion
- RAII pattern: Matrix class has proper destructor, copy constructor, and assignment operator
- Rule of thumb: If a function returns a pointer, check if you need to `delete[]` it

### Parallel Processing Strategy

- OpenMP pragmas are used throughout for parallelization
- Matrix operations have `_parallel` variants (e.g., `multiply_parallel()`)
- Both algorithm-level parallelism (Jacobi method) and operation-level parallelism (matrix multiplication)
- Thread safety is ensured through proper synchronization and avoiding shared state modifications in parallel regions

## Algorithm Selection

The `EigenSolver::solve()` method auto-selects algorithms based on matrix properties:

1. **Tridiagonal symmetric** → Divide and Conquer (fastest)
2. **Symmetric** → Jacobi Method (highly parallelizable)
3. **General square** → QR Algorithm with Wilkinson shifts

Manual selection via `-m` flag overrides auto-selection.

## Key Implementation Details

### Matrix Storage
- Default: CSC (Compressed Sparse Column) format for sparse matrices
- Dense format used when `is_dense = true` or for LAPACK operations
- Conversion functions: `toDenseMatrix()`, `toSparseMatrix()`

### BLAS/LAPACK Integration
- Direct wrapper functions in Matrix class for LAPACK operations
- Uses CBLAS interface for BLAS operations
- LAPACKE for LAPACK routines (eigenvalue computation for dense matrices)

### Error Handling
- Exception-based error propagation (primarily `std::runtime_error`)
- Input validation at CLI boundaries and in matrix operations
- No formal unit testing infrastructure - manual verification through CLI

## Common Development Patterns

When adding new eigenvalue algorithms:

1. Add static method to `EigenSolver` class (solver/EigenSolver.h)
2. Implement in solver/EigenSolver.cpp with OpenMP parallelization
3. Add algorithm case to `solve()` method's auto-selection logic
4. Update CLI help text in main/main.cpp to include new method name
5. Memory pattern: Return `new double[n]` array, caller deletes

When modifying matrix operations:

1. Maintain consistency between sparse and dense representations
2. Add OpenMP pragmas for computationally intensive operations
3. Consider both serial and parallel variants
4. Test with both sparse and dense matrix inputs

## Testing

No automated test suite exists. Manual testing workflow:

```bash
# Test with small random matrix
./build/eigensolver -s 10

# Test with specific algorithm
./build/eigensolver -s 50 -m jacobi

# Test convergence with custom tolerance
./build/eigensolver -s 100 -t 1e-12 -i 5000

# Verify results are displayed (built-in verification)
```

The main executable includes built-in result verification that checks eigenvalue accuracy using matrix properties.

## Performance Considerations

- Compiler optimization: `-O3` enabled by default in CMakeLists.txt
- OpenMP thread count controlled by `OMP_NUM_THREADS` environment variable
- For large matrices (>500x500), consider using parallel variants
- LAPACK/BLAS operations are preferred over custom implementations for dense matrices
- Sparse matrix operations are optimized for matrices with low density (<10% non-zeros)
