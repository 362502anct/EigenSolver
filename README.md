# EigenSolver

高性能 C++17 矩阵特征值求解库，支持 OpenMP 并行计算和 LAPACK/BLAS 优化。

## 特性

- **多种算法**: QR 算法、Jacobi 方法、幂法、反幂法、分治法
- **并行计算**: 基于 OpenMP 的并行化
- **灵活输入**: Matrix Market 文件或随机矩阵
- **性能优化**: 集成 LAPACK/BLAS 加速线性代数运算

## 构建

```bash
# 安装依赖 (Ubuntu/Debian)
sudo apt-get install build-essential cmake libopenblas-dev liblapack-dev libomp-dev

# 编译
mkdir build && cd build
cmake ..
make -j$(nproc)
```

## 使用

```bash
# 随机对称矩阵
./build/eigensolver -s 100

# 从文件读取
./build/eigensolver -f matrix.mtx

# 指定算法和参数
./build/eigensolver -s 50 -m jacobi -t 1e-12 -i 2000
```

## 命令行选项

| 选项 | 说明 |
|------|------|
| `-f <file>` | Matrix Market 格式输入文件 |
| `-s <size>` | 生成随机对称矩阵的大小 |
| `-m <method>` | 算法: `qr`, `jacobi`, `power`, `inverse-power`, `divide-conquer`, `auto` |
| `-t <tol>` | 收敛容差 (默认: 1e-10) |
| `-i <iter>` | 最大迭代次数 (默认: 1000) |

## 算法说明

- **QR**: 通用算法，适合非对称矩阵
- **Jacobi**: 对称矩阵，高度并行化
- **Power**: 求最大特征值
- **Inverse Power**: 求最小特征值
- **Divide and Conquer**: 对称三对角矩阵，最快

## 性能

- Jacobi 方法: 8 核系统上可达 6-8x 加速
- QR 算法: 4 核系统上可达 2-4x 加速

## 许可证

MIT License
