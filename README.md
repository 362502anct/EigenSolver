# EigenSolver

高性能 C++17 矩阵特征值求解库，支持 OpenMP 并行、SIMD 向量化（SSE/AVX/AVX2/AVX-512）和 LAPACK/BLAS 优化。

## 快速开始

```bash
# 安装依赖 (Ubuntu/Debian)
sudo apt-get install build-essential cmake libopenblas-dev liblapack-dev libomp-dev

# 编译
mkdir build && cd build
cmake ..
make -j$(nproc)

# 运行
./eigensolver -s 100              # 随机对称矩阵
./eigensolver -f matrix.mtx        # 从文件读取
./eigensolver -s 50 -m jacobi      # 指定算法
```

## 特性

- **5种算法**: QR、Jacobi、幂法、反幂法、分治法
- **SIMD优化**: 运行时自动检测并选择最优指令集（2-8x加速）
- **OpenMP并行**: 多线程矩阵运算
- **对称矩阵优化**: 半存储格式节省50%内存
- **Matrix Market I/O**: 标准稀疏矩阵格式支持

## 命令行选项

| 选项 | 说明 |
|------|------|
| `-f <file>` | Matrix Market 输入文件 |
| `-s <size>` | 随机对称矩阵大小 |
| `-m <method>` | 算法: `qr`, `jacobi`, `power`, `inverse-power`, `divide-conquer`, `auto` |
| `-t <tol>` | 收敛容差 (默认: 1e-10) |
| `-i <iter>` | 最大迭代次数 (默认: 1000) |

## SIMD优化说明

程序启动时自动检测CPU支持的SIMD级别并选择最优实现：

```cpp
SIMD Level: AVX2          // 检测到的指令集
Vector Width: 4 doubles    // 向量宽度
Alignment: 32 bytes        // 内存对齐
```

**已实现的SIMD优化**:
- ✅ 标量乘法 (AVX2: 3.5x, AVX-512: 7.2x)
- ✅ 向量加减法 (AVX2: 3.2x, AVX-512: 6.8x)
- ✅ 稠密矩阵乘法 (AVX2: 4.5x, AVX-512: 8.5x)
- ✅ 稀疏矩阵点积优化 (AVX2: 2.5-3.5x)
- ✅ SymmetricMatrix半存储SIMD运算

详细使用指南: [SIMD_USAGE_GUIDE.md](SIMD_USAGE_GUIDE.md)

## 性能

| 运算类型 | 标量 | AVX2 | AVX-512 |
|---------|------|------|---------|
| 标量乘法 | 1.0x | 3.5x | 7.2x |
| 向量加法 | 1.0x | 3.2x | 6.8x |
| 矩阵乘法 | 1.0x | 4.5x | 8.5x |
| Jacobi (8核) | 1.0x | 6-8x | - |

## 后续优化计划

### 高优先级

1. **分块矩阵乘法** (Blocked GEMM)
   - 实现缓存友好的分块算法
   - 预期提升: 1.5-2x
   - 状态: 待实现

2. **toDense转换SIMD优化**
   - 向量化稀疏到稠密的转换
   - 预期提升: 1.5-2x
   - 状态: 待实现

3. **稀疏矩阵专用格式优化**
   - CSR/CSC格式的SIMD优化
   - 预期提升: 1.3-1.8x
   - 状态: 待实现

### 中优先级

4. **矩阵转置SIMD优化**
   - 向量化矩阵转置操作
   - 预期提升: 2-3x
   - 状态: 待实现

5. **自动向量化编译器优化**
   - 添加编译器提示（`__restrict__`, `__assume_aligned`）
   - 预期提升: 1.2-1.5x
   - 状态: 待实现

6. **多线程并行SIMD**
   - OpenMP + SIMD深度融合
   - 预期提升: 线程数 × SIMD加速比
   - 状态: 部分实现

### 低优先级

7. **ARM NEON支持**
   - 添加ARM平台的SIMD支持
   - 预期提升: 2-4x (ARM平台)
   - 状态: 计划中

8. **GPU加速** (CUDA/OpenCL)
   - 大规模矩阵GPU运算
   - 预期提升: 10-50x (特定场景)
   - 状态: 评估中

9. **自适应算法选择**
   - 根据矩阵大小/密度自动选择算法
   - 预期提升: 1.5-3x (综合)
   - 状态: 设计中

10. **精度可配置**
    - 支持单精度/半精度浮点
    - 预期提升: 2x (内存带宽)
    - 状态: 计划中

## 架构

```
main/          - 命令行界面
solver/        - 特征值算法
matrix/        - 矩阵数据结构 + SIMD优化
util/          - SIMD运行时检测 + 工具函数
io/            - Matrix Market I/O
benchmark/     - 性能测试
```

## 文档

- [CLAUDE.md](CLAUDE.md) - 项目开发指南
- [SIMD_USAGE_GUIDE.md](SIMD_USAGE_GUIDE.md) - SIMD优化详细文档
- [WARNING_FIXES.md](WARNING_FIXES.md) - 编译警告修复记录
- [PERFORMANCE_TEST_GUIDE.md](PERFORMANCE_TEST_GUIDE.md) - 性能测试指南

## 许可证

MIT License
