# Grouped GEMM 工程原理与实现解析

## 1. 工程简介

`grouped_gemm` 是一个为 PyTorch 提供**分组矩阵乘法（Grouped GEMM）**内核的轻量级扩展库。
它主要用于优化深度学习中需要并发执行多个不同维度矩阵乘法的场景，最典型的应用是在**混合专家模型（Mixture of Experts, MoE）**中。

在 MoE 模型结构中，输入的一个 Batch 会被路由（Router）分配给多个不同的专家（Expert）网络（通常是独立的 MLP 权重）。如果逐个为每个专家调用标准的 GEMM（矩阵乘法）操作，会产生大量的 CUDA Kernel 启动开销（Launch Overhead），并且单个专家的计算量如果不大，将无法充分利用 GPU（如 A100/H100）的强大算力。

**Grouped GEMM 的核心目的，就是通过一个单独的 CUDA Kernel 并发地计算整个 Batch 中所有专家的矩阵乘法。**

## 2. 核心原理

### 2.1 传统方法 vs Grouped GEMM
- **传统方法 / cuBLAS 模式（保守模式）**：对于 $N$ 个专家，分别启动 $N$ 个独立的 GEMM kernels。虽然容易实现，但效率低下。
- **CUTLASS 模式（高性能模式）**：这是本库的核心价值所在。它底层依赖 NVIDIA 的 [CUTLASS](https://github.com/NVIDIA/cutlass) 库，将多个维度不一、数据指针不同的 GEMM 操作“打包”到一个 Kernel 内并发执行，从而榨干 GPU 的吞吐量。

### 2.2 前反向传播全栈支持
该工程不仅实现了前向（Forward）前向传播计算，还通过继承 `torch.autograd.Function` 实现了前向和反向传播（Backward）的自动求导机制。
- 在 `grouped_gemm/ops.py` 代码中可以看到，对于输入的激活值 `a` 和权重 `b`，自动求导引擎会在反向传播时，运用转置（Transpose）技巧，反向调用后端的 Grouped GEMM 函数计算出 `a` 的梯度（`agrad`）和参数 `b` 的梯度（`bgrad`）。

## 3. Python API 设计

从开放的底层接口设计看，主要的入口函数如下：
```python
import grouped_gemm as gg

# a: 输入特征张量（例如总 Token 数量 M，特征维度 K）
# b: 专家权重张量（例如有 E 个专家，每个专家的权重为 K x N）
# batch_sizes: 一个 1D 张量，表示在这 M 个 Token 中，每个专家分到了多少个 Token。
out = gg.ops.gmm(a, b, batch_sizes)
```

**工作流**：
1. **内存预分配**：Python 后台通过 `backend._allocate_output` 函数，先推断各个专家的输入分块情况，并一次性为输出分配好整块连续显存。
2. **C++ 底层调用**：调用由 `pybind11` 绑定的 C++ 后端：`grouped_gemm_backend.gmm` 执行具体的 CUDA 计算并在显存上就地写回（In-place）。

## 4. 编译与运行特性
- 默认安装下，库会使用基础 cuBLAS 发起循环 Kernel 凑合运行。
- 要发挥其实力，需要在编译安装时开启环境变量 `GROUPED_GEMM_CUTLASS=1`，例如针对 Ampere 架构（SM 8.0）：
  `TORCH_CUDA_ARCH_LIST=8.0 GROUPED_GEMM_CUTLASS=1 pip install .`

## 5. 总结

`grouped_gemm` 是大语言模型（特别是 Mixtral 等 MoE 架构模型）推理与训练中非常关键的高价值底层轮子。它将繁复的多个小矩阵运算通过 CUTLASS 完美融合，极大加速了 GPU 端上的吞吐效率，并无缝对接到了 PyTorch 生态体系中。
