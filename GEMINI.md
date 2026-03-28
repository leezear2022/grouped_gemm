# Grouped GEMM Project Context

This project is a lightweight PyTorch library exposing grouped GEMM (General Matrix Multiplication) kernels, primarily designed for Mixture of Experts (MoE) models.

## Project Overview

- **Purpose:** Optimizes MoE layers by fusing multiple GEMM operations into a single CUDA kernel to reduce launch overhead and increase throughput.
- **Main Technologies:**
    - **PyTorch:** Frontend and autograd support.
    - **CUDA:** High-performance GPU kernels.
    - **CUTLASS:** The preferred backend for grouped GEMM operations.
    - **cuBLAS:** Fallback backend when CUTLASS is not enabled or for certain data types.
    - **C++17:** Required by CUTLASS and used for the PyBind11 extension.

## Architecture

- **Python API (`grouped_gemm/ops.py`):** Provides the `gmm` function which wraps a `torch.autograd.Function`. It handles output allocation and provides a fallback for certain types (like FP8) if the backend has known issues.
- **C++ Backend (`csrc/`):**
    - `ops.cu`: PyBind11 module definition.
    - `grouped_gemm.cu`: Core implementation of grouped GEMM using either CUTLASS or cuBLAS.
    - `fill_arguments.cuh`: CUDA kernels for preparing CUTLASS argument arrays on the device.
    - `grouped_gemm.h`: Header for the backend functions.

## Building and Running

### Installation

To install the package in "conservative" mode (cuBLAS fallback only):
```bash
pip install .
```

To enable high-performance CUTLASS kernels (requires `third_party/cutlass` to be initialized):
```bash
TORCH_CUDA_ARCH_LIST=<arch_list> GROUPED_GEMM_CUTLASS=1 pip install .
```
Example for Ampere (SM 8.0):
```bash
TORCH_CUDA_ARCH_LIST=8.0 GROUPED_GEMM_CUTLASS=1 pip install .
```

### Benchmarking
Run the benchmark script to measure performance:
```bash
python benchmark.py
```

### Testing
There are several test scripts in the root directory, many focusing on FP8 correctness and performance:
- `test_fp8_correctness.py`
- `test_fp8_mm.py`
- `test_strides.py`

## Development Conventions

- **Grouped Execution:** The library aims to execute $N$ GEMMs in a single kernel.
- **Handling `k=0`:** CUTLASS has known issues with $k=0$ dimensions. The project manually zeroes out outputs and sets $m, n$ to 0 for these cases in `csrc/grouped_gemm.cu`.
- **FP8 Support:** The project is currently focused on FP8 (`torch.float8_e4m3fn`) support. Note that some parts of the CUDA code strictly enforce FP8 input types, while templates might still reference `bfloat16_t`.
- **Transposition:** Supports `trans_a` (for backward pass) or `trans_b` (weight transposition), but typically not both simultaneously.

## Key Files

- `setup.py`: Build configuration using `torch.utils.cpp_extension`.
- `csrc/grouped_gemm.cu`: The heart of the CUDA implementation.
- `grouped_gemm/ops.py`: The main entry point for PyTorch users.
- `project_understanding.md`: High-level explanation of the project's principles.


> [!IMPORTANT]
> 1. 以后所有的开发及运行调试都必须基于这个 `grouped_gemm` conda 环境。
> 2. 全程必须使用中文交流。
> 3. 思维链的信息也必须全部用中文撰写。