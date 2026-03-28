# Grouped GEMM 编译部署与依赖指南 (FP8 重构版)

由于本项目在底层进行了深度重构并引入了极其依赖 PyTorch 前沿特性的 `FP8 CUDA Graphs` 加速，以下记录了在您的设备（包含本地开发除错）上编译和运行这个新版本所需的所有依赖要求与操作步骤。

## 1. 核心依赖环境

- **系统平台**: Linux x86_64
- **Conda 环境**: 强烈建议在专门的名为 `grouped_gemm` 的 Conda 环境中运行并隔离变量。
- **Python**: `python >= 3.10`
- **PyTorch**: `torch >= 2.1.0` *(【必须】由于项目底层依赖了原生的 `at::_scaled_mm` 算子，需确保 torch 版本足够新以包含此 API。)*
- **CUDA Toolkit**: 建议包含 `nvcc >= 12.0` (以完美支持 `float8_e4m3fn` 在设备上的代码生成栈)。
- **C++ 标准**: 需宿主机 GCC/G++ 支持 `c++17`，以满足 CUTLASS 的宏编译要求。

## 2. 从源码编译安装 (完整性能模式)

为了让程序加载我们针对 FP8 打造的极速动态引擎以及底层高性能的 `CUTLASS` 轮子，您必须包含特有的编译环境变量。

1. 首先克隆仓库并进入根目录。
2. 确保 `third_party/cutlass` 模块已经被完整拉取。
3. 执行强制编译安装指引：

**【专供：如果是您的 H100 远程服务器 (Hopper 架构) + CUDA 12.4 环境】**
H100 的 Compute Capability 为 **SM 9.0**。配合您的 CUDA 12.4，我们强烈建议您安装 **PyTorch 2.3 或更新版本（带有官方 CUDA 12.4 支持）** 以原版适配底层 `_scaled_mm` 算子并避免 C++ ABI 冲突。
编译执行指令为：
```bash
TORCH_CUDA_ARCH_LIST=9.0 GROUPED_GEMM_CUTLASS=1 pip install -e .
```
*(注意：我们已经在底层兼容了 C++ CUTLASS 对于 Sm80 / bfloat16 的架构硬编码回滚，因此不用担心编译会在 Sm90 和 FP8 数据类型的检测上产生静态报错，程序会自动将大块的 FP8 数据代理给我们写的动态高速缓冲池图引擎！)*

**【如果是这台本地开发的 RTX 4060 (Ada Lovelace 架构) 的指令】**
4060 的算力限制设定为 **SM 8.9**，其编译指令应为：
```bash
TORCH_CUDA_ARCH_LIST=8.9 GROUPED_GEMM_CUTLASS=1 pip install -e .
```
*(注意：如果您是在作为开发者进行频繁修改调试，也可以使用我们之前常用的热更新语句：`python setup.py build_ext --inplace`，它会在当前目录下生成 `.so` 库供实时调用。)*

## 3. 保守模式安装 (基础 cuBLAS)

如果不想要编译高阶的 CUTLASS 或者您是在别的机器上做简单的 Python 代码测试，可以只安装降级版（该模式无法激活很多底层的专家拼接性能优势，但依然可以用我们的新版 FP8 缓存）。可以直接运行：
```bash
pip install -e .
```

## 4. 专项验证与基准测试

### (A) FP8 精确性测试验证
如果您后续更改了任何 CUDA 代码或增加了新算子封装，必须第一时间运行纠错脚本确保 FP8 （含转置与非均匀 batch 尺寸配置）数据未发发生位移与截断：
```bash
python test_fp8_correctness.py
# 期待输出: 🏆 ALL TESTS PASSED SUCCESSFULLY!
```

### (B) MoE 发射开销极大化速度测试 
检查 `ops.py` 内的 `CUDA Graphs` 并发抓取是否生效并规避了底层发射故障，请运行超大量级别的组会乘法测试：
```bash
python benchmark.py
# 若一切顺利，在 RTX 4060 上每循环 (Time per iteration) 约为十多毫秒内。
```
