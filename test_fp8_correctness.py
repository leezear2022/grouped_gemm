import torch
import grouped_gemm as gg
import sys

def test_correctness(M, K, N, E, trans_b):
    print(f"\n==================================================")
    print(f"Testing M={M}, K={K}, N={N}, E={E}, trans_b={trans_b}")
    a = (torch.randn(M, K, dtype=torch.bfloat16, device='cuda') * 0.1).to(torch.float8_e4m3fn)
    
    if trans_b:
        b = (torch.randn(E, N, K, dtype=torch.bfloat16, device='cuda') * 0.1).to(torch.float8_e4m3fn)
    else:
        b = (torch.randn(E, K, N, dtype=torch.bfloat16, device='cuda') * 0.1).to(torch.float8_e4m3fn)

    # 随机生成每个分块的大小
    batch_sizes = torch.randint(16, (M//E) + 16, (E,), device='cpu')
    batch_sizes = (batch_sizes // 16) * 16 # 保证对齐到 16，这是 cuBLASLt 对 FP8 的通常要求
    batch_sizes[-1] = M - batch_sizes[:-1].sum()
    if batch_sizes[-1] <= 0 or batch_sizes[-1] % 16 != 0:
        # Fallback 到平均分配
        batch_sizes = torch.tensor([M//E]*E, device='cpu')
        batch_sizes[-1] += M - batch_sizes.sum()

    print(f"Batch sizes: {batch_sizes.tolist()}")

    # 1. 我们的 C++ grouped_gemm API 结果
    try:
        out_gg = gg.ops.gmm(a, b, batch_sizes, trans_b=trans_b)
    except Exception as e:
        print(f"[-] Grouped gemm error: {e}")
        return False

    # 2. Python 手写 Baseline: 逐块切片并调用 torch._scaled_mm
    # 分配同样格式的输出以避免未初始化内存差异
    out_baseline = torch.zeros_like(out_gg)
    offset = 0
    scale_a = torch.tensor(1.0, dtype=torch.float32, device='cuda')
    scale_b = torch.tensor(1.0, dtype=torch.float32, device='cuda')
    
    try:
        for i in range(E):
            bs = batch_sizes[i].item()
            if bs == 0: continue
            a_slice = a[offset:offset+bs, :]
            b_slice = b[i]
            
            # 对于 PyTorch _scaled_mm_out，必须模拟一样的数据排布
            if trans_b:
                b_slice = b_slice.t()
            else:
                b_slice = b_slice.t().contiguous().t()
                
            res = torch._scaled_mm(a_slice, b_slice, scale_a=scale_a, scale_b=scale_b, out_dtype=out_gg.dtype)
            out_slice = res[0]
            out_baseline[offset:offset+bs, :] = out_slice
            offset += bs
    except Exception as e:
        print(f"[-] PyTorch baseline error: {e}")
        return False

    # 对比结果的最大绝对误差 
    diff = (out_gg.float() - out_baseline.float()).abs().max().item()
    print(f"[+] Max absolute difference: {diff}")
    
    if diff >= 1e-4:
        # Check if they are bitwise identical
        match_mask = (out_gg == out_baseline)
        match_ratio = match_mask.float().mean().item()
        print(f"[-] Bitwise match ratio: {match_ratio*100:.2f}%")
        
        # Print a sample of mismatched values
        mismatch_idx = torch.where(~match_mask)
        if len(mismatch_idx[0]) > 0:
            samp_i, samp_j = mismatch_idx[0][0], mismatch_idx[1][0]
            print(f"[-] First mismatch at [{samp_i}, {samp_j}]: GG={out_gg[samp_i, samp_j].float().item()} vs Baseline={out_baseline[samp_i, samp_j].float().item()}")

    if diff < 1e-4:
        print("[+] Correctness PASSED.")
        return True
    else:
        print("[-] Correctness FAILED!")
        return False

configs = [
    # (M, K, N, Experts, trans_b)
    (1024, 256, 128, 4, False),
    (1024, 256, 128, 4, True),
    (4096, 512, 512, 8, False),
    (4096, 512, 512, 8, True),
    (512, 1024, 2048, 2, False),
    (2048, 64, 64, 16, True),   # 窄维度 + 大规模分块测试
]

all_passed = True
for config in configs:
    if not test_correctness(*config):
        all_passed = False
        break

print("\n==================================================")
if all_passed:
    print("🏆 ALL TESTS PASSED SUCCESSFULLY!")
else:
    print("❌ SOME TESTS FAILED.")
    sys.exit(1)
