import torch
import grouped_gemm as gg


if __name__ == '__main__':
    # Adapted for RTX 4060 limits with deepseek-v2 styled micro-experts
    M = 8192
    K = 2048
    N = 2048 
    E = 64
    x = (torch.randn(M, K, dtype=torch.bfloat16, device='cuda') * 0.1).to(torch.float8_e4m3fn)
    w = (torch.randn(E, K, N, dtype=torch.bfloat16, device='cuda') * 0.1).to(torch.float8_e4m3fn)

    batch_sizes = torch.tensor([M//E]*E)

    # Warmup graph builder
    for _ in range(3):
        out = gg.ops.gmm(x, w, batch_sizes)
    torch.cuda.synchronize()

    iterations = 200
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(iterations):
            out = gg.ops.gmm(x, w, batch_sizes)

    torch.cuda.synchronize()
    device_time = prof.key_averages().total_average().device_time_total
    print(f"CUDA Graph benchmark - Experts: {E}, Iterations: {iterations}")
    print(f"Total gpu time: {device_time/1000:.2f} ms")
    print(f"Time per iteration: {device_time/iterations/1000:.2f} ms")
