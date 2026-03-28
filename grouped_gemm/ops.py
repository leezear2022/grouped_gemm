import torch
from grouped_gemm import backend

class _FP8GraphCache:
    """
    静态尺寸与维度的 CUDA Graphs 执行池。
    用以彻底消除极大 Expert (如 E>120) 模型细粒度 batch 下 Python 执行微内核带来的 Launch Overhead。
    当 (batch_sizes_tuple) 及张量尺寸保持不变时，直接回放录好的 Graph。
    """
    _cache = {}

    @classmethod
    def execute(cls, a, b, batch_sizes, trans_b):
        batch_sizes_tuple = tuple(batch_sizes.tolist())
        M, K = a.shape
        E = b.shape[0]
        N = b.shape[1] if trans_b else b.shape[2]
        
        key = (M, K, N, E, trans_b, batch_sizes_tuple)
        
        # 1. 命中缓存直接更新缓冲并放图 (Replay)
        if key in cls._cache:
            runner = cls._cache[key]
            # 为了防止 Graph 的 Static Pointer 发送越界断层，
            # 这里安全地进行了原生 Copy，消除所有由于上游 Python Tensor 
            # Slice/Concat 返回的新地址在底层被 C++ Kernel 拒绝的问题。
            runner['a_buf'].copy_(a)
            runner['b_buf'].copy_(b)
            runner['graph'].replay()
            # 必须 Clone 否则上游使用者若是修改了缓存内的 out_buf 会崩盘
            return runner['out_buf'].clone()
            
        # 2. Graph 未命中，开始静态分配并录制
        a_buf = torch.empty_like(a)
        b_buf = torch.empty_like(b)
        out_buf = torch.empty((M, N), device=a.device, dtype=a.dtype)
        scale = torch.tensor(1.0, dtype=torch.float32, device=a.device)
        
        def run_loop():
            offset = 0
            for i, bs in enumerate(batch_sizes_tuple):
                if bs == 0: continue
                a_slice = a_buf[offset:offset+bs]
                b_slice = b_buf[i].t() if trans_b else b_buf[i].t().contiguous().t()
                res = torch._scaled_mm(a_slice, b_slice, scale_a=scale, scale_b=scale, out_dtype=a.dtype)[0]
                out_buf[offset:offset+bs] = res
                offset += bs
                
        # Warmup streams 供 Graph Capturing 安全抓取
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                run_loop()
        torch.cuda.current_stream().wait_stream(s)
        
        # Capturing
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            run_loop()
            
        # 注册缓冲至池中
        cls._cache[key] = {
            'a_buf': a_buf,
            'b_buf': b_buf,
            'out_buf': out_buf,
            'graph': g
        }
        
        # 为了保证当次执行正确写入 out 并立刻生效
        a_buf.copy_(a)
        b_buf.copy_(b)
        g.replay()
        return out_buf.clone()


class GroupedGemm(torch.autograd.Function):

    @staticmethod
    def forward(ctx, a, b, batch_sizes, trans_b):
        ctx.save_for_backward(a, b, batch_sizes)
        ctx.trans_b = trans_b
        
        # FP8 安全降级代理：避开底层 C++ 混合调用的 Tensor Core 数据位对齐 Bug
        # 并应用 CUDA_Graphs 消除 Kernel Launch Overhead。
        if a.dtype == torch.float8_e4m3fn:
            return _FP8GraphCache.execute(a, b, batch_sizes, trans_b)
        
        return backend.gmm(a, b, batch_sizes, trans_a=False, trans_b=trans_b)

    @staticmethod
    def backward(ctx, grad):
        grad = grad.contiguous()
        a, b, batch_sizes = ctx.saved_tensors
        trans_b = ctx.trans_b

        agrad = None
        if ctx.needs_input_grad[0]:
            agrad = backend.gmm(
                grad, b, batch_sizes, trans_a=False, trans_b=not trans_b)

        bgrad = None
        if ctx.needs_input_grad[1]:
            lhs, rhs = (grad, a) if trans_b else (a, grad)
            bgrad = backend.gmm(
                lhs, rhs, batch_sizes, trans_a=True, trans_b=False)
        return agrad, bgrad, None, None


def gmm(a, b, batch_sizes, trans_b=False):
    return GroupedGemm.apply(a, b, batch_sizes, trans_b)
