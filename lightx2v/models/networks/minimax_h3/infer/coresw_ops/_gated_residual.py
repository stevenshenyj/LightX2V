import torch
import triton
import triton.language as tl


@triton.jit
def _gated_residual(residual, projected, gate, indices, output,
                    count: tl.constexpr, dimension: tl.constexpr,
                    gate_stride: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < count
    rows = offsets // dimension
    columns = offsets % dimension
    group = tl.load(indices + rows, mask=valid, other=0)
    gate_value = tl.load(gate + group * gate_stride + columns, mask=valid, other=0).to(tl.float32)
    branch = tl.load(projected + offsets, mask=valid, other=0).to(tl.float32)
    # Keep the multiply/add in fp32 and let the final store cast to BF16.
    # This removes a BF16 downcast + FP32 upcast pair per element while preserving
    # the original residual-add semantics closely enough for inference testing.
    original = tl.load(residual + offsets, mask=valid, other=0).to(tl.float32)
    tl.store(output + offsets, original + gate_value * branch, mask=valid)


def _check_gated_residual_args(residual, projected, gate, indices, output=None):
    if residual.ndim != 2 or residual.shape != projected.shape or residual.numel() == 0:
        raise ValueError('Expected matching positive [S,D] residual and projected tensors')
    tensors = (residual, projected, gate) if output is None else (residual, projected, gate, output)
    if any(tensor.device != residual.device or tensor.dtype != torch.bfloat16 for tensor in tensors):
        raise ValueError('Expected BF16 tensors on one GPU')
    if not residual.is_cuda or not residual.is_contiguous() or not projected.is_contiguous():
        raise ValueError('Expected contiguous GPU residual and projected tensors')
    if output is not None and (output.shape != residual.shape or not output.is_cuda or not output.is_contiguous()):
        raise ValueError('Expected contiguous GPU output matching residual')
    if gate.ndim != 2 or gate.shape[1] != residual.shape[1] or gate.shape[0] < 1 or gate.stride(1) != 1:
        raise ValueError('Expected [groups,D] gate with contiguous columns')
    if indices.shape != (residual.shape[0],) or indices.device != residual.device or not indices.is_contiguous() or indices.dtype not in (torch.int32, torch.int64):
        raise ValueError('Expected contiguous per-token int32/int64 indices on the same GPU')
    if any(tensor.requires_grad for tensor in tensors):
        raise ValueError('Inference only')


def gated_residual_out(residual, projected, gate, indices, output):
    """Fused output = residual + gate[indices] * projected into caller-provided output.

    This is intended for chunked TP pipelines: pass row-contiguous slices for
    residual/projected/indices/output so the pointwise slice + mul + add happens
    immediately after each chunk's communication completes.
    """
    _check_gated_residual_args(residual, projected, gate, indices, output)
    with torch.cuda.device(residual.device):
        _gated_residual[(triton.cdiv(residual.numel(), 1024),)](
            residual, projected, gate, indices, output, residual.numel(), residual.shape[1], gate.stride(0),
            BLOCK=1024, num_warps=4, enable_fp_fusion=False)
    return output


def gated_residual(residual, projected, gate, indices):
    _check_gated_residual_args(residual, projected, gate, indices)
    output = torch.empty_like(residual)
    return gated_residual_out(residual, projected, gate, indices, output)
