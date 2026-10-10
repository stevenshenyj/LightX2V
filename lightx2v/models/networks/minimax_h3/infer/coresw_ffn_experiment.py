import os

import torch
import torch.distributed as dist
import torch.nn.functional as F
from loguru import logger


def _env_enabled(name, default=True):
    value = os.environ.get(name)
    if value is None:
        return default
    return value not in ("0", "false", "False", "no", "NO", "off", "OFF")


def _active(owner):
    return bool(owner.config.get("coresw_ffn_experiment", False)) or _env_enabled("CORESW_FFN_EXPERIMENT", False)


def _env_int(name, default):
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _report_once(owner, key, message, level="info"):
    reported = getattr(owner, "_coresw_ffn_reports", set())
    if key in reported:
        return
    reported.add(key)
    owner._coresw_ffn_reports = reported
    getattr(logger, level)("[CORESW_FFN_EXPERIMENT] rank={} {}", os.environ.get("RANK", "?"), message)


def _local_weight(module, group, split, shape):
    from lightx2v.common.ops.mm.mm_weight import MMWeight, MMWeightTP, unwrap_tp_weight

    if not isinstance(module, MMWeightTP) or module.tp_size != 2 or module.tp_group is not group or module.split_dim != split:
        raise ValueError(f"requires TP2 {split} MMWeightTP, got {type(module)}")
    inner = unwrap_tp_weight(module)
    if type(inner) is not MMWeight or inner.has_lora_branch or inner.has_diff:
        raise ValueError("requires unquantized BF16 MMWeight without LoRA/diff")
    if getattr(inner, "bias", None) is not None or getattr(module, "_row_split_bias", None) is not None:
        raise ValueError("requires no bias")
    if split == "row" and not module.reduce_output:
        raise ValueError("row projections require reduce_output=True")
    if split == "col" and module.lora_column_chunks != 2:
        raise ValueError("FFN up requires paired local SwiGLU halves, lora_column_chunks=2")
    weight = inner._get_actual_weight()
    if tuple(weight.shape) != shape or weight.dtype != torch.bfloat16 or weight.stride() != (1, shape[0]):
        raise ValueError(f"unexpected weight layout shape={tuple(weight.shape)} stride={weight.stride()} dtype={weight.dtype}; expected {shape} stride={(1, shape[0])} bf16")
    return weight


def _basic_supported(owner, inputs, residual, gate, indices):
    if owner.tp_size != 2 or owner.tp_group is None or owner.seq_p_group is not None:
        return False, "requires TP2 without sequence parallelism"
    if inputs.ndim != 2 or inputs.shape[1] != 5376 or residual.shape != inputs.shape:
        return False, f"requires [S,5376] inputs/residual, got {tuple(inputs.shape)} {tuple(residual.shape)}"
    if inputs.dtype != torch.bfloat16 or residual.dtype != torch.bfloat16 or gate.dtype != torch.bfloat16:
        return False, "requires BF16 inputs/residual/gate"
    if not inputs.is_cuda or inputs.device != residual.device or gate.device != inputs.device or indices.device != inputs.device:
        return False, "requires same CUDA device tensors"
    if not inputs.is_contiguous() or not residual.is_contiguous() or not indices.is_contiguous():
        return False, "requires contiguous inputs/residual/indices"
    if any(t.requires_grad for t in (inputs, residual, gate)):
        return False, "inference only"
    if gate.ndim != 2 or gate.shape[1] != 5376 or gate.stride(1) != 1:
        return False, f"requires gate [groups,5376] contiguous columns, got shape={tuple(gate.shape)} stride={gate.stride()}"
    if indices.shape != (inputs.shape[0],) or indices.dtype not in (torch.int32, torch.int64):
        return False, f"requires indices [S] int32/int64, got shape={tuple(indices.shape)} dtype={indices.dtype}"
    return True, "ok"


def feed_forward_residual(owner, weights, inputs, residual, gate, indices):
    """Return full residual-applied FFN output, or None to use original path."""
    if not _active(owner):
        return None
    ok, reason = _basic_supported(owner, inputs, residual, gate, indices)
    if not ok:
        _report_once(owner, "fallback_basic_" + reason, "fallback: " + reason, "warning")
        return None

    use_up = _env_enabled("CORESW_FFN_UP", True)
    use_down = _env_enabled("CORESW_FFN_DOWN", False)
    use_reduce = _env_enabled("CORESW_FFN_REDUCE", False)
    try:
        if use_up:
            up_weight = _local_weight(weights.in_proj, owner.tp_group, "col", (5376, 14336))
        if use_down:
            down_weight = _local_weight(weights.out_proj, owner.tp_group, "row", (7168, 5376))

        if use_up and use_down and use_reduce:
            from .coresw_ops._ffn_pipeline import BufferedFFN

            workspace = getattr(owner, "_coresw_ffn_workspace", None)
            if workspace is None:
                # Same sizing as CoreSW: chunk by token rows for overlap/reduce.
                workspace = BufferedFFN(inputs.shape[0], 8192, inputs.device, owner.tp_group)
                owner._coresw_ffn_workspace = workspace
            if workspace.rows != inputs.shape[0] or workspace.device != inputs.device:
                raise ValueError("workspace shape/device changed")
            _report_once(owner, "active_up_down_reduce", "ACTIVE FFN_UP + FFN_DOWN + TP2 fused reduce/residual")
            return workspace.run(inputs, up_weight.t(), down_weight, residual, gate, indices)

        if use_up:
            from .coresw_ops._ffn_up import fused_up

            _report_once(owner, "active_up_only", "ACTIVE FFN_UP fused _up_swiglu only; out_proj remains original TP all_reduce")
            activation = fused_up(inputs, up_weight.t())
        else:
            value, activation_gate = weights.in_proj.apply(inputs).chunk(2, dim=-1)
            activation = value * F.silu(activation_gate)

        if use_down:
            from .coresw_ops._ffn_down import down_out

            if _env_enabled("CORESW_FFN_RCCL_OVERLAP", False):
                chunk_rows = max(128, _env_int("CORESW_FFN_CHUNK_ROWS", 8192))
                chunk_rows = ((chunk_rows + 127) // 128) * 128
                _report_once(
                    owner,
                    "active_down_rccl_overlap",
                    f"ACTIVE FFN_DOWN chunked GEMM + async RCCL all_reduce overlap + torch residual chunk_rows={chunk_rows}",
                )
                from .coresw_ops._gated_residual import gated_residual_out

                projected = torch.empty_like(residual)
                output = torch.empty_like(residual)
                nrows = activation.shape[0]
                works = []
                for first in range(0, nrows, chunk_rows):
                    last = min(first + chunk_rows, nrows)
                    down_out(activation[first:last], down_weight, projected[first:last])
                    work = dist.all_reduce(
                        projected[first:last],
                        op=dist.ReduceOp.SUM,
                        group=owner.tp_group,
                        async_op=True,
                    )
                    works.append((work, first, last))
                for work, first, last in works:
                    work.wait()
                    # Fuse the previous gate.index_select slice + mul + add into one Triton
                    # kernel per communication chunk, and keep it inside the pipeline tail.
                    gated_residual_out(
                        residual[first:last],
                        projected[first:last],
                        gate,
                        indices[first:last],
                        output[first:last],
                    )
                return output
            _report_once(owner, "active_down_rccl", "ACTIVE FFN_DOWN custom GEMM + original RCCL all_reduce + torch residual")
            projected = torch.empty_like(residual)
            down_out(activation, down_weight, projected)
            dist.all_reduce(projected, op=dist.ReduceOp.SUM, group=owner.tp_group)
        else:
            projected = weights.out_proj.apply(activation)
        return residual + gate.index_select(0, indices) * projected
    except Exception as exc:
        if _env_enabled("CORESW_FFN_STRICT", False):
            raise
        _report_once(owner, "fallback_exception_" + type(exc).__name__, f"fallback after {type(exc).__name__}: {exc}", "warning")
        return None


def attention_out_projection(owner, module, inputs):
    """Env-gated attention out projection with chunked GEMM + async RCCL all_reduce overlap.

    Returns projected [S,5376] tensor, or None to use the original MMWeightTP.apply path.
    This targets MiniMax-H3 TP2 attention out_proj: [S,3584] x [3584,5376].
    """
    if not _env_enabled("CORESW_ATTN_OUT_RCCL_OVERLAP", False):
        return None
    try:
        if owner.tp_size != 2 or owner.tp_group is None or owner.seq_p_group is not None:
            raise ValueError("requires TP2 without sequence parallelism")
        if inputs.ndim != 2 or inputs.shape[1] != 3584:
            raise ValueError(f"requires attention output [S,3584], got {tuple(inputs.shape)}")
        if inputs.dtype != torch.bfloat16 or not inputs.is_cuda:
            raise ValueError(f"requires CUDA BF16 input, got device={inputs.device} dtype={inputs.dtype}")
        if inputs.requires_grad:
            raise ValueError("inference only")
        if not inputs.is_contiguous():
            inputs = inputs.contiguous()
        weight = _local_weight(module, owner.tp_group, "row", (3584, 5376))
        chunk_rows = max(128, _env_int("CORESW_ATTN_OUT_CHUNK_ROWS", _env_int("CORESW_FFN_CHUNK_ROWS", 8192)))
        chunk_rows = ((chunk_rows + 127) // 128) * 128
        _report_once(
            owner,
            "active_attn_out_rccl_overlap",
            f"ACTIVE ATTN_OUT chunked GEMM + async RCCL all_reduce overlap chunk_rows={chunk_rows}",
        )
        output = torch.empty((inputs.shape[0], 5376), dtype=inputs.dtype, device=inputs.device)
        works = []
        for first in range(0, inputs.shape[0], chunk_rows):
            last = min(first + chunk_rows, inputs.shape[0])
            torch.mm(inputs[first:last], weight, out=output[first:last])
            work = dist.all_reduce(output[first:last], op=dist.ReduceOp.SUM, group=owner.tp_group, async_op=True)
            works.append(work)
        for work in works:
            work.wait()
        return output
    except Exception as exc:
        if _env_enabled("CORESW_ATTN_OUT_STRICT", False) or _env_enabled("CORESW_FFN_STRICT", False):
            raise
        _report_once(owner, "fallback_attn_out_" + type(exc).__name__, f"ATTN_OUT fallback after {type(exc).__name__}: {exc}", "warning")
        return None
