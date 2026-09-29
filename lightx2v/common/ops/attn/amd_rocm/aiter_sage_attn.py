# Target: lightx2v/common/ops/attn/amd_rocm/aiter_sage_attn.py
# aiter_sage_attn: int8/fp8 quantization offloaded to aiter's fused
# Triton sage_quant kernel (no PyTorch float32 upcast), then feeds the gfx1201
# FlyDSL build_sage_attention_v2_core core. Registered as "aiter_sage_attn_fused".
import importlib
import importlib.util
import os

import torch
from loguru import logger

from lightx2v.utils.registry_factory import ATTN_WEIGHT_REGISTER

from ..template import AttnWeightTemplate


def _load_sage_builder():
    try:
        from aiter.ops.flydsl.kernels.sage_attention_gfx1201 import (
            build_sage_attention_v2_core,
        )

        return build_sage_attention_v2_core
    except Exception as exc:  # noqa: BLE001
        path = os.environ.get("LIGHTX2V_AITER_SAGE_PATH")
        if path and os.path.isfile(path):
            try:
                spec = importlib.util.spec_from_file_location("aiter_sage_gfx1201", path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                return module.build_sage_attention_v2_core
            except Exception as fexc:  # noqa: BLE001
                logger.info(f"aiter sage fallback load failed: {fexc}")
                return None
        logger.info(f"aiter sage builder unavailable: {exc}")
        return None


try:
    from aiter.ops.triton.quant.sage_attention_quant_wrappers import sage_quant
except Exception as _exc:  # noqa: BLE001
    sage_quant = None
    logger.info(f"aiter sage_quant unavailable: {_exc}")

build_sage_attention_v2_core = _load_sage_builder()

_FP8_DTYPE = torch.float8_e4m3fn
_FP8_MAX = torch.finfo(_FP8_DTYPE).max


@ATTN_WEIGHT_REGISTER("aiter_sage_attn")
class AiterSageAttnWeight(AttnWeightTemplate):
    """gfx1201 SageAttention2 with aiter fused Triton quant (sage_quant).

    Quantization (int8 QK / fp8 V, K-smoothing, log2e Q folding) runs in one
    fused Triton kernel straight from BF16 -> no PyTorch float32 upcast. Only the
    small int8/fp8 tensors are padded to block_n=32 before the FlyDSL core.
    """

    def __init__(self):
        self.config = {}
        self._kernels = {}

    def _get_kernel(self, num_heads, seq_len):
        key = (num_heads, seq_len)
        kernel = self._kernels.get(key)
        if kernel is None:
            cfg = {
                "num_heads": num_heads,
                "head_dim": 128,
                "output_dtype": "bf16",
                "block_m": 128,
                "block_n": 32,
                "lds_padding": 8,
                "kv_prefetch_mode": "kv",
                "gated_o_rescale": True,
                "experimental_lds_reserve_bytes": 32768,
            }
            if seq_len % 32 != 0:
                cfg["experimental_key_tail_peel_seq_len"] = seq_len
            kernel = build_sage_attention_v2_core(**cfg)
            self._kernels[key] = kernel
        return kernel

    def apply(
        self,
        q,
        k,
        v,
        cu_seqlens_q=None,
        cu_seqlens_kv=None,
        max_seqlen_q=None,
        max_seqlen_kv=None,
        sm_scale=None,
        **kwargs,
    ):
        assert build_sage_attention_v2_core is not None, "aiter sage builder unavailable"
        assert sage_quant is not None, "aiter sage_quant unavailable"

        if q.ndim == 3:
            bs = 1
            q, k, v = q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)
        else:
            bs = q.shape[0]
        assert bs == 1, "aiter sage backend supports batch size 1 only"

        seq_len, heads, head_dim = q.shape[1], q.shape[2], q.shape[3]
        assert head_dim == 128, "aiter sage requires head_dim=128"
        assert q.dtype in (torch.float16, torch.bfloat16)
        if max_seqlen_q is None:
            max_seqlen_q = seq_len
        if sm_scale is None:
            sm_scale = float(head_dim**-0.5)

        padded = (seq_len + 31) // 32 * 32
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()

        # Fused Triton quant: BF16 -> int8 Q/K + fp8 V, per-32-block scales,
        # smooth-K and log2e*sm_scale folding all on-GPU. block_n=32 matches core.
        q_int8, q_scale, k_int8, k_scale, v_fp8, v_scale = sage_quant(
            q,
            k,
            v,
            FP8_TYPE=_FP8_DTYPE,
            FP8_MAX=_FP8_MAX,
            BLKQ=32,
            BLKK=32,
            sm_scale=float(sm_scale),
            layout="bshd",
            smooth_k=True,
        )

        pad = padded - seq_len
        if pad:
            q_int8 = torch.nn.functional.pad(q_int8, (0, 0, 0, 0, 0, pad))
            k_int8 = torch.nn.functional.pad(k_int8, (0, 0, 0, 0, 0, pad))
            v_fp8 = torch.nn.functional.pad(v_fp8, (0, 0, 0, 0, 0, pad))
        v_fp8 = v_fp8.permute(0, 2, 3, 1).contiguous()
        q_scale = q_scale.contiguous()
        k_scale = k_scale.contiguous()
        v_scale = v_scale.contiguous()

        kernel = self._get_kernel(heads, seq_len)
        out = torch.empty((1, padded, heads, 128), device=q_int8.device, dtype=torch.bfloat16)
        kernel(
            q_int8,
            k_int8,
            v_fp8,
            q_scale,
            k_scale,
            v_scale,
            out,
            1,
            padded,
            seq_len,
            stream=torch.cuda.current_stream(),
        )
        return out[:, :seq_len].reshape(bs * max_seqlen_q, -1)
