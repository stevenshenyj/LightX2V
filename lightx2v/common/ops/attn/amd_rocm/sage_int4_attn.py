# SPDX-License-Identifier: MIT
"""LightX2V attention backend "sage_attn2_int4": MiniMax-H3 int4/int8 mixed SageAttention on gfx12.

Kernels come from SageAttention (gfx1201 mixed int4/int8 branch, built with setup.py):
  - int4 calls: sageattention.gfx12_int4 (INT4 QK + FP8 PV) with the fused QK-norm/RoPE int4 prepare
  - int8 calls with the H3 fused int8 prepare in lane-major K/V (blocks 45/49):
    sageattention.gfx12_exact (exact-P INT8 QK + FP8 PV) on the prepacked tensors
  - everything else: LightX2V's own "sage_attn2" (sageattention.sageattn)

Settings (later wins): DEFAULT_SETTING < model config "sage_attn2_int4_setting" (via set_config) < env
SAGE_INT4_INT8_BLOCKS="45,49" / SAGE_INT4_MIN_SEQ_LEN=32768 / SAGE_INT4_FUSED_PREPARE=1 / SAGE_INT4_INT8_KERNEL=exact.

Fused prepare: the model calls prepare_norm_rope() on the raw projections of an int4 call (see routes_int4()) and
passes the returned PreparedQKV as q (k = v = None) to apply(). prefers_h3_lane_major_prepack asks the model for its
fused int8 prepare with lane-major K/V on the calls that do not run int4.
"""
import atexit
import os

import torch
from loguru import logger

from lightx2v.utils.registry_factory import ATTN_WEIGHT_REGISTER

from ..sage_attn import SageAttn2Weight

try:
    from sageattention.gfx12_exact import sageattn_qk_int8_pv_fp8_gfx1201_exact_prepacked
    from sageattention.gfx12_int4 import attention_int4_from_quantized, sageattn_qk_int4_pv_fp8_gfx12
    from sageattention.triton.quant_gfx12_int4 import prepare_int4_norm_rope
except ImportError as _e:
    _IMPORT_ERROR = _e
else:
    _IMPORT_ERROR = None

BACKEND_NAME = "sage_attn2_int4"
INT4_ARCHS = ("gfx1200", "gfx1201")
INT8_EXACT_ARCHS = ("gfx1201",)
INT8_KERNELS = ("exact", "sage_attn2")
DEFAULT_SETTING = {
    # MiniMax-H3 transformer blocks whose QK is not accurate enough in int4.
    "int8_blocks": [45, 49],
    # Shorter calls (e.g. the H3 text refiner, S=3582) do not run int4.
    "min_seq_len": 32768,
    # Offer prepare_norm_rope() (QK norm + RoPE fused into the int4 prepare); False -> it returns None.
    "fused_prepare": True,
    # Core for prepacked lane-major int8 calls: "exact" = SageAttention gfx1201 exact-P, "sage_attn2" = sage_attn2.
    "int8_kernel": "exact",
}


def _env_setting():
    out = {}
    if os.environ.get("SAGE_INT4_INT8_BLOCKS") is not None:
        out["int8_blocks"] = [int(b) for b in os.environ["SAGE_INT4_INT8_BLOCKS"].split(",") if b.strip()]
    if os.environ.get("SAGE_INT4_MIN_SEQ_LEN"):
        out["min_seq_len"] = int(os.environ["SAGE_INT4_MIN_SEQ_LEN"])
    if os.environ.get("SAGE_INT4_FUSED_PREPARE"):
        out["fused_prepare"] = os.environ["SAGE_INT4_FUSED_PREPARE"] == "1"
    if os.environ.get("SAGE_INT4_INT8_KERNEL"):
        out["int8_kernel"] = os.environ["SAGE_INT4_INT8_KERNEL"]
    return out


def _device_arch(device=None):
    if torch.version.hip is None or not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_properties(device or torch.cuda.current_device()).gcnArchName.split(":")[0]


# Shared by all per-block instances in this process: (route, q shape, block) calls seen, and unlogged ones.
_ROUTE_SEEN = set()
_ROUTE_PENDING = {}


def _block_ranges(blocks):
    nums = sorted(b for b in blocks if b is not None)
    spans, start = [], None
    for i, b in enumerate(nums):
        start = b if start is None else start
        if i + 1 == len(nums) or nums[i + 1] != b + 1:
            spans.append(str(b) if start == b else f"{start}-{b}")
            start = None
    return ",".join(spans + (["None"] if None in blocks else []))


def _flush_routes():
    if not _ROUTE_PENDING:
        return
    parts = [f"{route} q={shape} x{len(blocks)} blocks [{_block_ranges(blocks)}]" for (route, shape), blocks in _ROUTE_PENDING.items()]
    _ROUTE_PENDING.clear()
    logger.info(f"{BACKEND_NAME} routes (rank {os.environ.get('RANK', '0')}): " + "; ".join(parts))


def _note_route(route, shape, block_idx):
    """Collects first-seen routes; logs them as one line when a call repeats (next step) or at exit."""
    key = (route, tuple(shape), block_idx)
    if key in _ROUTE_SEEN:
        _flush_routes()
        return
    _ROUTE_SEEN.add(key)
    _ROUTE_PENDING.setdefault(key[:2], []).append(block_idx)


atexit.register(_flush_routes)


class PreparedQKV:
    """prepare_int4_norm_rope outputs of one attention call; passed to apply() as q."""

    is_sage_int4_prepared = True

    def __init__(self, tensors, seq_len, num_heads):
        self.tensors, self.seq_len, self.num_heads = tensors, seq_len, num_heads

    @property
    def shape(self):
        return (self.seq_len, self.num_heads, 128)

    @property
    def device(self):
        return self.tensors[0].device

    def contiguous(self):
        return self


@ATTN_WEIGHT_REGISTER(BACKEND_NAME)
class SageAttn2Int4Weight(SageAttn2Weight):
    """int4 for eligible calls of blocks not in `int8_blocks`; SageAttention exact-P int8 for lane-major prepacked int8
    calls; the inherited sage_attn2 path otherwise."""

    def __init__(self):
        if _IMPORT_ERROR is not None:
            raise ImportError(f"{BACKEND_NAME} needs sageattention built from the gfx1201 mixed int4/int8 branch "
                              f"(sageattention.gfx12_int4 / gfx12_exact): {_IMPORT_ERROR}") from _IMPORT_ERROR
        super().__init__()
        self._arch = None
        self._logged = set()
        self.set_config({})

    def set_config(self, config=None):
        merged = dict(DEFAULT_SETTING)
        merged.update(config or {})
        merged.update(_env_setting())
        merged["int8_blocks"] = frozenset(int(b) for b in merged["int8_blocks"])
        merged["min_seq_len"] = int(merged["min_seq_len"])
        merged["fused_prepare"] = bool(merged["fused_prepare"])
        if merged["int8_kernel"] not in INT8_KERNELS:
            raise ValueError(f"{BACKEND_NAME}: int8_kernel must be one of {INT8_KERNELS}, got {merged['int8_kernel']!r}")
        self.config = merged

    def _log_once(self, key, msg, warn=False):
        if key not in self._logged:
            self._logged.add(key)
            (logger.warning if warn else logger.info)(msg)

    def _get_arch(self, device=None):
        if self._arch is None:
            self._arch = _device_arch(device) or ""
        return self._arch

    def _int4_supported(self, device=None):
        return self._get_arch(device) in INT4_ARCHS

    @property
    def prefers_h3_lane_major_prepack(self):
        """Whether calls that do not run int4 want the model's fused int8 prepare with lane-major K/V (int8 core)."""
        return self.config["int8_kernel"] == "exact" and self._get_arch() in INT8_EXACT_ARCHS

    def routes_int4(self, block_idx, seq_len, device=None):
        """Whether a non-causal head_dim-128 self-attention call of `block_idx` with `seq_len` tokens runs int4.

        Lets the model skip its own int8 Q/K/V prepare for blocks that will run int4 (they need bf16 Q/K/V or
        prepare_norm_rope()).
        """
        if block_idx is None or seq_len < self.config["min_seq_len"] or int(block_idx) in self.config["int8_blocks"]:
            return False
        return self._int4_supported(device)

    def prepare_norm_rope(self, q, k, v, q_weight, k_weight, cos, sin, q_eps, k_eps):
        """Fused QK RMSNorm + split-half RoPE + int4/fp8 prepare (prepare_int4_norm_rope) for an int4 call.

        q/k/v: bf16 [S, H*128] projections. Returns a PreparedQKV to pass as q to apply(), or None when
        `fused_prepare` is off or the inputs are not bf16 2-D (then normalize/rotate and pass bf16 Q/K/V as before).
        """
        if not self.config["fused_prepare"] or q.dtype != torch.bfloat16 or q.ndim != 2 or q.shape[1] % 128:
            return None
        tensors = prepare_int4_norm_rope(q, k, v, q_weight, k_weight, cos, sin, q_eps, k_eps)
        return PreparedQKV(tensors, q.shape[0], q.shape[1] // 128)

    def _route_int8_exact(self, q):
        """Prepacked H3 int8 input that the SageAttention exact-P core can take."""
        if self.config["int8_kernel"] != "exact" or getattr(q, "head_dim", None) != 128:
            return False
        if not getattr(q, "lane_major_kv", False):
            # Row-major K/V (H3_SEPARATE_FUSED_SAGE_IMPL=h3_int8q) is only understood by sage_attn2.
            self._log_once("rowmajor", f"{BACKEND_NAME}: prepacked int8 input without lane-major K/V, using sage_attn2", True)
            return False
        # Other archs: prefers_h3_lane_major_prepack was False, so the layout came from the env that also picks the
        # sage_attn2 prepacked core.
        return self._get_arch(q.device) in INT8_EXACT_ARCHS

    def _route_int4(self, q, k, v, kwargs):
        # Q/K/V already int8-prepacked by the model (LightX2V H3 fused sage prepare): int8 core or sage_attn2.
        if getattr(q, "is_h3_fused_sage_prepare", False):
            return False
        if kwargs.get("causal", False) or q.shape != k.shape or q.shape != v.shape:
            return False
        if q.shape[-1] != 128 or q.shape[-3] < self.config["min_seq_len"] or not q.dtype.is_floating_point:
            return False
        if not self._int4_supported(q.device):
            self._log_once("arch", f"{BACKEND_NAME}: device is not gfx1200/gfx1201, using sage_attn2 for all calls", True)
            return False
        block_idx = kwargs.get("block_idx")
        if block_idx is None:
            # Without the block index the int8-only blocks cannot be honored.
            self._log_once("noidx", f"{BACKEND_NAME}: model passed no block_idx, using sage_attn2 for q={tuple(q.shape)}", True)
            return False
        return int(block_idx) not in self.config["int8_blocks"]

    def apply(
        self,
        q,
        k,
        v,
        cu_seqlens_q=None,
        cu_seqlens_kv=None,
        max_seqlen_q=None,
        max_seqlen_kv=None,
        **kwargs,
    ):
        block_idx = kwargs.get("block_idx")
        if getattr(q, "is_sage_int4_prepared", False):
            _note_route("int4 fused-prepare", q.shape, block_idx)
            return attention_int4_from_quantized(*q.tensors, q.seq_len).reshape(q.seq_len, -1)
        if getattr(q, "is_h3_fused_sage_prepare", False) and self._route_int8_exact(q):
            _note_route("int8 exact prepacked", q.shape, block_idx)
            out = sageattn_qk_int8_pv_fp8_gfx1201_exact_prepacked(
                q.q_int8, q.q_scale, q.k_int8, q.k_scale, q.value_native, q.value_scale, q.seq_len,
                sm_scale=q.head_dim**-0.5,
            )
            return out.reshape(q.seq_len, -1)
        if not self._route_int4(q, k, v, kwargs):
            _note_route("sage_attn2 int8", q.shape, block_idx)
            return super().apply(q, k, v, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv, **kwargs)
        _note_route("int4", q.shape, block_idx)
        if q.ndim == 3:
            q, k, v = q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)
        q, k, v = (t.contiguous().bfloat16() for t in (q, k, v))
        out = sageattn_qk_int4_pv_fp8_gfx12(q, k, v, tensor_layout="NHD")
        return out.reshape(q.shape[0] * q.shape[1], -1)
