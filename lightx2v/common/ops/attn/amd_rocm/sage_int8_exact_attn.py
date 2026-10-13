# gfx1201 SageAttention core: INT8 QK + FP8 PV WMMA with exact exp2 P.
# Kernel source in ./sage_int8_exact is copied unmodified from
# /app/sageattn_best_dualprob_no_precision_reg_20261013/src and built with that package's flags
# (common.build_port): output is bitwise equal to the dualprob core built with APPROX_FP8_P=0.
import hashlib
import os
from pathlib import Path

import torch
from loguru import logger

from lightx2v.utils.registry_factory import ATTN_WEIGHT_REGISTER

from ..template import AttnWeightTemplate

_SOURCE_DIR = Path(__file__).resolve().parent / "sage_int8_exact"
_SOURCE = _SOURCE_DIR / "attn_int8_port.cu"
_SOURCE_SHA256 = {
    "attn_int8_port.cu": "b171039eb1940b857aae84fe1c8494f9de20b47d614fcd7ca52374392d793463",
    "softmax_asm.inc": "909f19d5c929a5f499dc421e66c2e8ead74c6313b0599ee9a4a7c0190dd564a9",
    "rowmax_asm.inc": "b9ec8f7dee4d4ce7f57e62015bd7d0e8e8fef4f829efc98962b11121ee223ee1",
}
_PORT_FLAGS = ["-O3", "-fgpu-flush-denormals-to-zero", "-Xclang", "-target-feature", "-Xclang", "-real-true16"]
_MAX_ILP = ["-mllvm", "-amdgpu-sched-strategy=max-ilp"]

_module = None


def load_sage_int8_exact():
    global _module
    if _module is not None:
        return _module
    for name, digest in _SOURCE_SHA256.items():
        actual = hashlib.sha256((_SOURCE_DIR / name).read_bytes()).hexdigest()
        if actual != digest:
            raise RuntimeError(f"sage_int8_exact source {name} changed: sha256 {actual} != {digest}")
    from torch.utils.cpp_extension import load_inline

    sources = _SOURCE.read_bytes() + b"".join(p.read_bytes() for p in sorted(_SOURCE_DIR.glob("*.inc")))
    digest = hashlib.sha256(sources + " ".join(_MAX_ILP).encode()).hexdigest()[:16]
    cache = os.environ.get("LIGHTX2V_SAGE_INT8_EXACT_BUILD_DIR", str(Path.home() / ".cache/lightx2v/sage_int8_exact"))
    build_dir = Path(cache) / digest
    build_dir.mkdir(parents=True, exist_ok=True)
    saved_arch = os.environ.get("PYTORCH_ROCM_ARCH")
    # The kernel uses gfx12 WMMA builtins; a multi-arch list from the shell would break the build.
    os.environ["PYTORCH_ROCM_ARCH"] = "gfx1201"
    try:
        _module = load_inline(
            name=f"sage_int8_exact_{digest}",
            cpp_sources="",
            cuda_sources=f'#include "{_SOURCE}"\n',
            extra_cuda_cflags=[f"-DSAGE_PORT_SOURCE_HASH=0x{digest}", *_PORT_FLAGS, *_MAX_ILP],
            build_directory=str(build_dir),
            verbose=False,
        )
    finally:
        if saved_arch is None:
            os.environ.pop("PYTORCH_ROCM_ARCH", None)
        else:
            os.environ["PYTORCH_ROCM_ARCH"] = saved_arch
    logger.info(f"sage_attn2_int8_exact: loaded {_module.__file__}")
    return _module


@ATTN_WEIGHT_REGISTER("sage_attn2_int8_exact")
class SageAttn2Int8ExactWeight(AttnWeightTemplate):
    """MiniMax-H3 gfx1201 SageAttention, exact-P int8 core (no_precision_reg 20261013).

    Consumes the MiniMax-H3 fused QK-norm/RoPE/int8 prepare with lane-major K/V
    (fused_sage_prepare_separate); BF16 Q/K/V inputs are rejected.
    """

    requires_h3_lane_major_prepack = True

    def __init__(self):
        self.config = {}
        load_sage_int8_exact()

    def apply(self, q, k=None, v=None, cu_seqlens_q=None, cu_seqlens_kv=None, max_seqlen_q=None, max_seqlen_kv=None, **kwargs):
        if not getattr(q, "is_h3_fused_sage_prepare", False) or not getattr(q, "lane_major_kv", False):
            raise NotImplementedError(
                "sage_attn2_int8_exact needs the MiniMax-H3 fused int8 prepare with lane-major K/V; "
                "got BF16 Q/K/V (the fused prepare failed or is unavailable for this model)"
            )
        out = torch.empty((1, q.seq_len, q.num_heads, q.head_dim), device=q.device, dtype=torch.bfloat16)
        load_sage_int8_exact().attention(
            q.q_int8, q.k_int8, q.value_native, out, q.q_scale, q.k_scale, q.value_scale, q.head_dim**-0.5, q.seq_len,
        )
        return out.view(q.seq_len, -1)
