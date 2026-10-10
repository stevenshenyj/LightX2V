"""Separate Q/K/V fused Sage prepare prototypes for MiniMax-H3 gaosage.

Env gated by H3_SEPARATE_FUSED_SAGE_PREPARE=1.  This is prepared for
profiling experiments only; default model path is unchanged.

This variant emits prepacked int8-Q + int8-K + fp8-V for the fast
SageAttention int8-Q core.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass

import torch
import triton
import triton.language as tl

EPS = 1.0e-6
FP8MAX = 2.25
K_GROUP = 64
Q_GROUP = 32
V_GROUP = 64


@dataclass
class H3SeparateFusedSagePackedQKV:
    q_int8: torch.Tensor       # int8 Q [B,S,H,D]
    q_scale: torch.Tensor      # [B,H,ceil(S/32)]
    k_int8: torch.Tensor       # [B,H,padded_S,D]
    k_scale: torch.Tensor      # [B,H,padded_S/64]
    value_native: torch.Tensor # [B,H,D,padded_S] tiled-HDS layout for int8-Q core
    value_scale: torch.Tensor  # [B,H,D]
    seq_len: int
    num_heads: int
    head_dim: int
    dtype: torch.dtype

    @property
    def device(self):
        return self.q_int8.device

    @property
    def shape(self):
        return (self.seq_len, self.num_heads, self.head_dim)

    def contiguous(self):
        return self

    @property
    def is_h3_fused_sage_prepare(self):
        return True

    @property
    def is_h3_separate_fused_sage_prepare(self):
        return True


def _next_pow2(x: int) -> int:
    return 1 << (int(x) - 1).bit_length()


@triton.jit
def _q_norm_rope_int8(QIn, QW, COS, SIN, QI8, QS,
                      S:tl.constexpr, H:tl.constexpr, D:tl.constexpr, RD:tl.constexpr,
                      BLOCK_S:tl.constexpr, BLOCK_D:tl.constexpr, EPS_C:tl.constexpr):
    bid = tl.program_id(0)
    h = tl.program_id(1)
    rows = bid * BLOCK_S + tl.arange(0, BLOCK_S)
    d = tl.arange(0, BLOCK_D)
    pair = tl.where(d < RD, tl.where(d < RD // 2, d + RD // 2, d - RD // 2), d)
    sign = tl.where(d < RD // 2, -1.0, 1.0)
    mask = (rows[:, None] < S) & (d[None, :] < D)

    x = tl.load(QIn + (rows[:, None] * H + h) * D + d[None, :], mask=mask, other=0.0).to(tl.float32)
    rms = tl.sqrt(tl.sum(x * x, axis=1) / D + EPS_C)
    y = (x / rms[:, None] * tl.load(QW + d, mask=d < D, other=0.0)[None, :]).to(tl.bfloat16).to(tl.float32)

    xp = tl.load(QIn + (rows[:, None] * H + h) * D + pair[None, :], mask=(rows[:, None] < S) & (pair[None, :] < D), other=0.0).to(tl.float32)
    yp = (xp / rms[:, None] * tl.load(QW + pair, mask=pair < D, other=0.0)[None, :]).to(tl.bfloat16).to(tl.float32)
    c = tl.load(COS + rows[:, None] * RD + d[None, :], mask=(rows[:, None] < S) & (d[None, :] < RD), other=1.0)
    ss = tl.load(SIN + rows[:, None] * RD + d[None, :], mask=(rows[:, None] < S) & (d[None, :] < RD), other=0.0)
    rope = tl.where(d[None, :] < RD, y * c + yp * ss * sign[None, :], y)
    rope = tl.where(rows[:, None] < S, rope, 0.0)
    qmax = tl.maximum(tl.max(tl.max(tl.abs(rope), axis=1), axis=0), 1.0e-7)
    qi = tl.extra.libdevice.nearbyint(rope * (127.0 / qmax)).to(tl.int8)
    tl.store(QI8 + (rows[:, None] * H + h) * D + d[None, :], qi, mask=mask)
    tl.store(QS + h * tl.cdiv(S, BLOCK_S) + bid, qmax / 127.0, mask=bid * BLOCK_S < S)


@triton.jit
def _kmean_partial(KIn, KW, COS, SIN, PSUM,
                   S:tl.constexpr, H:tl.constexpr, D:tl.constexpr, RD:tl.constexpr,
                   BLOCK_S:tl.constexpr, BLOCK_D:tl.constexpr, EPS_C:tl.constexpr):
    bid = tl.program_id(0)
    h = tl.program_id(1)
    rows = bid * BLOCK_S + tl.arange(0, BLOCK_S)
    d = tl.arange(0, BLOCK_D)
    pair = tl.where(d < RD, tl.where(d < RD // 2, d + RD // 2, d - RD // 2), d)
    sign = tl.where(d < RD // 2, -1.0, 1.0)
    mask = (rows[:, None] < S) & (d[None, :] < D)

    x = tl.load(KIn + (rows[:, None] * H + h) * D + d[None, :], mask=mask, other=0.0).to(tl.float32)
    rms = tl.sqrt(tl.sum(x * x, axis=1) / D + EPS_C)
    y = (x / rms[:, None] * tl.load(KW + d, mask=d < D, other=0.0)[None, :]).to(tl.bfloat16).to(tl.float32)

    xp = tl.load(KIn + (rows[:, None] * H + h) * D + pair[None, :], mask=(rows[:, None] < S) & (pair[None, :] < D), other=0.0).to(tl.float32)
    yp = (xp / rms[:, None] * tl.load(KW + pair, mask=pair < D, other=0.0)[None, :]).to(tl.bfloat16).to(tl.float32)
    c = tl.load(COS + rows[:, None] * RD + d[None, :], mask=(rows[:, None] < S) & (d[None, :] < RD), other=1.0)
    s = tl.load(SIN + rows[:, None] * RD + d[None, :], mask=(rows[:, None] < S) & (d[None, :] < RD), other=0.0)
    rope = tl.where(d[None, :] < RD, y * c + yp * s * sign[None, :], y)
    ps = tl.sum(tl.where(rows[:, None] < S, rope, 0.0), axis=0)
    tl.store(PSUM + (h * tl.cdiv(S, BLOCK_S) + bid) * D + d, ps, mask=d < D)


@triton.jit
def _kmean_finalize(PSUM, KMEAN,
                    NB:tl.constexpr, D:tl.constexpr, S:tl.constexpr,
                    BLOCK_NB:tl.constexpr):
    h = tl.program_id(0)
    d = tl.program_id(1)
    offs = tl.arange(0, BLOCK_NB)
    ps = tl.load(PSUM + (h * NB + offs) * D + d, mask=offs < NB, other=0.0)
    tl.store(KMEAN + h * D + d, tl.sum(ps, axis=0) / S)


@triton.jit
def _k_norm_rope_quant_hnd(KIn, KW, COS, SIN, KMEAN, KI8, KS,
                           S:tl.constexpr, PAD_S:tl.constexpr, H:tl.constexpr, D:tl.constexpr, RD:tl.constexpr,
                           BLOCK_D:tl.constexpr, EPS_C:tl.constexpr, LaneMajorKV:tl.constexpr = False):
    g = tl.program_id(0)
    h = tl.program_id(1)
    rows = g * K_GROUP + tl.arange(0, K_GROUP)
    d = tl.arange(0, BLOCK_D)
    pair = tl.where(d < RD, tl.where(d < RD // 2, d + RD // 2, d - RD // 2), d)
    sign = tl.where(d < RD // 2, -1.0, 1.0)

    x = tl.load(KIn + (rows[:, None] * H + h) * D + d[None, :], mask=(rows[:, None] < S) & (d[None, :] < D), other=0.0).to(tl.float32)
    rms = tl.sqrt(tl.sum(x * x, axis=1) / D + EPS_C)
    y = (x / rms[:, None] * tl.load(KW + d, mask=d < D, other=0.0)[None, :]).to(tl.bfloat16).to(tl.float32)

    xp = tl.load(KIn + (rows[:, None] * H + h) * D + pair[None, :], mask=(rows[:, None] < S) & (pair[None, :] < D), other=0.0).to(tl.float32)
    yp = (xp / rms[:, None] * tl.load(KW + pair, mask=pair < D, other=0.0)[None, :]).to(tl.bfloat16).to(tl.float32)
    c = tl.load(COS + rows[:, None] * RD + d[None, :], mask=(rows[:, None] < S) & (d[None, :] < RD), other=1.0)
    s = tl.load(SIN + rows[:, None] * RD + d[None, :], mask=(rows[:, None] < S) & (d[None, :] < RD), other=0.0)
    rope = tl.where(d[None, :] < RD, y * c + yp * s * sign[None, :], y)
    mean = tl.load(KMEAN + h * D + d, mask=d < D, other=0.0)
    centered = tl.where(rows[:, None] < S, rope - mean[None, :], 0.0)
    mx = tl.maximum(tl.max(tl.max(tl.abs(centered), axis=1), axis=0), 1.0e-7)
    qi = tl.extra.libdevice.nearbyint(centered * (127.0 / mx)).to(tl.int8)
    if LaneMajorKV:
        koff = h * PAD_S * D + (rows[:, None] // 32) * 4096 + (((rows[:, None] % 32 // 16) * 8 + d[None, :] // 16) * 32 + rows[:, None] % 16 + 16 * (d[None, :] % 16 // 8)) * 8 + d[None, :] % 8
    else:
        koff = (h * PAD_S + rows[:, None]) * D + d[None, :]
    tl.store(KI8 + koff, qi, mask=(rows[:, None] < PAD_S) & (d[None, :] < D))
    tl.store(KS + h * tl.cdiv(PAD_S, K_GROUP) + g, mx / 127.0, mask=g * K_GROUP < PAD_S)


@triton.jit
def _k_norm_rope_quant_hnd_no_mean(KIn, KW, COS, SIN, KI8, KS,
                                   S:tl.constexpr, PAD_S:tl.constexpr, H:tl.constexpr, D:tl.constexpr, RD:tl.constexpr,
                                   BLOCK_D:tl.constexpr, EPS_C:tl.constexpr, LaneMajorKV:tl.constexpr = False):
    g = tl.program_id(0)
    h = tl.program_id(1)
    rows = g * 64 + tl.arange(0, 64)
    d = tl.arange(0, BLOCK_D)
    pair = tl.where(d < RD, tl.where(d < RD // 2, d + RD // 2, d - RD // 2), d)
    sign = tl.where(d < RD // 2, -1.0, 1.0)

    x = tl.load(KIn + (rows[:, None] * H + h) * D + d[None, :], mask=(rows[:, None] < S) & (d[None, :] < D), other=0.0).to(tl.float32)
    rms = tl.sqrt(tl.sum(x * x, axis=1) / D + EPS_C)
    y = (x / rms[:, None] * tl.load(KW + d, mask=d < D, other=0.0)[None, :]).to(tl.bfloat16).to(tl.float32)

    xp = tl.load(KIn + (rows[:, None] * H + h) * D + pair[None, :], mask=(rows[:, None] < S) & (pair[None, :] < D), other=0.0).to(tl.float32)
    yp = (xp / rms[:, None] * tl.load(KW + pair, mask=pair < D, other=0.0)[None, :]).to(tl.bfloat16).to(tl.float32)
    c = tl.load(COS + rows[:, None] * RD + d[None, :], mask=(rows[:, None] < S) & (d[None, :] < RD), other=1.0)
    ss = tl.load(SIN + rows[:, None] * RD + d[None, :], mask=(rows[:, None] < S) & (d[None, :] < RD), other=0.0)
    rope = tl.where(d[None, :] < RD, y * c + yp * ss * sign[None, :], y)
    centered = tl.where(rows[:, None] < S, rope, 0.0)
    mx = tl.maximum(tl.max(tl.max(tl.abs(centered), axis=1), axis=0), 1.0e-7)
    qi = tl.extra.libdevice.nearbyint(centered * (127.0 / mx)).to(tl.int8)
    if LaneMajorKV:
        koff = h * PAD_S * D + (rows[:, None] // 32) * 4096 + (((rows[:, None] % 32 // 16) * 8 + d[None, :] // 16) * 32 + rows[:, None] % 16 + 16 * (d[None, :] % 16 // 8)) * 8 + d[None, :] % 8
    else:
        koff = (h * PAD_S + rows[:, None]) * D + d[None, :]
    tl.store(KI8 + koff, qi, mask=(rows[:, None] < PAD_S) & (d[None, :] < D))
    tl.store(KS + h * tl.cdiv(PAD_S, 64) + g, mx / 127.0, mask=g * 64 < PAD_S)


@triton.jit
def _vabsmax_partial(VIn, PMAX,
                     S:tl.constexpr, H:tl.constexpr, D:tl.constexpr,
                     BLOCK_S:tl.constexpr, BLOCK_D:tl.constexpr):
    bid = tl.program_id(0)
    h = tl.program_id(1)
    rows = bid * BLOCK_S + tl.arange(0, BLOCK_S)
    d = tl.arange(0, BLOCK_D)
    vx = tl.load(VIn + (rows[:, None] * H + h) * D + d[None, :], mask=(rows[:, None] < S) & (d[None, :] < D), other=0.0).to(tl.float32)
    pm = tl.max(tl.where(rows[:, None] < S, tl.abs(vx), 0.0), axis=0)
    tl.store(PMAX + (h * tl.cdiv(S, BLOCK_S) + bid) * D + d, pm, mask=d < D)


@triton.jit
def _vscale_finalize(PMAX, VSCALE,
                     NB:tl.constexpr, D:tl.constexpr,
                     BLOCK_NB:tl.constexpr, FP8MAX_C:tl.constexpr):
    h = tl.program_id(0)
    d = tl.program_id(1)
    offs = tl.arange(0, BLOCK_NB)
    pm = tl.load(PMAX + (h * NB + offs) * D + d, mask=offs < NB, other=0.0)
    vmax = tl.max(pm, axis=0)
    tl.store(VSCALE + h * D + d, tl.maximum(vmax / FP8MAX_C, 1.0e-12))


@triton.jit
def _v_quant_hds(VIn, VSCALE, VFP8,
                 S:tl.constexpr, PAD_S:tl.constexpr, H:tl.constexpr, D:tl.constexpr,
                 BLOCK_S:tl.constexpr, BLOCK_D:tl.constexpr, LaneMajorKV:tl.constexpr = False):
    g = tl.program_id(0)
    h = tl.program_id(1)
    rows = g * BLOCK_S + tl.arange(0, BLOCK_S)
    d = tl.arange(0, BLOCK_D)
    vx = tl.load(VIn + (rows[:, None] * H + h) * D + d[None, :], mask=(rows[:, None] < S) & (d[None, :] < D), other=0.0).to(tl.float32)
    sc = tl.load(VSCALE + h * D + d, mask=d < D, other=1.0)
    vq = (vx / sc[None, :]).to(tl.float8e4nv).to(tl.uint8, bitcast=True)
    # int8-Q H3 core expects [B,H,D,PAD_S]. For permute/lane-major core,
    # use the same store layout as SageAttention per_channel_fp8_nhd(...,
    # tiled=True, head_first=True, lane_major_kv=True).
    if LaneMajorKV:
        voff = h * D * PAD_S + (rows[:, None] // 32) * 4096 + (((rows[:, None] % 32 // 16) * 8 + d[None, :] // 16) * 32 + d[None, :] % 16 + 16 * (rows[:, None] % 16 // 8)) * 8 + rows[:, None] % 8
    else:
        tile = 32
        voff = h * D * PAD_S + (rows[:, None] // tile) * (D * tile) + d[None, :] * tile + (rows[:, None] % tile)
    tl.store(VFP8 + voff, vq, mask=(rows[:, None] < PAD_S) & (d[None, :] < D))


def _view_3d(x: torch.Tensor, num_heads: int, head_dim: int) -> torch.Tensor:
    if x.dim() == 3:
        return x.contiguous()
    return x.view(x.shape[0], num_heads, head_dim).contiguous()


def separate_fused_sage_prepare(q_2d, k_2d, v_2d, norm_q, norm_k, rotary_emb, *, num_heads: int, head_dim: int,
                                k_block_s: int = 128, v_block_s: int = 256):
    """Return prepacked int8-Q/int8-K/fp8-V tensors for the fast int8-Q native attention."""
    q = _view_3d(q_2d, num_heads, head_dim)
    k = _view_3d(k_2d, num_heads, head_dim)
    v = _view_3d(v_2d, num_heads, head_dim)
    S = int(q.shape[0]); H = int(num_heads); D = int(head_dim)
    cos, sin = rotary_emb
    cos = cos.contiguous(); sin = sin.contiguous()
    RD = int(cos.shape[-1])
    pad = ((S + 255) // 256) * 256
    qw = norm_q.weight.contiguous()
    kw = norm_k.weight.contiguous()
    # h3_int8q_permute core requires LaneMajorKv/SwapHalfPermute K/V layout.
    # Keep explicit env override, and auto-enable it when the prepacked Sage impl selects permute.
    lane_major_kv = (
        os.environ.get("H3_SEPARATE_FUSED_LANE_MAJOR_KV", "0") == "1"
        or os.environ.get("H3_SEPARATE_FUSED_SAGE_IMPL", "h3_int8q") == "h3_int8q_permute"
    )

    q_i8 = torch.empty((1, S, H, D), device=q.device, dtype=torch.int8)
    q_scale = torch.empty((1, H, math.ceil(S / Q_GROUP)), device=q.device, dtype=torch.float32)
    _q_norm_rope_int8[(math.ceil(S / Q_GROUP), H)](q, qw, cos, sin, q_i8, q_scale, S, H, D, RD, BLOCK_S=Q_GROUP, BLOCK_D=128, EPS_C=EPS, num_warps=4)

    k_i8 = torch.empty((1, H, pad, D), device=q.device, dtype=torch.int8)
    k_scale = torch.empty((1, H, math.ceil(pad / K_GROUP)), device=q.device, dtype=torch.float32)
    # The int8-Q H3 core path uses K without mean-smoothing: K fused prepare
    # emits only k_int8 + k_scale, avoiding the extra K-mean passes.
    if os.environ.get("H3_SEPARATE_FUSED_K_MEAN", "0") == "1":
        knb = math.ceil(S / k_block_s)
        psum = torch.empty((H, knb, D), device=q.device, dtype=torch.float32)
        kmean = torch.empty((H, D), device=q.device, dtype=torch.float32)
        _kmean_partial[(knb, H)](k, kw, cos, sin, psum, S, H, D, RD, BLOCK_S=k_block_s, BLOCK_D=128, EPS_C=EPS, num_warps=4)
        _kmean_finalize[(H, D)](psum, kmean, NB=knb, D=D, S=S, BLOCK_NB=_next_pow2(knb), num_warps=8)
        _k_norm_rope_quant_hnd[(math.ceil(pad / K_GROUP), H)](k, kw, cos, sin, kmean, k_i8, k_scale, S, pad, H, D, RD, BLOCK_D=128, EPS_C=EPS, LaneMajorKV=lane_major_kv, num_warps=4)
    else:
        _k_norm_rope_quant_hnd_no_mean[(math.ceil(pad / K_GROUP), H)](k, kw, cos, sin, k_i8, k_scale, S, pad, H, D, RD, BLOCK_D=128, EPS_C=EPS, LaneMajorKV=lane_major_kv, num_warps=4)

    vnb = math.ceil(S / v_block_s)
    pmax = torch.empty((H, vnb, D), device=q.device, dtype=torch.float32)
    vscale = torch.empty((H, D), device=q.device, dtype=torch.float32)
    _vabsmax_partial[(vnb, H)](v, pmax, S, H, D, BLOCK_S=v_block_s, BLOCK_D=128, num_warps=8)
    _vscale_finalize[(H, D)](pmax, vscale, NB=vnb, D=D, BLOCK_NB=_next_pow2(vnb), FP8MAX_C=FP8MAX, num_warps=4)
    v_fp8 = torch.empty((1, H, D, pad), device=q.device, dtype=torch.uint8)
    _v_quant_hds[(math.ceil(pad / V_GROUP), H)](v, vscale, v_fp8, S, pad, H, D, BLOCK_S=V_GROUP, BLOCK_D=128, LaneMajorKV=lane_major_kv, num_warps=8)

    return H3SeparateFusedSagePackedQKV(q_i8, q_scale, k_i8, k_scale, v_fp8, vscale.unsqueeze(0).contiguous(), S, H, D, q_2d.dtype)
