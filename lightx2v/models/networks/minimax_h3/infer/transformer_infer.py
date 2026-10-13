from functools import cache
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F
from loguru import logger

from lightx2v.common.transformer_infer.transformer_infer import BaseTransformerInfer
from lightx2v.models.networks.minimax_h3.adaln_cache import load_persistent_adaln_cache
from lightx2v.models.networks.minimax_h3.infer import fused_qkv  # noqa: F401 - registers the Triton backend
from lightx2v.models.networks.minimax_h3.infer import coresw_ffn_experiment
from lightx2v.models.networks.minimax_h3.infer import fused_sage_prepare_separate
from lightx2v.utils.envs import GET_DTYPE
from lightx2v.utils.registry_factory import QKV_NORM_ROPE_REGISTER
from lightx2v_platform.base.global_var import AI_DEVICE


@cache
def _warn_fused_qkv_fallback_once():
    logger.warning("Fused QKV projection is unavailable (for example, because LoRA/diff weights are active); falling back to separate Q/K/V projections")


@cache
def _warn_qkv_norm_rope_fallback_once(backend):
    logger.warning(
        "Fused QKV Norm/RoPE backend '{}' is unavailable or incompatible with the current inputs; falling back to separate Norm and RoPE operations",
        backend,
    )


class MiniMaxH3TransformerInfer(BaseTransformerInfer):
    def __init__(self, config):
        self.config = config
        self.hidden_size = int(config.get("hidden_size", 5376))
        self.global_num_heads = int(config.get("num_attention_heads", 56))
        if config.get("tensor_parallel", False):
            self.tp_group = config["device_mesh"].get_group(mesh_dim="tensor_p")
            self.tp_size = dist.get_world_size(self.tp_group)
            self.tp_rank = dist.get_rank(self.tp_group)
        else:
            self.tp_group = None
            self.tp_size = 1
            self.tp_rank = 0
        self.num_heads = self.global_num_heads // self.tp_size
        self.head_dim = int(config.get("attention_head_dim", 128))
        self.infer_dtype = GET_DTYPE()
        self.use_fused_qkv = bool(config.get("use_fused_qkv", False))
        self.use_fused_qkv_norm_rope = bool(config.get("use_fused_qkv_norm_rope", False))
        self.qkv_norm_rope_type = config.get("qkv_norm_rope_type", "triton")
        if self.qkv_norm_rope_type not in QKV_NORM_ROPE_REGISTER:
            raise ValueError(f"Unsupported qkv_norm_rope_type: {self.qkv_norm_rope_type}")
        self.qkv_norm_rope = QKV_NORM_ROPE_REGISTER[self.qkv_norm_rope_type]()
        self.fused_adaln = None
        if config.get("use_aiter_fused_adaln", False):
            # Explicitly enabled: fail at init rather than silently keeping the unfused chain.
            from aiter.ops.triton.fusions.fused_rmsnorm_indexed_adaln import fused_rmsnorm_indexed_adaln

            self.fused_adaln = fused_rmsnorm_indexed_adaln
            logger.info("MiniMax-H3 DiT AdaLN uses AITER fused_rmsnorm_indexed_adaln (use_aiter_fused_adaln=true)")
        if config.get("seq_parallel", False):
            self.seq_p_group = config["device_mesh"].get_group(mesh_dim="seq_p")
            parallel = config.get("parallel", {})
            self.seq_p_prepost_backend = parallel.get("seq_p_prepost_backend", "torch")
            self.seq_p_a2a_backend = parallel.get("seq_p_a2a_backend", "torch")
            self.seq_p_quant_scheme = parallel.get("seq_p_quant_scheme")
            if self.seq_p_quant_scheme is not None and self.seq_p_quant_scheme not in ("fp8", "fp4"):
                raise ValueError(f"Unknown seq_p_quant_scheme={self.seq_p_quant_scheme!r}; expected None, 'fp8', or 'fp4'.")
            self.seq_p_tensor_fusion = parallel.get("seq_p_tensor_fusion", False)
            self.seq_p_head_parallel = parallel.get("seq_p_head_parallel", False)
        else:
            self.seq_p_group = None
        self.infer_func = self.infer_without_offload
        self.use_adaln_cache = bool(config.get("use_adaln_cache", False))
        self._current_adaln_tables = None
        self._adaln_cache_hit = False
        if self.use_adaln_cache:
            self._adaln_cache, self._norm_out_cache = load_persistent_adaln_cache(config, self._cache_device())
        self.init_compile(config)

    @staticmethod
    def _cache_device():
        device_module = getattr(torch, AI_DEVICE)
        return torch.device(AI_DEVICE, device_module.current_device())

    def _gather_tp_last_dim(self, tensor):
        if self.tp_size == 1:
            return tensor
        gathered = [torch.empty_like(tensor) for _ in range(self.tp_size)]
        dist.all_gather(gathered, tensor.contiguous(), group=self.tp_group)
        return torch.cat(gathered, dim=-1)

    def _attention_routes_int4(self, weights, q):
        # sage_attn2_int4 blocks quantize Q/K/V themselves, so they skip the int8 fused sage prepare.
        routes_int4 = getattr(weights.calculate, "routes_int4", None)
        return routes_int4 is not None and routes_int4(self.block_idx, q.shape[0], q.device)

    def _int4_fused_prepare(self, weights, q, k, v, rotary_emb):
        # sage_int4 >= 0.1.3 fuses the Triton qkv_norm_rope math into its int4 prepare (no bf16 Q/K/V round trip).
        prepare = getattr(weights.calculate, "prepare_norm_rope", None)
        if prepare is None or self.seq_p_group is not None or not self.use_fused_qkv_norm_rope or self.qkv_norm_rope_type != "triton":
            return None
        freqs = fused_qkv.prepare_qkv_norm_rope(q, k, v, weights.norm_q, weights.norm_k, weights.rope, rotary_emb)
        if freqs is None:
            return None
        return prepare(q, k, v, weights.norm_q.weight, weights.norm_k.weight, *freqs, weights.norm_q.eps, weights.norm_k.eps)

    def _prepare_qkv(self, weights, hidden_states, rotary_emb):
        """Project QKV and return Q/K with normalization and RoPE applied."""
        if self.use_fused_qkv and weights.has_fused_qkv:
            q, k, v = weights.to_qkv.apply(hidden_states).chunk(3, dim=-1)
        else:
            if self.use_fused_qkv:
                _warn_fused_qkv_fallback_once()
            q = weights.to_q.apply(hidden_states)
            k = weights.to_k.apply(hidden_states)
            v = weights.to_v.apply(hidden_states)

        # Backends that only take the prepacked lane-major layout (sage_attn2_int8_exact) always use the fused prepare.
        needs_prepack = getattr(weights.calculate, "requires_h3_lane_major_prepack", False)
        if self._attention_routes_int4(weights, q):
            prepared = self._int4_fused_prepare(weights, q, k, v, rotary_emb)
            if prepared is not None:
                return prepared, None, None
        elif needs_prepack or os.environ.get("H3_SEPARATE_FUSED_SAGE_PREPARE", "0") == "1":
            try:
                k_block_s = int(os.environ.get("H3_SEPARATE_FUSED_K_BLOCK_S", "128"))
                v_block_s = int(os.environ.get("H3_SEPARATE_FUSED_V_BLOCK_S", "256"))
                return fused_sage_prepare_separate.separate_fused_sage_prepare(
                    q, k, v, weights.norm_q, weights.norm_k, rotary_emb,
                    num_heads=self.num_heads, head_dim=self.head_dim, k_block_s=k_block_s, v_block_s=v_block_s,
                    lane_major_kv=True if needs_prepack else None,
                ), None, None
            except Exception as exc:
                if needs_prepack or os.environ.get("H3_SEPARATE_FUSED_SAGE_PREPARE_STRICT", "0") == "1":
                    raise
                logger.warning("H3_SEPARATE_FUSED_SAGE_PREPARE failed; falling back to normal QKV prepare: {}", exc)

        if self.use_fused_qkv_norm_rope:
            fused = self.qkv_norm_rope.apply(
                q,
                k,
                v,
                weights.norm_q,
                weights.norm_k,
                weights.rope,
                rotary_emb,
            )
            # None means the norm/RoPE contract is unsupported or no usable
            # fused backend exists; continue through the ordinary path.
            if fused is not None:
                return fused
            _warn_qkv_norm_rope_fallback_once(self.qkv_norm_rope_type)

        q = weights.norm_q.apply(q.unflatten(-1, (self.num_heads, self.head_dim)))
        k = weights.norm_k.apply(k.unflatten(-1, (self.num_heads, self.head_dim)))
        v = v.unflatten(-1, (self.num_heads, self.head_dim))

        q, k = weights.rope.apply(
            q,
            k,
            rotary_emb,
            rotary_dim=rotary_emb[0].shape[-1],
        )
        return q, k, v

    def _attention(self, weights, hidden_states, pre_infer_out):
        q, k, v = self._prepare_qkv(weights, hidden_states, pre_infer_out.rotary_emb)
        sp_state = pre_infer_out.sequence_parallel_state
        attention_kwargs = {
            "causal": False,
            "scheduler": self.scheduler,
            "block_idx": self.block_idx,
        }
        if sp_state is None:
            seq_len = q.shape[0]
            cu_seqlens = torch.tensor((0, seq_len), dtype=torch.int32, device=q.device)
            out = weights.calculate.apply(
                q=q,
                k=k,
                v=v,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_kv=cu_seqlens,
                max_seqlen_q=seq_len,
                max_seqlen_kv=seq_len,
                **attention_kwargs,
            )
        else:
            aux_length = sp_state.aux_length
            out, aux_out = weights.calculate_parallel.apply(
                q=q[aux_length:].contiguous(),
                k=k[aux_length:].contiguous(),
                v=v[aux_length:].contiguous(),
                aux_q=q[:aux_length].contiguous() if aux_length else None,
                aux_k=k[:aux_length].contiguous() if aux_length else None,
                aux_v=v[:aux_length].contiguous() if aux_length else None,
                attention_module=weights.calculate,
                seq_p_group=self.seq_p_group,
                prepost_backend=self.seq_p_prepost_backend,
                a2a_backend=self.seq_p_a2a_backend,
                quant_scheme=self.seq_p_quant_scheme,
                tensor_fusion=self.seq_p_tensor_fusion,
                head_parallel=self.seq_p_head_parallel,
                aux_first=True,
                attention_kwargs=attention_kwargs,
            )
            if aux_out is not None:
                out = torch.cat((aux_out, out), dim=0)
        out = out.to(self.infer_dtype)
        projected = coresw_ffn_experiment.attention_out_projection(self, weights.to_out, out)
        if projected is not None:
            return projected
        return weights.to_out.apply(out)

    def _ff_residual(self, weights, hidden_states, residual, gate, indices):
        fused = coresw_ffn_experiment.feed_forward_residual(self, weights, hidden_states, residual, gate, indices)
        if fused is not None:
            return fused
        value, activation_gate = weights.in_proj.apply(hidden_states).chunk(2, dim=-1)
        projected = weights.out_proj.apply(value * F.silu(activation_gate))
        return residual + gate.index_select(0, indices) * projected

    @staticmethod
    def _ff(weights, hidden_states):
        value, gate = weights.in_proj.apply(hidden_states).chunk(2, dim=-1)
        return weights.out_proj.apply(value * F.silu(gate))

    def _fused_adaln_apply(self, norm, hidden_states, shift, scale, indices):
        return self.fused_adaln(hidden_states, norm._get_actual_weight(), shift, scale, indices, eps=norm.eps, round_intermediate=True)

    def infer_block(self, weights, hidden_states, pre_infer_out, modulation=None):
        # Keep the Python cache lookup outside the compiled block.
        if modulation is None:
            modulation = self._compute_adaln_table(weights, pre_infer_out)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulation.chunk(6, dim=-1)
        indices = pre_infer_out.adaln_indices

        residual = hidden_states
        if self.fused_adaln is not None:
            normed = self._fused_adaln_apply(weights.norm1, hidden_states, shift_msa, scale_msa, indices)
        else:
            normed = weights.norm1.apply(hidden_states)
            normed = normed * (1.0 + scale_msa.index_select(0, indices))
            normed = normed + shift_msa.index_select(0, indices)
        hidden_states = residual + gate_msa.index_select(0, indices) * self._attention(weights.attn, normed, pre_infer_out)

        residual = hidden_states
        if self.fused_adaln is not None:
            normed = self._fused_adaln_apply(weights.norm2, hidden_states, shift_mlp, scale_mlp, indices)
        else:
            normed = weights.norm2.apply(hidden_states)
            normed = normed * (1.0 + scale_mlp.index_select(0, indices))
            normed = normed + shift_mlp.index_select(0, indices)
        hidden_states = self._ff_residual(weights.ff, normed, residual, gate_mlp, indices)
        return hidden_states

    def _compute_adaln_table(self, weights, pre_infer_out):
        # ADALN CACHE SYNC: This projection is reproduced by the offline builder.
        # Keep the offline AdaLN cache builder and its tests aligned with changes
        # to activation placement, dtype, projection, gather, or reshape.
        # Activation is evaluated in fp32, then cast to the inference dtype
        # immediately before the (possibly quantized) AdaLN projection.
        if pre_infer_out.temb is None:
            raise RuntimeError("MiniMax-H3 timestep embedding is missing")
        modulation = weights.adaln.apply(F.silu(pre_infer_out.temb).to(self.infer_dtype))
        modulation = self._gather_tp_last_dim(modulation)
        return modulation.view(-1, 6 * self.hidden_size)

    def _clear_adaln_cache(self):
        # Online mode has no reusable cache; it only creates a temporary table
        # for the current block invocation in _compute_adaln_table().
        self._current_adaln_tables = None
        self._adaln_cache_hit = False

    def _prepare_adaln_cache(self, pre_infer_out):
        current_timesteps = tuple(self.scheduler.unique_timesteps_cpu.tolist())
        cached_tables = self._adaln_cache.get(current_timesteps)
        norm_out_modulation = self._norm_out_cache.get(current_timesteps)
        if cached_tables is not None and norm_out_modulation is not None:
            self._current_adaln_tables = cached_tables
            pre_infer_out.norm_out_modulation = norm_out_modulation
            self._adaln_cache_hit = True
        else:
            raise KeyError(f"Persistent MiniMax-H3 AdaLN cache has no entry for timesteps {current_timesteps}")

    def _get_cached_adaln(self, block_index):
        if self._adaln_cache_hit:
            return self._current_adaln_tables[block_index]
        raise RuntimeError("Persistent MiniMax-H3 AdaLN cache was not prepared")

    def run_block(self, block_idx, block, hidden_states, pre_infer_out):
        if self.use_adaln_cache:
            adaln_table = self._get_cached_adaln(block_idx)
            return super().run_block(block_idx, block, hidden_states, pre_infer_out, adaln_table)
        return super().run_block(block_idx, block, hidden_states, pre_infer_out)

    def infer_without_offload(self, blocks, hidden_states, pre_infer_out):
        for block_index, block in enumerate(blocks):
            self.block_idx = block_index
            hidden_states = self.run_block(block_index, block, hidden_states, pre_infer_out)
        return hidden_states

    def infer(self, block_weights, pre_infer_out):
        if self.use_adaln_cache:
            self._prepare_adaln_cache(pre_infer_out)
        return self.infer_func(block_weights.blocks, pre_infer_out.hidden_states, pre_infer_out)
