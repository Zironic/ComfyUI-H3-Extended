"""H3-owned replacement for the 50 main DiT attention forwards.

This is a small, deliberate fork of core MiniMax H3 ``Attention.forward``. It
exists because an ``optimized_attention_override`` cannot release the fused QKV
projection: the caller still holds Q/K/V views while the override runs.

The custom backend contract is two-stage. ``prepare`` consumes the values into
independent quantized tensors, then this module visibly deletes all three source
views before ``execute`` launches the expensive attention kernel.
"""

import torch

import comfy.ldm.minimax.model as h3_model
import comfy.model_management
import comfy.quant_ops

from .observer import notify_attention, marked_observed


def project_qkv(module, x, rope_freqs):
    """Mirror core's fused projection plus fused RMSNorm/RoPE."""
    seq = x.shape[0]
    inner = module.heads * module.head_dim

    # Do not bind the fused output separately. These three views are the only
    # Python references keeping the full allocation alive.
    q, k, v = module.qkv_proj(x).split(inner, dim=-1)
    v = v.view(seq, module.heads, module.head_dim)

    if rope_freqs is not None:
        if comfy.model_management.in_training:
            raise RuntimeError(
                "h3_attention.forward is inference-only; core training uses an "
                "out-of-place RMSNorm/RoPE path that defeats lifetime control.")
        q = q.view(1, seq, module.heads, module.head_dim)
        k = k.view(1, seq, module.heads, module.head_dim)
        qw = comfy.model_management.cast_to(module.q_norm.weight, device=x.device)
        kw = comfy.model_management.cast_to(module.k_norm.weight, device=x.device)
        rot = rope_freqs.shape[-3] * 2
        comfy.quant_ops.ck.rms_rope_split_half_(
            q, k, rope_freqs, qw, kw,
            epsilon=module.q_norm.eps,
            rot_dim=rot,
        )
        q = q[0]
        k = k[0]
    else:
        q = module.q_norm(q.view(seq, module.heads, module.head_dim))
        k = module.k_norm(k.view(seq, module.heads, module.head_dim))

    return q, k, v


def to_hnd(q, k, v):
    """``[seq, heads, dim]`` -> HND views; copies nothing."""
    return (
        q.transpose(0, 1).unsqueeze(0),
        k.transpose(0, 1).unsqueeze(0),
        v.transpose(0, 1).unsqueeze(0),
    )


def _legacy_attention(module, q, k, v, transformer_options, attention=None):
    attention_fn = attention if attention is not None else h3_model.optimized_attention
    # The custom forward already emitted the observation with an explicit layer.
    with marked_observed(transformer_options):
        return attention_fn(
            q, k, v, module.heads,
            mask=None,
            skip_reshape=True,
            transformer_options=transformer_options,
        )


def make_forward(module, layer_index, backend=None, attention=None):
    """Build one reversible block-forward replacement.

    ``backend`` is the production two-stage backend. ``attention`` is retained
    only for parity tests and characterization of the legacy path.
    """
    if backend is not None and attention is not None:
        raise ValueError("pass either backend or attention, not both")

    def forward(x, rope_freqs=None, transformer_options=None):
        transformer_options = transformer_options if transformer_options is not None else {}
        q, k, v = project_qkv(module, x, rope_freqs)
        q, k, v = to_hnd(q, k, v)

        with torch.no_grad():
            notify_attention(
                q, k, v,
                layer_index=layer_index,
                transformer_options=transformer_options,
            )

        if backend is None:
            out = _legacy_attention(
                module, q, k, v, transformer_options, attention=attention)
        else:
            prepared = backend.prepare(
                q, k, v,
                layer_index=layer_index,
                transformer_options=transformer_options,
            )

            # Load-bearing ownership boundary: after prepare returns, the
            # prepared object must not retain any view into the fused storage.
            del q, k, v

            out_hnd = backend.execute(prepared)
            del prepared
            if out_hnd.ndim != 4:
                raise RuntimeError(
                    "%s returned rank-%d output; expected HND rank 4"
                    % (getattr(backend, "name", type(backend).__name__), out_hnd.ndim))
            out = out_hnd.transpose(1, 2).reshape(
                out_hnd.shape[0], out_hnd.shape[2], module.heads * module.head_dim)

        return module.out_proj(out.squeeze(0))

    forward._h3_attention = True
    forward._h3_layer_index = layer_index
    forward._h3_backend = getattr(backend, "name", None)
    return forward

# Backward-compatible imports for the existing characterization test. The guard
# now belongs to the Sage-specific V preparation path and is not called by the
# generic forward.
from .sage_mem_eff import V_OFFSET_LIMIT, guard_v_stride  # noqa: E402,F401
