"""H3-owned replacement for the DiT block attention forward.

This is a deliberate, small fork of `comfy.ldm.minimax.model.Attention.forward`.
It exists because an `optimized_attention_override` cannot control activation
lifetime: core's forward holds `q`, `k` and `v` as locals across the attention
call, and those are views into one fused QKV projection, so the whole bf16 buffer
stays resident through the attention peak. At C=73 that buffer is 1.518 GB - 36%
of the measured sampling transient. Owning the forward is the only place the
references can be dropped early. See PLAN.md §1.6.

The cost of that is a copy of a core method that can drift silently. The parity
tests in tests/test_h3_attention_forward.py turn drift into a loud failure rather
than a subtle numerical one; they do not prevent it. Re-run them after a ComfyUI
update that touches the MiniMax model.
"""

import torch

import comfy.ldm.minimax.model as h3_model
import comfy.model_management
import comfy.quant_ops

from .observer import notify_attention


def project_qkv(module, x, rope_freqs):
    """Fused QKV projection plus fused RMSNorm/RoPE, mirroring core.

    Returns `(q, k, v)` as `[seq, heads, head_dim]` views into a single fused
    allocation. The caller owns their lifetime - that is the whole point.
    """
    seq = x.shape[0]
    inner = module.heads * module.head_dim

    # the fused projection output is deliberately never bound to a local: q, k
    # and v are the only references keeping it alive, so dropping them frees it
    q, k, v = module.qkv_proj(x).split(inner, dim=-1)
    v = v.view(seq, module.heads, module.head_dim)

    if rope_freqs is not None:
        if comfy.model_management.in_training:
            raise RuntimeError(
                "h3_attention.forward is inference-only; core's training branch "
                "uses the out-of-place rms_rope_split_half, which defeats the "
                "activation-lifetime control this forward exists for.")
        q = q.view(1, seq, module.heads, module.head_dim)
        k = k.view(1, seq, module.heads, module.head_dim)
        qw = comfy.model_management.cast_to(module.q_norm.weight, device=x.device)
        kw = comfy.model_management.cast_to(module.k_norm.weight, device=x.device)
        # partial rotary: the table carries rot_dim/2 pair-rotations, the norm
        # always spans the full head_dim
        rot = rope_freqs.shape[-3] * 2
        comfy.quant_ops.ck.rms_rope_split_half_(
            q, k, rope_freqs, qw, kw, epsilon=module.q_norm.eps, rot_dim=rot)
        q = q[0]
        k = k[0]
    else:
        q = module.q_norm(q.view(seq, module.heads, module.head_dim))
        k = module.k_norm(k.view(seq, module.heads, module.head_dim))

    return q, k, v


def to_hnd(q, k, v):
    """`[seq, heads, dim]` -> `[1, heads, seq, dim]`. Views; copies nothing."""
    return (q.transpose(0, 1).unsqueeze(0),
            k.transpose(0, 1).unsqueeze(0),
            v.transpose(0, 1).unsqueeze(0))


def make_forward(module, layer_index, attention=None):
    """Build the replacement forward for one block's attention module.

    `add_object_patch` sets this on the instance, so it is called unbound: the
    module is closed over rather than arriving as `self`.

    `attention` defaults to whatever `comfy.ldm.minimax.model.optimized_attention`
    is at call time, so probe install/uninstall is still honored. Commit 2 uses
    that default and changes no numerics; a memory-efficient backend replaces it.
    """

    def forward(x, rope_freqs=None, transformer_options={}):
        q, k, v = project_qkv(module, x, rope_freqs)
        q, k, v = to_hnd(q, k, v)

        # observers see post-norm, post-rope Q/K in the same HND layout the
        # module-global path delivers, but with a real block index attached
        with torch.no_grad():
            notify_attention(q, k, layer_index=layer_index,
                             transformer_options=transformer_options)

        attention_fn = attention if attention is not None else h3_model.optimized_attention
        out = attention_fn(q, k, v, module.heads, mask=None, skip_reshape=True,
                           transformer_options=transformer_options)
        return module.out_proj(out.squeeze(0))

    forward._h3_attention = True
    forward._h3_layer_index = layer_index
    return forward
