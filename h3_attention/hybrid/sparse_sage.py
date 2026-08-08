"""Sparse-Sage executor used by the H3 hybrid backend."""

from dataclasses import dataclass
import importlib
import importlib.metadata

import torch

from .router import KV_TILE, Q_TILE


class SparseSageError(RuntimeError):
    pass


@dataclass(frozen=True)
class SparseSageAPI:
    version: str
    block_sparse: object


@dataclass
class PreparedSparseSage:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    mask_id: torch.Tensor
    output_dtype: torch.dtype
    layer_index: int
    sequence: int
    heads: int
    metadata: dict


def load_sparse_sage_api():
    """Import the public API and both compiled extensions it depends on."""
    try:
        module = importlib.import_module("spas_sage_attn")
        importlib.import_module("spas_sage_attn._fused")
        block_sparse = getattr(module, "block_sparse_sage2_attn_cuda")
        version = importlib.metadata.version("spas-sage-attn")
    except Exception as exc:
        raise SparseSageError(
            "Hybrid Sparse Attention requires SpargeAttention with its compiled "
            "_qattn and _fused extensions"
        ) from exc
    qattn_error = None
    for extension in ("spas_sage_attn._qattn_sm89", "spas_sage_attn._qattn"):
        try:
            importlib.import_module(extension)
            break
        except Exception as exc:
            qattn_error = exc
    else:
        raise SparseSageError(
            "Hybrid Sparse Attention requires SpargeAttention's SM89 _qattn extension"
        ) from qattn_error
    if not callable(block_sparse):
        raise SparseSageError(
            "SpargeAttention does not expose callable block_sparse_sage2_attn_cuda"
        )
    return SparseSageAPI(version=version, block_sparse=block_sparse)


def preflight_sparse_sage(api_loader=load_sparse_sage_api, cuda_available=None,
                          capability_getter=None):
    """Fail before model mutation unless the installed extension targets Ada."""
    api = api_loader()
    cuda_available = cuda_available or torch.cuda.is_available
    capability_getter = capability_getter or torch.cuda.get_device_capability
    if not cuda_available():
        raise SparseSageError("Hybrid Sparse Attention requires CUDA")
    capability = tuple(capability_getter())
    if capability != (8, 9):
        raise SparseSageError(
            "Hybrid Sparse Attention Phase A is SM89-only; device capability is %d.%d"
            % capability
        )
    return api


class SparseSageExecutor:
    def __init__(self, api, *, allow_cpu_for_tests=False):
        if api is None:
            raise TypeError("SparseSageExecutor requires a preflighted API")
        self.api = api
        self.allow_cpu_for_tests = bool(allow_cpu_for_tests)

    def _validate(self, q, k, v, mask_id):
        if q.shape != k.shape or q.shape != v.shape:
            raise SparseSageError(
                "Sparse Sage requires equal self-attention Q/K/V shapes; got %s %s %s"
                % (tuple(q.shape), tuple(k.shape), tuple(v.shape))
            )
        if q.ndim != 4:
            raise SparseSageError("Sparse Sage expects HND rank-4 tensors")
        batch, heads, sequence, head_dim = q.shape
        if batch != 1:
            raise SparseSageError("released H3 expects attention batch 1; got %d" % batch)
        if head_dim != 128:
            raise SparseSageError("Sparse Sage requires head_dim 128; got %d" % head_dim)
        if q.dtype not in (torch.float16, torch.bfloat16):
            raise SparseSageError("Sparse Sage requires fp16 or bf16 Q/K/V")
        if q.dtype != k.dtype or q.dtype != v.dtype:
            raise SparseSageError("Sparse Sage Q/K/V dtypes differ")
        if q.device != k.device or q.device != v.device:
            raise SparseSageError("Sparse Sage Q/K/V devices differ")
        if q.stride(-1) != 1 or k.stride(-1) != 1 or v.stride(-1) != 1:
            raise SparseSageError("Sparse Sage Q/K/V last dimension must be contiguous")
        expected_mask = (
            batch,
            heads,
            (sequence + Q_TILE - 1) // Q_TILE,
            (sequence + KV_TILE - 1) // KV_TILE,
        )
        if tuple(mask_id.shape) != expected_mask:
            raise SparseSageError(
                "Sparse Sage mask shape is %s; expected %s"
                % (tuple(mask_id.shape), expected_mask)
            )
        if mask_id.dtype != torch.bool or not mask_id.is_contiguous():
            raise SparseSageError("Sparse Sage mask must be contiguous bool")
        if mask_id.device != q.device:
            raise SparseSageError("Sparse Sage mask and Q/K/V devices differ")
        if not self.allow_cpu_for_tests:
            if not q.is_cuda:
                raise SparseSageError("Sparse Sage requires CUDA")
            capability = tuple(torch.cuda.get_device_capability(q.device))
            if capability != (8, 9):
                raise SparseSageError(
                    "Hybrid Sparse Attention Phase A is SM89-only; device capability is %d.%d"
                    % capability
                )
        return heads, sequence

    def prepare(self, q, k, v, mask_id, *, layer_index, metadata):
        heads, sequence = self._validate(q, k, v, mask_id)
        if not self.allow_cpu_for_tests:
            torch.cuda.set_device(q.device)
        q_prepared = q.contiguous()
        k_prepared = k.contiguous()
        v_prepared = torch.empty(v.shape, dtype=torch.float16, device=v.device)
        v_prepared.copy_(v)
        return PreparedSparseSage(
            q=q_prepared,
            k=k_prepared,
            v=v_prepared,
            mask_id=mask_id,
            output_dtype=q.dtype,
            layer_index=int(layer_index),
            sequence=int(sequence),
            heads=int(heads),
            metadata=dict(metadata),
        )

    def execute(self, prepared):
        try:
            output = self.api.block_sparse(
                prepared.q,
                prepared.k,
                prepared.v,
                mask_id=prepared.mask_id,
                scale=128 ** -0.5,
                tensor_layout="HND",
            )
        except Exception as exc:
            raise SparseSageError(
                "Sparse Sage kernel failed: layer=%d sequence=%d heads=%d "
                "dtype=%s SpargeAttention=%s"
                % (
                    prepared.layer_index,
                    prepared.sequence,
                    prepared.heads,
                    prepared.output_dtype,
                    self.api.version,
                )
            ) from exc
        if not torch.is_tensor(output) or output.shape != prepared.q.shape:
            raise SparseSageError(
                "Sparse Sage returned %s; expected HND shape %s"
                % (getattr(output, "shape", type(output).__name__), tuple(prepared.q.shape))
            )
        if output.dtype != prepared.output_dtype:
            raise SparseSageError(
                "Sparse Sage returned %s; expected %s"
                % (output.dtype, prepared.output_dtype)
            )
        return output
