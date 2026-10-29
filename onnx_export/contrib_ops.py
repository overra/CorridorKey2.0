"""Export CorridorKey's RoPE as one com.microsoft::RotaryEmbedding node.

rope.apply_rotary rotates interleaved pairs (x[2i], x[2i+1]) by per-token
angles. As exported it is ~12 small kernels per q/k (strided slices, four
multiplies, stack, reshape). RotaryEmbedding(interleaved=1) computes the same
thing in one kernel when given the per-token (N, d/2) cos/sin table as its
"cache" and position offset 0, so token i reads row i.

    install_rotary()  # before export; patches corridorkey.blocks.apply_rotary
    torch.onnx.export(..., custom_translation_table=CUSTOM_TRANSLATIONS)
"""

from __future__ import annotations

import onnxscript
import torch
from onnxscript import opset18 as op

from corridorkey import blocks
from corridorkey.rope import apply_rotary as _apply_rotary_ref


def _register_rotary_schema() -> None:
    """onnxscript resolves ops through ONNX's schema registry, which does not
    know ORT's com.microsoft ops; register the one we emit."""
    from onnx import defs
    if defs.has("RotaryEmbedding", "com.microsoft"):
        return
    P = defs.OpSchema.FormalParameter
    defs.register_schema(defs.OpSchema(
        "RotaryEmbedding", "com.microsoft", 1,
        inputs=[P("input", "T"), P("position_ids", "M"), P("cos_cache", "T"), P("sin_cache", "T")],
        outputs=[P("output", "T")],
        type_constraints=[("T", ["tensor(float)", "tensor(float16)"], ""), ("M", ["tensor(int64)"], "")],
        attributes=[defs.OpSchema.Attribute("interleaved", defs.OpSchema.AttrType.INT, "")],
    ))


_register_rotary_schema()
_MSFT = onnxscript.values.Opset("com.microsoft", 1)


@torch.library.custom_op("ck::rotary", mutates_args=())
def ck_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (B, heads, N, d); cos/sin: (N, d/2)."""
    return _apply_rotary_ref(x, cos, sin)


@ck_rotary.register_fake
def _(x, cos, sin):
    return torch.empty_like(x)


def _rotary_onnx(x, cos, sin):
    position_offset = op.Constant(value_ints=[0])
    return _MSFT.RotaryEmbedding(x, position_offset, op.CastLike(cos, x), op.CastLike(sin, x), interleaved=1)


def _apply_rotary_export(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    if x.dim() == 4 and cos.dim() == 2:
        return ck_rotary(x, cos.to(x.dtype), sin.to(x.dtype))
    return _apply_rotary_ref(x, cos, sin)


CUSTOM_TRANSLATIONS = {torch.ops.ck.rotary.default: _rotary_onnx}


def install_rotary() -> None:
    blocks.apply_rotary = _apply_rotary_export


def uninstall_rotary() -> None:
    blocks.apply_rotary = _apply_rotary_ref
