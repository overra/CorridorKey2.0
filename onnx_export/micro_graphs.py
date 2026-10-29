#!/usr/bin/env python3
"""Single-op ONNX graphs at CorridorKey shapes for WebGPU microbenchmarks.

Each case is a tiny fp16 graph with random weights; web/micro.html times it.
Pairs ("x" vs "x_alt") compare an op as exported against a rewrite.

    python onnx_export/micro_graphs.py --out web/models/micro
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

F16 = TensorProto.FLOAT16
rng = np.random.default_rng(0)


def w(*shape, scale=0.02):
    return (rng.standard_normal(shape) * scale).astype(np.float16)


def model(nodes, inputs, outputs, inits):
    g = helper.make_graph(nodes, "micro", inputs, outputs,
                          [numpy_helper.from_array(v, k) for k, v in inits.items()])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 18)])
    m.ir_version = 10
    return m


def vi(name, shape):
    return helper.make_tensor_value_info(name, F16, shape)


def add_chain(names, out):
    nodes, acc = [], names[0]
    for i, n in enumerate(names[1:]):
        dst = out if i == len(names) - 2 else f"{out}_{i}"
        nodes.append(helper.make_node("Add", [acc, n], [dst]))
        acc = dst
    return nodes


def dw2_conv(C, H, W, B=1):
    """Exported DepthwiseSwiGLU2D: one grouped conv with 2 outputs per group."""
    x = vi("x", [B, C, H + 6, W + 6])
    y = vi("y", [B, 2 * C, H, W])
    return model([helper.make_node("Conv", ["x", "w", "b"], ["y"], group=C, kernel_shape=[7, 7])],
                 [x], [y], {"w": w(2 * C, 1, 7, 7), "b": w(2 * C)})


def dw2_split(C, H, W, B=1):
    """Same math as two standard depthwise convs (value, gate) + SiLU gate."""
    x = vi("x", [B, C, H + 6, W + 6])
    y = vi("y", [B, C, H, W])
    nodes = [
        helper.make_node("Conv", ["x", "wv", "bv"], ["v"], group=C, kernel_shape=[7, 7]),
        helper.make_node("Conv", ["x", "wg", "bg"], ["g"], group=C, kernel_shape=[7, 7]),
        helper.make_node("Sigmoid", ["g"], ["sg"]),
        helper.make_node("Mul", ["g", "sg"], ["silu"]),
        helper.make_node("Mul", ["v", "silu"], ["y"]),
    ]
    return model(nodes, [x], [y], {"wv": w(C, 1, 7, 7), "bv": w(C), "wg": w(C, 1, 7, 7), "bg": w(C)})


def deconv(C, H, W, k, s, p, out=4):
    """Exported overlap Depatchify: ConvTranspose kernel 2s, stride s, pad s/2."""
    x = vi("x", [1, C, H, W])
    y = vi("y", [1, out, H * s, W * s])
    return model([helper.make_node("ConvTranspose", ["x", "w", "b"], ["y"], kernel_shape=[k, k],
                                   strides=[s, s], pads=[p, p, p, p])],
                 [x], [y], {"w": w(C, out, k, k), "b": w(out)})


def deconv_alt(C, H, W, s, out=4, weight=None, bias=None):
    """ConvTranspose(kernel 2s, stride s, pad s/2) rewritten without ConvTranspose.

    Split the kernel into its four s x s quadrants (qy, qx). Each quadrant is a
    1x1 conv to out*s*s channels + DepthToSpace(s), i.e. a non-overlapping
    upsample; quadrant q lands one token later along an axis when q = 1. With
    the tokens zero-padded by one on each side, quadrant (qy, qx) is the crop
    starting at (p + (1 - qy) s, p + (1 - qx) s), and the output is their sum.
    """
    weight = weight if weight is not None else w(C, out, 2 * s, 2 * s)
    bias = bias if bias is not None else w(out)
    p = s // 2
    x = vi("x", [1, C, H, W])
    y = vi("y", [1, out, H * s, W * s])
    nodes = [helper.make_node("Pad", ["x", "pads"], ["xp"], mode="constant")]
    inits = {"pads": np.array([0, 0, 1, 1, 0, 0, 1, 1], np.int64)}
    parts = []
    for qy in (0, 1):
        for qx in (0, 1):
            q = f"{qy}{qx}"
            sub = weight[:, :, qy * s:(qy + 1) * s, qx * s:(qx + 1) * s]       # (C, out, s, s)
            inits[f"w{q}"] = np.ascontiguousarray(
                sub.transpose(1, 2, 3, 0).reshape(out * s * s, C, 1, 1))     # CRD channel order
            nodes.append(helper.make_node("Conv", ["xp", f"w{q}"], [f"c{q}"], kernel_shape=[1, 1]))
            nodes.append(helper.make_node("DepthToSpace", [f"c{q}"], [f"d{q}"], blocksize=s, mode="CRD"))
            y0, x0 = p + (1 - qy) * s, p + (1 - qx) * s
            inits[f"st{q}"] = np.array([y0, x0], np.int64)
            inits[f"en{q}"] = np.array([y0 + H * s, x0 + W * s], np.int64)
            nodes.append(helper.make_node("Slice", [f"d{q}", f"st{q}", f"en{q}", "axes"], [f"s{q}"]))
            parts.append(f"s{q}")
    inits["axes"] = np.array([2, 3], np.int64)
    nodes += add_chain(parts, "sum")  # ORT WebGPU has no Sum kernel (CPU fallback)
    inits["bias"] = bias.reshape(1, out, 1, 1)
    nodes.append(helper.make_node("Add", ["sum", "bias"], ["y"]))
    return model(nodes, [x], [y], inits)


def deconv_mm(C, H, W, s, out=4, weight=None, bias=None):
    """ConvTranspose(kernel 2s, stride s, pad s/2) as one MatMul producing every
    token's 2s x 2s output patch, then col2im: the four s x s quadrants are laid
    out with Reshape/Transpose (no DepthToSpace), padded one token toward
    their landing side, and summed."""
    weight = weight if weight is not None else w(C, out, 2 * s, 2 * s)
    bias = bias if bias is not None else w(out)
    p = s // 2
    x = vi("x", [1, C, H, W])
    y = vi("y", [1, out, H * s, W * s])
    # (C, out, 2, s, 2, s) -> (C, [qy, qx, out, s, s]) so each quadrant is contiguous
    wq = weight.reshape(C, out, 2, s, 2, s).transpose(0, 2, 4, 1, 3, 5).reshape(C, 4 * out * s * s)
    inits = {"w": np.ascontiguousarray(wq), "perm_shape": np.array([1, C, H * W], np.int64)}
    nodes = [
        helper.make_node("Reshape", ["x", "perm_shape"], ["xf"]),                    # (1, C, HW)
        helper.make_node("Transpose", ["xf"], ["xt"], perm=[0, 2, 1]),               # (1, HW, C)
        helper.make_node("MatMul", ["xt", "w"], ["pm"]),                             # (1, HW, 4*out*s*s)
    ]
    inits["q_shape"] = np.array([H, W, 4, out, s, s], np.int64)
    nodes.append(helper.make_node("Reshape", ["pm", "q_shape"], ["pq"]))
    inits["img_shape"] = np.array([4, out, H * s, W * s], np.int64)
    nodes.append(helper.make_node("Transpose", ["pq"], ["pt"], perm=[2, 3, 0, 4, 1, 5]))  # (4, out, H, s, W, s)
    nodes.append(helper.make_node("Reshape", ["pt", "img_shape"], ["imgs"]))       # (4, out, Hs, Ws)
    parts = []
    for qi, (qy, qx) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
        inits[f"i{qi}"] = np.array([qi], np.int64)
        nodes.append(helper.make_node("Gather", ["imgs", f"i{qi}"], [f"g{qi}"], axis=0))   # (1, out, Hs, Ws)
        # quadrant q lands q tokens later; output crop starts at p
        top, left = qy * s - p, qx * s - p
        pads = [0, 0, max(top, 0), max(left, 0), 0, 0, max(-top, 0), max(-left, 0)]
        inits[f"pd{qi}"] = np.array(pads, np.int64)
        nodes.append(helper.make_node("Pad", [f"g{qi}", f"pd{qi}"], [f"pp{qi}"], mode="constant"))
        st = [max(-top, 0), max(-left, 0)]
        inits[f"st{qi}"] = np.array(st, np.int64)
        inits[f"en{qi}"] = np.array([st[0] + H * s, st[1] + W * s], np.int64)
        nodes.append(helper.make_node("Slice", [f"pp{qi}", f"st{qi}", f"en{qi}", "axes"], [f"s{qi}"]))
        parts.append(f"s{qi}")
    inits["axes"] = np.array([2, 3], np.int64)
    nodes += add_chain(parts, "sum")  # ORT WebGPU has no Sum kernel (CPU fallback)
    inits["bias"] = bias.reshape(1, out, 1, 1)
    nodes.append(helper.make_node("Add", ["sum", "bias"], ["y"]))
    return model(nodes, [x], [y], inits)


def matmul(M, K, N, gemm=False):
    a = vi("a", [M, K])
    y = vi("y", [M, N])
    if gemm:
        nodes = [helper.make_node("Gemm", ["a", "b", "c"], ["y"])]
        return model(nodes, [a], [y], {"b": w(K, N), "c": w(N)})
    nodes = [helper.make_node("MatMul", ["a", "b"], ["m"]), helper.make_node("Add", ["m", "c"], ["y"])]
    return model(nodes, [a], [y], {"b": w(K, N), "c": w(N)})


def conv(cin, cout, H, W, k, st, p, groups=1):
    x = vi("x", [1, cin, H, W])
    Ho, Wo = (H + 2 * p - k) // st + 1, (W + 2 * p - k) // st + 1
    y = vi("y", [1, cout, Ho, Wo])
    return model([helper.make_node("Conv", ["x", "w", "b"], ["y"], kernel_shape=[k, k], strides=[st, st],
                                   pads=[p, p, p, p], group=groups)],
                 [x], [y], {"w": w(cout, cin // groups, k, k), "b": w(cout)})


def reflect_pad(C, H, W, p=3):
    x = vi("x", [1, C, H, W])
    y = vi("y", [1, C, H + 2 * p, W + 2 * p])
    return model([helper.make_node("Pad", ["x", "pads"], ["y"], mode="reflect")], [x], [y],
                 {"pads": np.array([0, 0, p, p, 0, 0, p, p], np.int64)})


def softmax(shape):
    x = vi("x", shape)
    return model([helper.make_node("Softmax", ["x"], ["y"], axis=-1)], [x], [vi("y", shape)], {})


def bmm(a_shape, b_shape):
    out = a_shape[:-1] + [b_shape[-1]]
    return model([helper.make_node("MatMul", ["a", "b"], ["y"])], [vi("a", a_shape), vi("b", b_shape)],
                 [vi("y", out)], {})


def rms_decomposed(M, C):
    """The exported F.rms_norm: Pow, ReduceMean, Add, Sqrt, Reciprocal, Mul, Mul."""
    x = vi("x", [M, C])
    nodes = [
        helper.make_node("Pow", ["x", "two"], ["sq"]),
        helper.make_node("ReduceMean", ["sq", "ax"], ["ms"], keepdims=1),
        helper.make_node("Add", ["ms", "eps"], ["mse"]),
        helper.make_node("Sqrt", ["mse"], ["rt"]),
        helper.make_node("Reciprocal", ["rt"], ["inv"]),
        helper.make_node("Mul", ["x", "inv"], ["xn"]),
        helper.make_node("Mul", ["xn", "g"], ["y"]),
    ]
    return model(nodes, [x], [vi("y", [M, C])], {"two": np.array(2, np.float16), "ax": np.array([-1], np.int64),
                                                   "eps": np.array(1e-6, np.float16), "g": w(C)})


def rms_fused(M, C):
    x = vi("x", [M, C])
    return model([helper.make_node("SimplifiedLayerNormalization", ["x", "g"], ["y"], axis=-1, epsilon=1e-6, stash_type=1)],
                 [x], [vi("y", [M, C])], {"g": w(C)})


def rope_decomposed(shape):
    """rope.apply_rotary as exported: strided Slices, 4 Mul, Sub, Add, stack, flatten."""
    *lead, N, d = shape
    x = vi("x", shape)
    h = d // 2
    nodes = [
        helper.make_node("Slice", ["x", "s0", "e", "ax", "st"], ["x1"]),
        helper.make_node("Slice", ["x", "s1", "e", "ax", "st"], ["x2"]),
        helper.make_node("Mul", ["x1", "cos"], ["a"]), helper.make_node("Mul", ["x2", "sin"], ["b"]),
        helper.make_node("Sub", ["a", "b"], ["o1"]),
        helper.make_node("Mul", ["x1", "sin"], ["c"]), helper.make_node("Mul", ["x2", "cos"], ["dd"]),
        helper.make_node("Add", ["c", "dd"], ["o2"]),
        helper.make_node("Unsqueeze", ["o1", "lax"], ["u1"]), helper.make_node("Unsqueeze", ["o2", "lax"], ["u2"]),
        helper.make_node("Concat", ["u1", "u2"], ["st2"], axis=-1),
        helper.make_node("Reshape", ["st2", "shp"], ["y"]),
    ]
    inits = {"s0": np.array([0], np.int64), "s1": np.array([1], np.int64), "e": np.array([d], np.int64),
             "ax": np.array([-1], np.int64), "st": np.array([2], np.int64), "lax": np.array([-1], np.int64),
             "cos": w(N, h, scale=1.0), "sin": w(N, h, scale=1.0), "shp": np.array(shape, np.int64)}
    return model(nodes, [x], [vi("y", shape)], inits)


def window_transpose():
    """window_partition_2d on the stage-2 grid: (1, 2, 16, 4, 16, 1280) perm (0, 1, 3, 2, 4, 5)."""
    shape = [1, 2, 16, 4, 16, 1280]
    out = [1, 2, 4, 16, 16, 1280]
    return model([helper.make_node("Transpose", ["x"], ["y"], perm=[0, 1, 3, 2, 4, 5])],
                 [vi("x", shape)], [vi("y", out)], {})


CASES = {
    # stage 2 local mixer, d=1280 at /16 of 896x512 (32x56 tokens)
    "s2_dw2_conv": (dw2_conv(1280, 32, 56), {"x": [1, 1280, 38, 62]}),
    "s2_dw2_split": (dw2_split(1280, 32, 56), {"x": [1, 1280, 38, 62]}),
    # stage 1 local mixer, d=512, T=4 frames
    "s1_dw2_conv": (dw2_conv(512, 32, 56, B=4), {"x": [4, 512, 38, 62]}),
    "s1_dw2_split": (dw2_split(512, 32, 56, B=4), {"x": [4, 512, 38, 62]}),
    # depatchify heads
    "s2_deconv": (deconv(1280, 32, 56, 32, 16, 8), {"x": [1, 1280, 32, 56]}),
    "s2_deconv_alt": (deconv_alt(1280, 32, 56, 16), {"x": [1, 1280, 32, 56]}),
    "s2_deconv_mm": (deconv_mm(1280, 32, 56, 16), {"x": [1, 1280, 32, 56]}),
    "s1_aux_deconv": (deconv(512, 32, 56, 16, 8, 4), {"x": [1, 512, 32, 56]}),
    # patchify / misc convs at stage-2 shapes
    "s2_patchify": (conv(3, 768, 512, 896, 32, 16, 8), {"x": [1, 3, 512, 896]}),
    "s2_qkv_dw": (conv(3840, 3840, 38, 62, 7, 1, 0, groups=3840), {"x": [1, 3840, 38, 62]}),
    "s2_reflect_pad": (reflect_pad(1280, 32, 56), {"x": [1, 1280, 32, 56]}),
    "s1_down_k8s4": (conv(512, 1536, 32, 56, 8, 4, 2), {"x": [1, 512, 32, 56]}),
    # attention at stage-2 full-spatial shape (16 heads, 1792 tokens, d=80)
    "attn_qk": (bmm([16, 1792, 80], [16, 80, 1792]), {"a": [16, 1792, 80], "b": [16, 80, 1792]}),
    "attn_softmax": (softmax([16, 1792, 1792]), {"x": [16, 1792, 1792]}),
    "attn_pv": (bmm([16, 1792, 1792], [16, 1792, 80]), {"a": [16, 1792, 1792], "b": [16, 1792, 80]}),
    "win_attn_qk": (bmm([128, 256, 80], [128, 80, 256]), {"a": [128, 256, 80], "b": [128, 80, 256]}),
    # norms / rope / layout at stage-2 shapes
    "rms_decomposed": (rms_decomposed(1792, 1280), {"x": [1792, 1280]}),
    "rms_fused": (rms_fused(1792, 1280), {"x": [1792, 1280]}),
    "rope_decomposed": (rope_decomposed([1, 16, 1792, 80]), {"x": [1, 16, 1792, 80]}),
    "win_transpose": (window_transpose(), {"x": [1, 2, 16, 4, 16, 1280]}),
    # stage 2 dense layers (1792 tokens): qkv, mlp up/gate, mlp down
    "mm_qkv": (matmul(1792, 1280, 3840), {"a": [1792, 1280]}),
    "gemm_qkv": (matmul(1792, 1280, 3840, gemm=True), {"a": [1792, 1280]}),
    "mm_up": (matmul(1792, 1280, 3200), {"a": [1792, 1280]}),
    "mm_down": (matmul(1792, 3200, 1280), {"a": [1792, 3200]}),
    "gemm_down": (matmul(1792, 3200, 1280, gemm=True), {"a": [1792, 3200]}),
}
FLOPS = {
    "attn_qk": 2 * 16 * 1792 * 1792 * 80, "attn_pv": 2 * 16 * 1792 * 1792 * 80,
    "s2_patchify": 2 * 1792 * 768 * 3 * 32 * 32, "s1_down_k8s4": 2 * 112 * 1536 * 512 * 64,
    "mm_qkv": 2 * 1792 * 1280 * 3840, "gemm_qkv": 2 * 1792 * 1280 * 3840,
    "mm_up": 2 * 1792 * 1280 * 3200, "mm_down": 2 * 1792 * 3200 * 1280, "gemm_down": 2 * 1792 * 3200 * 1280,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    root = Path(args.out)
    manifest = {}
    for name, (m, inputs) in CASES.items():
        onnx.checker.check_model(m)
        d = root / name / "onnx"
        d.mkdir(parents=True, exist_ok=True)
        onnx.save_model(m, str(d / "model.onnx"))
        (root / name / "config.json").write_text(json.dumps({"model_type": f"micro_{name}"}))
        manifest[name] = {"inputs": inputs, "flops": FLOPS.get(name)}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"wrote {len(CASES)} cases to {root}")


if __name__ == "__main__":
    main()
