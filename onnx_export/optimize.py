#!/usr/bin/env python3
"""Turn exported fp32 graphs into browser variants.

    python onnx_export/optimize.py --models web/models/896x512 --variant fp16
    python onnx_export/optimize.py --models web/models/896x512 --variant q4f16

Reads <models>/raw/<component>.onnx and writes the transformers.js layout:
<models>/<component>/onnx/model_<variant>.onnx (+ model_<variant>.onnx_data),
registered in <models>/config.json, so a preset folder can be served (or
uploaded to the Hugging Face Hub) as one model repo.

Variants (graph inputs/outputs are fp32 in all of them):
    fp32   reference; fp32 weights and activations.
    fp16   fp16 weights and activations.
    q8     8-bit weight-only MatMulNBits for every constant-weight MatMul,
           fp32 activations (dequantized in-shader, fp32 accumulation).
    q4     the same with 4-bit weights.
    q4f16  4-bit MatMulNBits weights with fp16 activations.
    q8f16  8-bit MatMulNBits weights with fp16 activations.
MatMulNBits only covers MatMul: convolutions exported as Conv keep the
activation precision, while the strided convolutions and depatchify heads that
export.py rewrites as MatMuls (patches.install_conv_rewrites) are quantized
like any other weight; the measured q8 accuracy includes them. Weights are
never stored narrower and cast up in-graph: ORT Web would fold those casts on
the CPU and pull every weight through the 4 GiB wasm heap.

Files are named model_<variant>.onnx (+ .onnx_data) for every variant; the
browser loads them with transformers.js `model_file_name: 'model_<variant>'`.

Quantization runs on the raw export, before ORT's EP-agnostic basic pass
(constant folding, redundant node removal), because that pass fuses
MatMul+Add into Gemm, which MatMulNBits cannot cover.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import onnx
import onnxruntime as ort

from common import update_preset_config

COMPONENTS = ("cradio", "moge", "rvm", "stage1", "stage2")


def save(model: onnx.ModelProto, path: Path, data: str | None = None) -> None:
    data = data or path.name + ".data"
    for p in (path, path.parent / data):
        if p.exists():
            p.unlink()
    onnx.save_model(model, str(path), save_as_external_data=True,
                    all_tensors_to_one_file=True, location=data, size_threshold=1024)


def ort_basic_optimize(src: Path, dst: Path) -> None:
    """Offline ORT_ENABLE_BASIC pass (portable across execution providers)."""
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    so.optimized_model_filepath = str(dst)
    so.add_session_config_entry(
        "session.optimized_model_external_initializers_file_name", dst.name + ".data")
    so.add_session_config_entry(
        "session.optimized_model_external_initializers_min_size_in_bytes", "1024")
    so.log_severity_level = 3
    ort.InferenceSession(str(src), so, providers=["CPUExecutionProvider"])


def to_fp16(model: onnx.ModelProto) -> onnx.ModelProto:
    from onnxconverter_common import float16

    model = float16.convert_float_to_float16(
        model, keep_io_types=True, disable_shape_infer=True,
        op_block_list=float16.DEFAULT_OP_BLOCK_LIST)
    # The converter can leave stale dtype annotations behind; let the runtime
    # re-infer intermediate types instead.
    del model.graph.value_info[:]
    return model


def strip_softmax_nan_guards(model: onnx.ModelProto) -> int:
    """Drop Where(IsNaN(softmax), 0, softmax) guards emitted for masked SDPA.

    Every masked attention row in this model has at least one real key, so the
    guard is a no-op that costs a full pass over the attention probabilities.
    """
    g = model.graph
    producer = {o: n for n in g.node for o in n.output}
    rename, dead = {}, set()
    for n in g.node:
        if n.op_type != "Where" or len(n.input) != 3:
            continue
        cond, _zero, val = n.input
        isnan = producer.get(cond)
        soft = producer.get(val)
        if (isnan is None or soft is None or isnan.op_type != "IsNaN"
                or soft.op_type != "Softmax" or isnan.input[0] != val):
            continue
        rename[n.output[0]] = val
        dead.update((id(n), id(isnan)))
    if not rename:
        return 0
    outputs = {o.name for o in g.output}
    assert not (outputs & rename.keys()), "guard feeds a graph output"
    keep = [n for n in g.node if id(n) not in dead]
    for n in keep:
        for i, name in enumerate(n.input):
            n.input[i] = rename.get(name, name)
    del g.node[:]
    g.node.extend(keep)
    return len(rename)


def isolate_graph_outputs(model: onnx.ModelProto) -> int:
    """Give graph outputs that other nodes also read a private Identity.

    With keep_io_types the fp16 converter casts the graph output tensor back
    to fp32 in place, so a node reading that same name (e.g. RVM's state,
    which is both returned and sliced into features) would see fp32.
    """
    g = model.graph
    outputs = {o.name for o in g.output}
    read = {i for n in g.node for i in n.input}
    shared = outputs & read
    for name in shared:
        internal = name + "__internal"
        for n in g.node:
            n.output[:] = [internal if o == name else o for o in n.output]
            n.input[:] = [internal if i == name else i for i in n.input]
        g.node.append(onnx.helper.make_node("Identity", [internal], [name], name=name + "__out"))
    return len(shared)


def quantize_nbits(src: Path, dst: Path, bits: int, block_size: int) -> None:
    from onnxruntime.quantization import matmul_nbits_quantizer as mnq

    model = onnx.load(str(src))
    # accuracy_level left unset: level 4 lets kernels quantize activations to
    # int8 on the fly (DP4A paths), which would defeat the fp32-activation variants.
    cfg = mnq.DefaultWeightOnlyQuantConfig(
        block_size=block_size, is_symmetric=True, accuracy_level=None,
        quant_format=mnq.QuantFormat.QOperator, bits=bits)
    q = mnq.MatMulNBitsQuantizer(model, algo_config=cfg)
    q.process()
    # ORT Web's MatMulNBits treats a *missing* accuracy_level as 4, and some
    # quantizer versions omit the attribute when it is 0; write it explicitly.
    for node in q.model.model.graph.node:
        if node.op_type == "MatMulNBits":
            attrs = [a for a in node.attribute if a.name != "accuracy_level"]
            del node.attribute[:]
            node.attribute.extend(attrs + [onnx.helper.make_attribute("accuracy_level", 0)])
    save(q.model.model, dst)


VARIANTS = {  # name: (weight bits or None, fp16 activations)
    "fp32": (None, False),
    "fp16": (None, True),
    "q8": (8, False),
    "q4": (4, False),
    "q4f16": (4, True),
    "q8f16": (8, True),
}


def fuse_transformer_ops(src: Path, dst: Path) -> dict:
    """ORT's transformer fusions that the WebGPU EP has kernels for: RMSNorm ->
    SimplifiedLayerNormalization, erf GELU -> Gelu, Add+LayerNorm ->
    SkipLayerNormalization. Attention and rotary fusion are left off: ORT's
    patterns do not match this model's windowed/gated attention or 3-axis RoPE."""
    from onnxruntime.transformers.fusion_options import FusionOptions
    from onnxruntime.transformers.optimizer import optimize_model

    opts = FusionOptions("bert")
    opts.enable_attention = False
    opts.enable_rotary_embeddings = False
    opts.enable_embed_layer_norm = False
    m = optimize_model(str(src), model_type="bert", num_heads=0, hidden_size=0,
                       optimization_options=opts, opt_level=0, use_gpu=True)
    save(m.model, dst)
    return m.get_fused_operator_statistics()


def process(models: Path, name: str, variant: str, block_size: int,
            fuse: bool = False, tag: str = "") -> dict:
    t0 = time.time()
    src = models / "raw" / f"{name}.onnx"
    comp_dir = models / name
    (comp_dir / "onnx").mkdir(parents=True, exist_ok=True)
    out = comp_dir / "onnx" / f"model_{variant}{tag}.onnx"
    data_name = f"model_{variant}{tag}.onnx_data"
    bits, fp16 = VARIANTS[variant]
    fused = {}
    with tempfile.TemporaryDirectory(dir=models) as td:
        if fuse:
            fpath = Path(td) / f"{name}_fused.onnx"
            fused = {k: v for k, v in fuse_transformer_ops(src, fpath).items() if v}
            src = fpath
        if bits:
            qpath = Path(td) / f"{name}_q.onnx"
            quantize_nbits(src, qpath, bits, block_size)
            src = qpath
        stage = Path(td) / f"{name}.onnx"
        ort_basic_optimize(src, stage)
        # Fresh, complete dtype annotations: the fp16 converter places its
        # casts from value_info, and ORT's saved annotations are partial.
        bare = onnx.load(str(stage), load_external_data=False)
        del bare.graph.value_info[:]
        onnx.save_model(bare, str(stage))
        onnx.shape_inference.infer_shapes_path(str(stage), str(stage))
        model = onnx.load(str(stage))
        guards = strip_softmax_nan_guards(model)
        isolate_graph_outputs(model)
        save(to_fp16(model) if fp16 else model, out, data_name)
    size = out.stat().st_size + (out.parent / data_name).stat().st_size
    ops = {}
    for n in onnx.load(str(out), load_external_data=False).graph.node:
        ops[n.op_type] = ops.get(n.op_type, 0) + 1
    info = {"file": f"{name}/onnx/{out.name}", "bytes": size, "fused": fused, "nodes": sum(ops.values()), "nan_guards_removed": guards, "seconds": round(time.time() - t0, 1),
            "MatMulNBits": ops.get("MatMulNBits", 0), "MatMul": ops.get("MatMul", 0),
            "Cast": ops.get("Cast", 0)}
    print(f"[{variant}] {name}: {size / 2**20:.1f} MiB, {info}")
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--variant", required=True, choices=tuple(VARIANTS))
    ap.add_argument("--components", nargs="*", default=list(COMPONENTS))
    ap.add_argument("--block_size", type=int, default=32)
    ap.add_argument("--fuse", action="store_true", help="apply ORT transformer fusions first")
    ap.add_argument("--tag", default="", help="suffix for the variant name, e.g. _fused")
    args = ap.parse_args()
    models = Path(args.models)
    key = args.variant + args.tag
    for name in args.components:
        info = process(models, name, args.variant, args.block_size, fuse=args.fuse, tag=args.tag)
        update_preset_config(models, corridorkey={"variants": {key: {name: info}}},
                             external_data={f"model_{key}.onnx": 1})


if __name__ == "__main__":
    main()
