#!/usr/bin/env python3
"""Tiny shape-agnostic ONNX graph for the browser's running baton average:

    out = a + t * (b - a)

The browser accumulates a frame's stage-1 batons from overlapping Hann windows
as a running weighted mean, t = w_new / (w_acc + w_new), keeping the tensors on
the GPU. Stage 2 performs the final blend itself; this graph covers frames
seen by three or more windows (the end-aligned last window).

    python onnx_export/blend_graph.py --models web/models/896x512
"""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnx import TensorProto, helper


def build() -> onnx.ModelProto:
    dims = ["n", "c", "h", "w"]
    a = helper.make_tensor_value_info("a", TensorProto.FLOAT, dims)
    b = helper.make_tensor_value_info("b", TensorProto.FLOAT, dims)
    t = helper.make_tensor_value_info("t", TensorProto.FLOAT, [1])
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT, dims)
    nodes = [
        helper.make_node("Sub", ["b", "a"], ["d"]),
        helper.make_node("Mul", ["d", "t"], ["td"]),
        helper.make_node("Add", ["a", "td"], ["out"]),
    ]
    graph = helper.make_graph(nodes, "baton_blend", [a, b, t], [out])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 10  # onnx>=1.20 stamps IR 14; ORT Web accepts <= 13
    onnx.checker.check_model(model)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True, help="preset folder")
    args = ap.parse_args()
    out = Path(args.models) / "blend" / "onnx"
    out.mkdir(parents=True, exist_ok=True)
    # Not model_fp32.onnx: config.json's external-data entries are keyed by file
    # name across every subfolder, and the fp32 variants of the big graphs use one.
    onnx.save_model(build(), str(out / "model.onnx"))
    print(f"wrote {out / 'model.onnx'}")


if __name__ == "__main__":
    main()
