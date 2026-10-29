#!/usr/bin/env python3
"""Export CorridorKey v2 components to static-shape fp32 ONNX graphs.

    python onnx_export/export.py --height 512 --width 896 --frames 4 \
        --out web/models/896x512

Writes <out>/raw/<component>.onnx (+ external data) and the preset dims into
<out>/config.json.
`optimize.py` turns these into the browser variants.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from common import load_model, update_preset_config  # also sets up sys.path
import contrib_ops
import patches
from wrappers import CradioGraph, MogeGraph, RvmStepGraph, Stage1Graph, Stage2Graph

COMPONENTS = ("cradio", "moge", "rvm", "stage1", "stage2")


def build(model, name: str, H: int, W: int, T: int):
    """Return (module, example_args, input_names, output_names)."""
    f = model.foundations
    h64, w64 = H // 64, W // 64
    frame = torch.rand(1, 3, H, W)
    if name == "cradio":
        return CradioGraph(f, (H, W)), (frame,), ["frames"], ["cradio"]
    if name == "moge":
        return MogeGraph(f, (H, W)), (frame,), ["frames"], ["moge"]
    if name == "rvm":
        state = torch.zeros(1, 1 + h64 * w64, f.rvm.embed_dim)
        return RvmStepGraph(f, (H, W)), (frame, state), ["frame", "state"], ["rvm", "state_out"]
    if name == "stage1":
        s1 = model.stage1
        args = (
            *(torch.rand(1, 3, H, W) for _ in range(T)),
            *(torch.zeros(1, 1, H, W) for _ in range(T)),
            torch.full((1, T), -1.0),
            *(torch.randn(1, s1.cradio_dim, h64, w64) for _ in range(T)),
            *(torch.randn(1, s1.moge_dim, h64, w64) for _ in range(T)),
            *(torch.randn(1, s1.rvm_dim, h64, w64) for _ in range(T)),
        )
        ins, outs = Stage1Graph.io_names(T)
        return Stage1Graph(s1, (H, W), T), args, ins, outs
    if name == "stage2":
        baton = torch.randn(1, model.stage1.d_16, H // 16, W // 16)
        return (Stage2Graph(model.stage2), (frame, baton, baton.clone(), torch.zeros(1)),
                ["frame", "baton_a", "baton_b", "blend"], ["alpha", "fg"])
    raise ValueError(name)


def export_one(model, name, H, W, T, raw_dir: Path, opset: int):
    mod, args, in_names, out_names = build(model, name, H, W, T)
    mod = mod.eval()
    path = raw_dir / f"{name}.onnx"
    t0 = time.time()
    with torch.no_grad():
        torch.onnx.export(
            mod, args, str(path),
            input_names=in_names, output_names=out_names,
            opset_version=opset, dynamo=True, external_data=True,
            optimize=True, custom_translation_table=contrib_ops.CUSTOM_TRANSLATIONS,
        )
    print(f"[export] {name}: {path} in {time.time() - t0:.1f}s")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--width", type=int, default=896)
    ap.add_argument("--frames", type=int, default=4, help="Stage-1 window length T.")
    ap.add_argument("--components", nargs="*", default=list(COMPONENTS))
    ap.add_argument("--opset", type=int, default=18)
    ap.add_argument("--out", required=True)
    ap.add_argument("--plain", action="store_true",
                    help="skip the WebGPU rewrites (conv/deconv as MatMul, RotaryEmbedding)")
    args = ap.parse_args()

    H, W, T = args.height, args.width, args.frames
    assert H % 128 == 0 and W % 128 == 0, "H and W must be multiples of 128"
    out_dir = Path(args.out)
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    patches.install()
    model = load_model("cpu")
    if not args.plain:
        print("conv rewrites:", patches.install_conv_rewrites(model))
        contrib_ops.install_rotary()
    for name in args.components:
        export_one(model, name, H, W, T, raw_dir, args.opset)
    update_preset_config(out_dir, corridorkey={
        "height": H, "width": W, "frames": T, "opset": args.opset,
        "cradio_dim": model.stage1.cradio_dim, "moge_dim": model.stage1.moge_dim,
        "rvm_dim": model.stage1.rvm_dim, "baton_dim": model.stage1.d_16,
        "rvm_tokens": 1 + (H // 64) * (W // 64)})
    # The demo page cannot list directories; keep an index of presets.
    index = out_dir.parent / "presets.json"
    presets = set(json.loads(index.read_text())) if index.exists() else set()
    def by_area(name: str) -> int:
        w, _, h = name.partition("x")
        return int(w) * int(h) if w.isdigit() and h.isdigit() else 0

    index.write_text(json.dumps(sorted(presets | {out_dir.name}, key=by_area)))


if __name__ == "__main__":
    main()
