#!/usr/bin/env python3
"""Reference mattes for a whole clip from the real CUDA runtime.

Runs corridorkey.runtime's own Hann-window path (precompute_rvm_to_memmap +
hann_stage1_stream with carry hints, then stage 2 as write_prediction_batches
does) on a frame directory, so the browser's temporal logic (corridorkey.js
CorridorKey.process) can be checked frame by frame against production. The
runtime runs in bf16, so expect bf16-level differences everywhere; a
windowing or carry bug shows up as frame-specific outliers.

    python onnx_export/seq_reference.py --models web/models/896x512 --clip clip.mp4 --frames 14
    python onnx_export/seq_reference.py ... --name golden_seq_hint --hint-from golden_seq

--hint-from uses frame 0 of an earlier reference's matte (as an 8-bit PNG, the
runtime's --hint_first_frame input) to exercise the first-frame hint path.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch

from common import load_model, read_clip, read_preset


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--clip", required=True)
    ap.add_argument("--frames", type=int, default=14)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--name", default="golden_seq")
    ap.add_argument("--stride", type=int, default=None, help="Hann stride (default T/2)")
    ap.add_argument("--hint-from", default=None, help="golden folder whose frame-0 alpha is the hint")
    ap.add_argument("--hint-quality", type=float, default=0.95)
    args = ap.parse_args()

    from corridorkey.runtime import hann_stage1_stream, precompute_rvm_to_memmap
    from corridorkey.video_io import (
        SequentialFrameDirWindowReader, load_initial_hint_frame, probe_frame_dir_meta)

    models = Path(args.models)
    preset = read_preset(models)
    H, W, T = preset["height"], preset["width"], preset["frames"]
    stride = args.stride or max(1, T // 2)
    frames = read_clip(args.clip, args.frames, H, W, start=args.start)
    frames8 = np.clip(np.round(frames * 255.0), 0, 255).astype(np.uint8)

    model = load_model("cuda")
    out = models / args.name
    out.mkdir(exist_ok=True)
    alphas = {}
    with tempfile.TemporaryDirectory() as td:
        frame_dir = Path(td) / "frames"
        frame_dir.mkdir()
        for i, f in enumerate(frames8):
            cv2.imwrite(str(frame_dir / f"{i:05d}.png"), cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        meta = probe_frame_dir_meta(str(frame_dir), num_frames=len(frames8), shape_quantum=128,
                                    fps=24.0, exr_linear=False, model_scale=1.0)
        rvm_path = Path(td) / "rvm.f16"
        precompute_rvm_to_memmap(model, str(frame_dir), meta, rvm_path, chunk=T, reader_kind="frame_dir")
        initial_hint, hint8 = None, None
        if args.hint_from:
            src = models / args.hint_from
            shape = json.loads((src / "manifest.json").read_text())["alpha"]
            alpha0 = np.fromfile(src / "alpha.bin", dtype="<f4").reshape(shape)[0, 0]
            hint8 = np.clip(np.round(alpha0 * 255.0), 0, 255).astype(np.uint8)
            hint_png = Path(td) / "hint.png"
            cv2.imwrite(str(hint_png), hint8)
            initial_hint = load_initial_hint_frame(str(hint_png), meta)

        def on_baton_frames(items):
            for k, baton, model_frame in items:
                rgb = model_frame[None, None].to("cuda")
                b = torch.from_numpy(baton)[None, None].to("cuda").float()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model.stage2(rgb, b)
                alphas[k] = logits[0, 0, 0:1].float().clamp(0, 1).cpu().numpy()

        hann_stage1_stream(
            model, str(frame_dir), meta, reader_cls=SequentialFrameDirWindowReader,
            reader_kind="frame_dir", chunk=T, stride=stride, baton_path=Path(td) / "baton.f16",
            initial_hint=initial_hint, initial_hint_quality=args.hint_quality, full_hint_video_path=None,
            full_hint_quality=0.95, rvm_path=rvm_path, foundation_paths=None,
            carry_hint_from_aux=True, carry_hint_quality=0.95, on_baton_frames=on_baton_frames)

    alpha = np.stack([alphas[k] for k in range(len(frames8))])
    tensors = {"frames": (frames8.astype(np.float32) / 255.0).transpose(0, 3, 1, 2), "alpha": alpha}
    manifest = {"stride": stride}
    if hint8 is not None:
        tensors["hint"] = (hint8.astype(np.float32) / 255.0)[None]
        manifest["hint_quality"] = args.hint_quality
    for k, v in tensors.items():
        v = np.ascontiguousarray(v, dtype="<f4")
        v.tofile(out / f"{k}.bin")
        manifest[k] = list(v.shape)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"wrote {out}: {manifest}")


if __name__ == "__main__":
    main()
