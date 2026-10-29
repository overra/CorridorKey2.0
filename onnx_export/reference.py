#!/usr/bin/env python3
"""Reference PyTorch run of every CorridorKey component on one clip window.

Writes a golden .npz (inputs + per-component outputs) used by the ONNX parity
checks, and reports per-component timing, peak memory, bf16-vs-fp32 error and
activation ranges (fp16 overflow risk).

    python onnx_export/reference.py --height 512 --width 896 --frames 8
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from common import load_model, prepare_inputs, read_clip, synthetic_clip


@torch.no_grad()
def run_components(model, inp, *, autocast: bool):
    """Run foundations, RVM, stage 1 and stage 2 like the CUDA runtime does."""
    ctx = (torch.autocast("cuda", dtype=torch.bfloat16) if autocast
           else torch.autocast("cuda", enabled=False))
    f = model.foundations
    T = inp["rgb"].shape[1]
    H, W = inp["rgb"].shape[-2:]
    out, times = {}, {}

    def timed(name, fn):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with ctx:
            r = fn()
        torch.cuda.synchronize()
        times[name] = time.perf_counter() - t0
        return r

    ff = timed("cradio+moge", lambda: f(rgb_cradio=inp["rgb_cradio"], rgb_moge=inp["rgb_moge"]))
    out["cradio"] = ff["cradio"].float()
    out["moge"] = ff["moge"].float()

    def rvm_seq():
        feats, state = f.rvm(inp["rgb_rvm"], state=None)
        return feats, state
    rvm_feats, rvm_state = timed("rvm", rvm_seq)
    mean = f.rvm_mean.view(1, 1, -1, 1, 1).float()
    std = f.rvm_std.view(1, 1, -1, 1, 1).float().clamp(min=1e-6)
    out["rvm"] = ((rvm_feats.float() - mean) / std)[0]
    out["rvm_state"] = rvm_state.float()

    hq = torch.full((1, T), -1.0, device="cuda")
    hint = torch.zeros(1, T, 1, H, W, device="cuda")

    def s1():
        return model.stage1(
            inp["rgb"], hint,
            out["moge"].reshape(1, T, -1, H // 64, W // 64),
            cradio_feat=out["cradio"].reshape(1, T, -1, H // 64, W // 64),
            rvm_feat=out["rvm"].reshape(1, T, -1, H // 64, W // 64),
            t_stride=1.0, hint_quality=hq)
    s1o = timed("stage1", s1)
    out["baton"] = s1o["baton"][0].float()
    out["aux"] = s1o["aux"][0].float()

    def s2():
        logits = []
        for t in range(T):
            logits.append(model.stage2(inp["rgb"][:, t:t + 1], out["baton"][None, t:t + 1])[0, 0])
        return torch.stack(logits, 0)
    out["logits"] = timed("stage2", s2).float()
    return out, times


def activation_ranges(model, inp):
    """Max |activation| per module type (fp32 run) to flag fp16 overflow risk."""
    stats = defaultdict(float)
    hooks = []

    def hook(name):
        def fn(_m, _i, o):
            if isinstance(o, torch.Tensor) and o.is_floating_point():
                stats[name] = max(stats[name], float(o.detach().abs().max()))
        return fn

    for name, m in model.named_modules():
        if name and len(list(m.children())) == 0:
            hooks.append(m.register_forward_hook(hook(name)))
    try:
        run_components(model, inp, autocast=False)
    finally:
        for h in hooks:
            h.remove()
    return dict(stats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--width", type=int, default=896)
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--clip", default=None, help="Optional video; synthetic clip if omitted.")
    ap.add_argument("--out", default="onnx_export/out/reference")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = (read_clip(args.clip, args.frames, args.height, args.width) if args.clip
              else synthetic_clip(args.frames, args.height, args.width))
    model = load_model("cuda")
    inp = prepare_inputs(frames)

    run_components(model, inp, autocast=True)  # warm-up
    torch.cuda.reset_peak_memory_stats()
    ref32, t32 = run_components(model, inp, autocast=False)
    peak32 = torch.cuda.max_memory_allocated() / 2**30
    torch.cuda.reset_peak_memory_stats()
    ref16, t16 = run_components(model, inp, autocast=True)
    peak16 = torch.cuda.max_memory_allocated() / 2**30

    report = {"shape": [args.frames, args.height, args.width],
              "time_fp32_s": t32, "time_bf16_s": t16,
              "peak_gib_fp32": peak32, "peak_gib_bf16": peak16, "bf16_vs_fp32": {}}
    for k in ref32:
        a, b = ref32[k], ref16[k]
        report["bf16_vs_fp32"][k] = {
            "max_abs": float((a - b).abs().max()),
            "mean_abs": float((a - b).abs().mean()),
            "ref_absmax": float(a.abs().max()),
        }
    a32 = ref32["logits"][:, 0].clamp(0, 1)
    a16 = ref16["logits"][:, 0].clamp(0, 1)
    report["alpha_bf16_vs_fp32"] = {"max_abs": float((a32 - a16).abs().max()),
                                    "mean_abs": float((a32 - a16).abs().mean())}

    ranges = activation_ranges(model, inp)
    top = sorted(ranges.items(), key=lambda kv: -kv[1])[:25]
    report["activation_absmax_top25"] = top
    report["activations_over_fp16_max"] = [k for k, v in ranges.items() if v > 65504]

    np.savez(out_dir / "golden.npz",
             frames=frames,
             **{f"in_{k}": v.float().cpu().numpy() for k, v in inp.items()},
             **{f"ref32_{k}": v.cpu().numpy() for k, v in ref32.items()},
             **{f"ref16_{k}": v.cpu().numpy() for k, v in ref16.items()})
    (out_dir / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "activation_absmax_top25"}, indent=1))
    print("top activations:")
    for k, v in top:
        print(f"  {v:12.1f}  {k}")


if __name__ == "__main__":
    main()
