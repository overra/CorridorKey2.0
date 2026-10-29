#!/usr/bin/env python3
"""Check exported ONNX graphs against the fp32 PyTorch wrappers.

Runs one Hann window (T frames, no hint, RVM from a zero state) through every
graph with onnxruntime and compares to the same wrappers in PyTorch fp32,
per graph (reference inputs) and chained end to end. The production bf16
error on the same inputs is reported as the bar to beat.

    python onnx_export/parity.py --models web/models/896x512 --variant fp16
    python onnx_export/parity.py --models web/models/896x512 --write-golden

--write-golden stores the input frames and fp32 reference outputs as raw
little-endian float32 .bin files under <models>/golden/ for the browser
benchmark (web/bench.html).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from common import load_model, read_clip, read_preset, synthetic_clip
from wrappers import CradioGraph, MogeGraph, RvmStepGraph, Stage1Graph, Stage2Graph


def model_path(models: Path, name: str, variant: str) -> Path:
    if variant == "fp32":
        return models / "raw" / f"{name}.onnx"
    return models / name / "onnx" / f"model_{variant}.onnx"


def session(path: Path, provider: str):
    so = ort.SessionOptions()
    so.log_severity_level = 3
    providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                 if provider == "cuda" else ["CPUExecutionProvider"])
    return ort.InferenceSession(str(path), so, providers=providers)


def run(sess, feeds: dict) -> list[np.ndarray]:
    types = {i.name: i.type for i in sess.get_inputs()}
    cast = {k: v.astype(np.float16 if types[k] == "tensor(float16)" else np.float32)
            for k, v in feeds.items()}
    return [o.astype(np.float32) for o in sess.run(None, cast)]


def err(a, b) -> dict:
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    d = np.abs(a - b)
    return {"max_abs": float(d.max()), "mean_abs": float(d.mean()),
            "over1": float((d > 1 / 255).mean()), "over4": float((d > 4 / 255).mean()),
            "ref_absmax": float(np.abs(b).max()), "nan": bool(np.isnan(a).any())}


@torch.no_grad()
def torch_ref(model, H, W, T, frames, autocast: bool) -> dict:
    """One window through the PyTorch wrappers (fp32 or bf16 autocast)."""
    dev = "cuda"
    ctx = (torch.autocast("cuda", dtype=torch.bfloat16) if autocast
           else torch.autocast("cuda", enabled=False))
    f = model.foundations
    disp = torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous().to(dev)
    out = {}
    with ctx:
        out["cradio"] = CradioGraph(f, (H, W)).to(dev).eval()(disp).float()
        out["moge"] = MogeGraph(f, (H, W)).to(dev).eval()(disp).float()
        rv = RvmStepGraph(f, (H, W)).to(dev).eval()
        state = torch.zeros(1, 1 + (H // 64) * (W // 64), f.rvm.embed_dim, device=dev)
        feats = []
        for t in range(T):
            ft, state = rv(disp[t:t + 1], state)
            feats.append(ft[0].float())
            state = state.float()
        out["rvm"] = torch.stack(feats)
        out["rvm_state"] = state
        s1 = Stage1Graph(model.stage1, (H, W), T).to(dev).eval()
        args = (*disp.split(1), *torch.zeros(T, 1, 1, H, W, device=dev),
                torch.full((1, T), -1.0, device=dev),
                *out["cradio"].split(1), *out["moge"].split(1), *out["rvm"].split(1))
        res = s1(*args)
        out["baton"] = torch.cat(res[:T]).float()
        out["carry"] = torch.cat(res[T:]).float()
        s2 = Stage2Graph(model.stage2).to(dev).eval()
        zero = torch.zeros(1, device=dev)
        alphas, fgs = [], []
        for t in range(T):
            b = out["baton"][t:t + 1]
            a, fg = s2(disp[t:t + 1], b, b, zero)
            alphas.append(a[0].float())
            fgs.append(fg[0].float())
        out["alpha"], out["fg"] = torch.stack(alphas), torch.stack(fgs)
    return {k: v.cpu().numpy() for k, v in out.items()}


def onnx_pipeline(models: Path, variant: str, provider: str, H, W, T, frames, ref) -> dict:
    """Each graph on reference inputs (isolated error) and chained (end to end)."""
    disp = frames.transpose(0, 3, 1, 2).astype(np.float32)
    res, ch = {}, {}
    s = session(model_path(models, "cradio", variant), provider)
    ch["cradio"] = np.concatenate([run(s, {"frames": disp[t:t + 1]})[0] for t in range(T)])
    res["cradio"] = err(ch["cradio"], ref["cradio"])
    s = session(model_path(models, "moge", variant), provider)
    ch["moge"] = np.concatenate([run(s, {"frames": disp[t:t + 1]})[0] for t in range(T)])
    res["moge"] = err(ch["moge"], ref["moge"])
    s = session(model_path(models, "rvm", variant), provider)
    state = np.zeros((1, 1 + (H // 64) * (W // 64), 384), np.float32)
    feats = []
    for t in range(T):
        ft, state = run(s, {"frame": disp[t:t + 1], "state": state})
        feats.append(ft)
    ch["rvm"] = np.concatenate(feats)
    res["rvm"] = err(ch["rvm"], ref["rvm"])
    res["rvm_state"] = err(state, ref["rvm_state"])

    s = session(model_path(models, "stage1", variant), provider)

    def s1_feeds(c, m, r):
        feeds = {"hint_quality": np.full((1, T), -1.0, np.float32)}
        for t in range(T):
            feeds[f"frame_{t}"] = disp[t:t + 1]
            feeds[f"hint_{t}"] = np.zeros((1, 1, H, W), np.float32)
            feeds[f"cradio_{t}"] = c[t:t + 1]
            feeds[f"moge_{t}"] = m[t:t + 1]
            feeds[f"rvm_{t}"] = r[t:t + 1]
        return feeds

    iso = run(s, s1_feeds(ref["cradio"], ref["moge"], ref["rvm"]))
    res["baton(isolated)"] = err(np.concatenate(iso[:T]), ref["baton"])
    out = run(s, s1_feeds(ch["cradio"], ch["moge"], ch["rvm"]))
    baton, carry = np.concatenate(out[:T]), np.concatenate(out[T:])
    res["baton(chained)"] = err(baton, ref["baton"])
    res["carry(chained)"] = err(carry, ref["carry"])

    s = session(model_path(models, "stage2", variant), provider)
    zero = np.zeros(1, np.float32)

    def s2(b):
        return run(s, {"frame": disp[t:t + 1], "baton_a": b, "baton_b": b, "blend": zero})

    a_iso = []
    for t in range(T):
        a_iso.append(s2(ref["baton"][t:t + 1])[0])
    res["alpha(isolated)"] = err(np.concatenate(a_iso), ref["alpha"])
    outs = []
    for t in range(T):
        outs.append(s2(baton[t:t + 1]))
    res["alpha(end-to-end)"] = err(np.concatenate([o[0] for o in outs]), ref["alpha"])
    res["fg(end-to-end)"] = err(np.concatenate([o[1] for o in outs]), ref["fg"])
    return res


def write_golden(models: Path, frames: np.ndarray, ref: dict, name: str = "golden") -> None:
    out = models / name
    out.mkdir(exist_ok=True)
    tensors = {"frames": frames.transpose(0, 3, 1, 2), **ref}
    manifest = {}
    for k, v in tensors.items():
        v = np.ascontiguousarray(v, dtype="<f4")
        v.tofile(out / f"{k}.bin")
        manifest[k] = list(v.shape)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"golden written to {out}: {manifest}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--variant", default="fp32", help="fp32 (raw export), fp16, q8f16, q4f16")
    ap.add_argument("--provider", default="cuda", choices=("cuda", "cpu"))
    ap.add_argument("--clip", default=None)
    ap.add_argument("--write-golden", action="store_true")
    ap.add_argument("--golden-name", default="golden", help="folder under --models, e.g. golden_real")
    ap.add_argument("--start", type=int, default=0, help="first frame of --clip")
    args = ap.parse_args()

    models = Path(args.models)
    preset = read_preset(models)
    H, W, T = preset["height"], preset["width"], preset["frames"]
    frames = read_clip(args.clip, T, H, W, start=args.start) if args.clip else synthetic_clip(T, H, W)
    model = load_model("cuda")
    ref = torch_ref(model, H, W, T, frames, autocast=False)
    if args.write_golden:
        write_golden(models, frames, ref, args.golden_name)
        return
    bf16 = torch_ref(model, H, W, T, frames, autocast=True)
    del model
    torch.cuda.empty_cache()

    res = onnx_pipeline(models, args.variant, args.provider, H, W, T, frames, ref)
    res["bf16 production: alpha"] = err(bf16["alpha"], ref["alpha"])
    res["bf16 production: baton"] = err(bf16["baton"], ref["baton"])
    for k, v in res.items():
        print(f"{k:26s} max_abs={v['max_abs']:.3e} mean_abs={v['mean_abs']:.3e} "
              f">1/255={v['over1']:.2e} >4/255={v['over4']:.2e} "
              f"ref_absmax={v['ref_absmax']:.2f}{'  NaN!' if v['nan'] else ''}")
    (models / f"parity_{args.variant}_{args.provider}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
