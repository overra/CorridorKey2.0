"""Shared helpers for the ONNX / browser export of CorridorKey v2.

Everything here mirrors the CUDA runtime (`corridorkey/runtime.py`,
`corridorkey/video_io.py`) so exported graphs can be checked against the
reference PyTorch path on identical inputs.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from corridorkey import color_utils as cu  # noqa: E402

# References must be true fp32: on Ampere and newer, PyTorch runs cuDNN
# convolutions in TF32 by default (10-bit mantissa inputs).
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

SHAPE_QUANTUM = 128


def load_model(device: str = "cuda"):
    """Load the full CorridorKey v2 runtime model (foundations + stages)."""
    from corridorkey.runtime import RuntimePaths, load_model as _load

    paths = RuntimePaths.from_package(REPO_ROOT)
    model = _load(paths, device=device)
    # The runtime builds stage 1/2 on CPU and moves them per phase.
    model.to(device).eval()
    return model


def read_preset(models: Path) -> dict:
    """The preset's `corridorkey` block from <models>/config.json."""
    import json
    return json.loads((Path(models) / "config.json").read_text())["corridorkey"]


def update_preset_config(models: Path, *, corridorkey: dict | None = None,
                         external_data: dict | None = None) -> None:
    """Merge into <models>/config.json, the one transformers.js reads for every
    component (they load with subfolder='<component>/onnx')."""
    import json
    path = Path(models) / "config.json"
    cfg = json.loads(path.read_text()) if path.exists() else {
        "model_type": "corridorkey_v2_web",
        "corridorkey": {},
        "transformers.js_config": {"use_external_data_format": {}},
    }
    if corridorkey:
        _deep_update(cfg["corridorkey"], corridorkey)
    if external_data:
        cfg["transformers.js_config"]["use_external_data_format"].update(external_data)
    # Browsers cache model files by URL without revalidating; the runtime keys
    # its cache on this id, so every rebuild invalidates stale copies.
    import uuid
    cfg["corridorkey"]["build_id"] = uuid.uuid4().hex[:12]
    path.write_text(json.dumps(cfg, indent=1))


def _deep_update(dst: dict, src: dict) -> None:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v


# ---------------------------------------------------------------------------
# Test clips
# ---------------------------------------------------------------------------


def synthetic_clip(T: int, H: int, W: int, seed: int = 0) -> np.ndarray:
    """A deterministic green-screen-like clip, (T, H, W, 3) sRGB float32 in [0, 1].

    Uneven green backing with falloff and noise, a moving subject with a
    soft-edged body, skin/cloth colors, green spill on the edges, and thin
    hair-like strands so fine alpha detail is exercised.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.meshgrid(np.linspace(0, 1, H, dtype=np.float32),
                         np.linspace(0, 1, W, dtype=np.float32), indexing="ij")
    strands = [(rng.uniform(-0.25, 0.25), rng.uniform(0.02, 0.06),
                rng.uniform(0.0, 2 * np.pi)) for _ in range(40)]
    frames = []
    for t in range(T):
        phase = t / max(T, 1)
        falloff = 1.0 - 0.35 * ((xx - 0.5) ** 2 + (yy - 0.45) ** 2)
        bg = np.stack([0.10 * falloff, 0.62 * falloff, 0.18 * falloff], -1)
        bg += rng.normal(0, 0.012, bg.shape).astype(np.float32)

        cx = 0.45 + 0.1 * np.sin(2 * np.pi * phase)
        cy = 0.42
        head = ((xx - cx) / 0.09) ** 2 + ((yy - cy) / 0.13) ** 2
        torso = ((xx - cx) / 0.2) ** 2 + ((yy - (cy + 0.42)) / 0.3) ** 2
        body = np.clip(1.6 - np.minimum(head, torso) * 1.2, 0, 1)

        hair = np.zeros_like(xx)
        for dx, amp, ph in strands:
            x0 = cx + dx * 0.5
            path = x0 + amp * np.sin(12 * yy + ph + 3 * phase)
            d = np.abs(xx - path)
            m = (yy > cy - 0.2) & (yy < cy + 0.05 + 0.1 * abs(dx))
            hair = np.maximum(hair, np.where(m, np.exp(-(d / 0.0025) ** 2), 0))
        alpha = np.clip(np.maximum(body, 0.85 * hair), 0, 1)[..., None]

        skin = np.array([0.86, 0.62, 0.50], np.float32)
        cloth = np.array([0.20, 0.25, 0.55], np.float32)
        hair_c = np.array([0.25, 0.16, 0.10], np.float32)
        fg = np.where((head < 1.0)[..., None], skin, cloth)
        fg = np.where((hair > 0.3)[..., None], hair_c, fg)
        fg = fg * (0.75 + 0.25 * (1 - yy))[..., None]
        spill = (1 - alpha) * alpha * 4
        fg = fg + np.concatenate([0 * spill, 0.25 * spill, 0 * spill], -1)
        frame = fg * alpha + bg * (1 - alpha)
        frames.append(np.clip(frame, 0, 1).astype(np.float32))
    return np.stack(frames, 0)


def read_clip(path: str, T: int, H: int, W: int, start: int = 0) -> np.ndarray:
    """Read T frames from a video, resized to cover (H, W) and center-cropped."""
    import cv2

    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames = []
    while len(frames) < T:
        ok, bgr = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        h, w = rgb.shape[:2]
        s = max(H / h, W / w)
        nh, nw = max(H, round(h * s)), max(W, round(w * s))
        rgb = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
        y0, x0 = (nh - H) // 2, (nw - W) // 2
        frames.append(rgb[y0:y0 + H, x0:x0 + W])
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames read from {path}")
    return np.stack(frames, 0)


# ---------------------------------------------------------------------------
# Input preparation (mirrors runtime._build_foundation_inputs and
# runtime.precompute_rvm_to_memmap)
# ---------------------------------------------------------------------------


def prepare_inputs(frames_srgb: np.ndarray, device: str = "cuda") -> dict:
    """frames_srgb: (T, H, W, 3) display-coded RGB in [0, 1], H/W multiples of 128."""
    T, H, W, _ = frames_srgb.shape
    assert H % SHAPE_QUANTUM == 0 and W % SHAPE_QUANTUM == 0, (H, W)
    display = torch.from_numpy(frames_srgb).permute(0, 3, 1, 2).contiguous()
    display = display.to(device=device, dtype=torch.float32)
    linear = cu.srgb_to_linear(display)
    model_rgb = cu.linear_to_asinh(linear)
    rgb_cradio = F.interpolate(display, size=(H // 4, W // 4), mode="area")
    rgb_moge = F.interpolate(display, size=((H * 7) // 32, (W * 7) // 32), mode="area")
    rgb_rvm = F.interpolate(display, size=(H // 4, W // 4), mode="bilinear",
                            align_corners=False).clamp(0.0, 1.0)
    return {
        "display": display,                 # (T, 3, H, W)
        "rgb": model_rgb.unsqueeze(0),      # (1, T, 3, H, W) asinh encoded
        "rgb_cradio": rgb_cradio,           # (T, 3, H/4, W/4)
        "rgb_moge": rgb_moge,               # (T, 3, 7H/32, 7W/32)
        "rgb_rvm": rgb_rvm.unsqueeze(0),    # (1, T, 3, H/4, W/4)
    }


def decode_outputs(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Stage-2 logits (..., 4, H, W) -> (alpha [0,1], fg sRGB [0,1])."""
    alpha = logits[..., 0:1, :, :].float().clamp(0.0, 1.0)
    fg_lin = cu.asinh_to_linear(logits[..., 1:4, :, :].float()).clamp(min=0.0)
    fg = cu.linear_to_srgb(fg_lin).clamp(0.0, 1.0)
    return alpha, fg


def save_png(path: str | Path, img: np.ndarray) -> None:
    """Save (H, W) or (H, W, 3) float [0, 1] as 8-bit PNG."""
    import cv2

    img = np.clip(img, 0.0, 1.0)
    img = (img * 255.0 + 0.5).astype(np.uint8)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), img)
