"""Export-friendly wrappers around the CorridorKey v2 components.

Every graph takes the same input: the padded display-coded sRGB frame(s) in
[0, 1] at model resolution. Resizing, color encoding and feature
normalization are folded into the graphs, so the browser uploads one frame
tensor and chains GPU-resident outputs between sessions.

Graphs (static shapes, one set per resolution preset):
    cradio : frames (N,3,H,W)                       -> feat (N,1152,H/64,W/64)
    moge   : frames (N,3,H,W)                       -> feat (N,768,H/64,W/64)
    rvm    : frame (1,3,H,W), state (1,1+n,384)     -> feat (1,384,H/64,W/64), state
    stage1 : per-frame frame_i, hint_i, cradio_i, moge_i, rvm_i + hint_quality (1,T)
                                                    -> baton_i (1,512,H/16,W/16), carry_i (1,1,H,W)
    stage2 : frame (1,3,H,W), baton_a, baton_b (1,512,H/16,W/16), blend (1,)
                                                    -> alpha (1,1,H,W), fg (1,3,H,W)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from corridorkey import color_utils as cu

ASINH_DENOM = float(np.arcsinh(cu.HDR_ASINH_REF / cu.HDR_ASINH_KNEE))


def adaptive_avg_matrix(n_in: int, n_out: int) -> torch.Tensor:
    """(n_out, n_in) matrix equal to 1-D adaptive average pooling.

    F.interpolate(mode='area') is adaptive_avg_pool2d, which is separable, so
    two of these reproduce it exactly with plain MatMuls.
    """
    m = torch.zeros(n_out, n_in, dtype=torch.float64)
    for i in range(n_out):
        s = (i * n_in) // n_out
        e = -((-(i + 1) * n_in) // n_out)
        m[i, s:e] = 1.0 / (e - s)
    return m.float()


def bilinear_matrix(n_in: int, n_out: int) -> torch.Tensor:
    """(n_out, n_in) matrix equal to 1-D bilinear resize, align_corners=False, no antialias."""
    m = torch.zeros(n_out, n_in, dtype=torch.float64)
    scale = n_in / n_out
    for i in range(n_out):
        src = max((i + 0.5) * scale - 0.5, 0.0)
        i0 = min(int(np.floor(src)), n_in - 1)
        i1 = min(i0 + 1, n_in - 1)
        w1 = src - i0
        m[i, i0] += 1.0 - w1
        m[i, i1] += w1
    return m.float()


class SeparableResize(nn.Module):
    """Exact separable resize as two MatMuls with constant matrices."""

    def __init__(self, mat_h: torch.Tensor, mat_w: torch.Tensor):
        super().__init__()
        self.register_buffer("mh", mat_h)               # (h_out, h_in)
        self.register_buffer("mwt", mat_w.t().contiguous())  # (w_in, w_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(torch.matmul(self.mh, x), self.mwt)


def srgb_to_model_rgb(x: torch.Tensor) -> torch.Tensor:
    """Display sRGB [0,1] -> signed-asinh encoded scene-linear (model contract).

    asinh(z) is written as log(z + sqrt(z^2 + 1)), valid for z >= 0, so the
    graph needs no Asinh op.
    """
    x = x.clamp(0.0, 1.0)
    lin = torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
    z = lin / cu.HDR_ASINH_KNEE
    return torch.log(z + torch.sqrt(z * z + 1.0)) / ASINH_DENOM


def model_fg_to_srgb(y: torch.Tensor) -> torch.Tensor:
    """Asinh-encoded FG -> display sRGB [0,1] (runtime preview path)."""
    v = y * ASINH_DENOM
    lin = (0.5 * cu.HDR_ASINH_KNEE) * (torch.exp(v) - torch.exp(-v))
    lin = lin.clamp(min=0.0)
    srgb = torch.where(lin <= 0.0031308, lin * 12.92,
                       1.055 * lin.clamp(min=0.0031308) ** (1.0 / 2.4) - 0.055)
    return srgb.clamp(0.0, 1.0)


class CradioGraph(nn.Module):
    def __init__(self, foundations, hw):
        super().__init__()
        H, W = hw
        self.cradio = foundations.cradio
        self.register_buffer("mean", foundations.cradio_mean.view(1, -1, 1, 1).float())
        self.register_buffer("inv_std", 1.0 / foundations.cradio_std.view(1, -1, 1, 1).float().clamp(min=1e-6))
        self.hw = (H // 4, W // 4)

    def forward(self, frames):
        x = F.avg_pool2d(frames, 4)          # exact 'area' for an integer factor
        feat = self.cradio(x).features       # HF wrapper returns NLC tokens
        h, w = self.hw[0] // 16, self.hw[1] // 16
        feat = feat.transpose(1, 2).reshape(feat.shape[0], feat.shape[2], h, w)
        return (feat.float() - self.mean) * self.inv_std


class MogeGraph(nn.Module):
    def __init__(self, foundations, hw):
        super().__init__()
        H, W = hw
        hm, wm = (H * 7) // 32, (W * 7) // 32
        self.resize = SeparableResize(adaptive_avg_matrix(H, hm), adaptive_avg_matrix(W, wm))
        self.moge = foundations.moge
        self.base = (hm // 14, wm // 14)
        self.register_buffer("mean", foundations.moge_mean.view(1, -1, 1, 1).float())
        self.register_buffer("inv_std", 1.0 / foundations.moge_std.view(1, -1, 1, 1).float().clamp(min=1e-6))

    def forward(self, frames):
        x = self.resize(frames)
        # DINOv2Encoder.forward without its first step: an antialiased bilinear
        # resize to (rows*14, cols*14), which is the size x already has and
        # so is an identity (ORT WebGPU has no antialiased Resize).
        enc = self.moge.encoder
        rows, cols = self.base
        x = (x - enc.image_mean) / enc.image_std
        layers = enc.backbone.get_intermediate_layers(x, n=enc.intermediate_layers, return_class_token=True)
        feat = torch.stack([
            proj(f.permute(0, 2, 1).unflatten(2, (rows, cols)).contiguous())
            for proj, (f, _cls) in zip(enc.output_projections, layers)
        ], dim=1).sum(dim=1)
        return (feat.float() - self.mean) * self.inv_std


class RvmStepGraph(nn.Module):
    """One recurrent RVM step. Initial state is all zeros (runtime default)."""

    def __init__(self, foundations, hw):
        super().__init__()
        H, W = hw
        self.resize = SeparableResize(bilinear_matrix(H, H // 4), bilinear_matrix(W, W // 4))
        self.rvm = foundations.rvm
        self.register_buffer("mean", foundations.rvm_mean.view(1, -1, 1, 1).float())
        self.register_buffer("inv_std", 1.0 / foundations.rvm_std.view(1, -1, 1, 1).float().clamp(min=1e-6))

    def forward(self, frame, state):
        x = self.resize(frame).clamp(0.0, 1.0)
        # Same as RVMVideoSiamMAE.forward for T=1, without the per-step
        # scatter into a preallocated feature buffer.
        tokens = self.rvm._run_encoder(self.rvm._tokenize(x))
        new_state, _ = self.rvm.rnn_core(tokens, state)
        h, w = x.shape[-2] // 16, x.shape[-1] // 16
        feat = new_state[:, 1:].reshape(1, h, w, -1).permute(0, 3, 1, 2)
        return (feat.float() - self.mean) * self.inv_std, new_state


class Stage1Graph(nn.Module):
    """Stage 1 on one Hann window of T frames, with per-frame inputs/outputs.

    Per-frame tensors let the browser chain GPU-resident results between
    sessions (foundation features in, batons/carry hints out) without any
    concat/split session or CPU readback. The carry hint is the aux alpha
    upsampled to full resolution, as runtime.hann_stage1_stream does.

    Inputs (in order): frame_0..frame_{T-1} (1,3,H,W), hint_0..hint_{T-1}
    (1,1,H,W), hint_quality (1,T), cradio_i, moge_i, rvm_i (1,C,H/64,W/64).
    Outputs: baton_0..baton_{T-1} (1,512,H/16,W/16), carry_0..carry_{T-1} (1,1,H,W).
    """

    def __init__(self, stage1, hw, T: int):
        super().__init__()
        self.stage1 = stage1
        self.stage1.bottleneck_head_stride = None  # supervision-only head
        self.hw = hw
        self.T = T

    def forward(self, *args):
        T = self.T
        H, W = self.hw
        frames = torch.stack(args[0:T], dim=1)
        hint = torch.stack(args[T:2 * T], dim=1)
        hint_quality = args[2 * T]
        o = 2 * T + 1
        cradio = torch.stack(args[o:o + T], dim=1)
        moge = torch.stack(args[o + T:o + 2 * T], dim=1)
        rvm = torch.stack(args[o + 2 * T:o + 3 * T], dim=1)
        rgb = srgb_to_model_rgb(frames)
        out = self.stage1(rgb, hint, moge, cradio_feat=cradio, rvm_feat=rvm,
                          t_stride=1.0, hint_quality=hint_quality)
        aux_alpha = out["aux"][0, :, 0:1]                       # (T, 1, H/2, W/2)
        carry = F.interpolate(aux_alpha, size=(H, W), mode="bilinear",
                              align_corners=False).clamp(0.0, 1.0)
        batons = out["baton"][0].unbind(0)
        carries = carry.unbind(0)
        return (*(b.unsqueeze(0) for b in batons), *(c.unsqueeze(0) for c in carries))

    @staticmethod
    def io_names(T: int):
        ins = ([f"frame_{i}" for i in range(T)] + [f"hint_{i}" for i in range(T)] + ["hint_quality"]
               + [f"cradio_{i}" for i in range(T)] + [f"moge_{i}" for i in range(T)]
               + [f"rvm_{i}" for i in range(T)])
        outs = [f"baton_{i}" for i in range(T)] + [f"carry_{i}" for i in range(T)]
        return ins, outs


class Stage2Graph(nn.Module):
    """Stage 2 on one frame. Blends the frame's batons from two overlapping
    Hann windows in-graph: baton = baton_a + blend * (baton_b - baton_a), where
    blend = w_b / (w_a + w_b) from the taper weights (0 when only one window
    covers the frame; pass the same tensor twice)."""

    def __init__(self, stage2):
        super().__init__()
        self.stage2 = stage2

    def forward(self, frame, baton_a, baton_b, blend):
        baton = baton_a + blend.view(1, 1, 1, 1) * (baton_b - baton_a)
        rgb = srgb_to_model_rgb(frame).unsqueeze(1)
        logits = self.stage2(rgb, baton.unsqueeze(1))[:, 0]
        alpha = logits[:, 0:1].clamp(0.0, 1.0)
        fg = model_fg_to_srgb(logits[:, 1:4])
        return alpha, fg
