"""Export-time replacements for the shifted-window attention fast path.

`SwinBlock2D`/`SwinBlock3D` group shifted-window tokens into per-region SDPA
calls using Python index lists derived with `nonzero().tolist()`. That is a
CUDA launch-count optimization and cannot be traced. The replacements below
compute the same attention with one masked SDPA per block: tokens attend only
to tokens of the same shifted region, exactly as the region groups do. Padded
query rows (cropped away afterwards) attend to the real keys, as in
`blocks._shifted_attention_mask_2d`, so every softmax row stays finite.

Masks are built in numpy for the static export shape, so they enter the graph
as constants instead of traced ops. Call `install()` before exporting and
`check_equivalence()` to compare against the original modules.
"""

from __future__ import annotations

import math
import types

import numpy as np
import torch
import torch.nn.functional as F

from corridorkey import blocks

MASK_NEG = -30000.0  # finite in fp16; exp() underflows to exactly 0

_ORIG = {
    "SwinBlock2D.forward": blocks.SwinBlock2D.forward,
    "SwinBlock3D.forward": blocks.SwinBlock3D.forward,
}
# numpy, not torch: the first lookup happens inside an export trace, and a
# cached tensor would leak that trace's fake-tensor mode into the next export.
_BIAS_CACHE: dict[tuple, np.ndarray] = {}


def _region_ids_np(H, W, Hp, Wp, w, sh, sw) -> np.ndarray:
    """numpy port of blocks._shifted_region_ids_2d: (num_windows, w*w) int ids, -1 = pad."""
    region = np.full((Hp, Wp), -1, np.int32)
    real = np.zeros((Hp, Wp), bool)
    real[:H, :W] = True
    h_slices = (slice(0, -w), slice(-w, -sh), slice(-sh, None)) if sh > 0 else (slice(0, None),)
    w_slices = (slice(0, -w), slice(-w, -sw), slice(-sw, None)) if sw > 0 else (slice(0, None),)
    idx = 0
    for hs in h_slices:
        for ws in w_slices:
            sub = region[hs, ws]
            sub[real[hs, ws]] = idx
            idx += 1
    region = np.roll(region, shift=(-sh, -sw), axis=(0, 1))
    nH, nW = Hp // w, Wp // w
    return region.reshape(nH, w, nW, w).transpose(0, 2, 1, 3).reshape(nH * nW, w * w)


def shifted_window_bias(H, W, Hp, Wp, w, sh, sw) -> torch.Tensor:
    """Additive attention bias (num_windows, w*w, w*w): 0 = attend, MASK_NEG = blocked."""
    key = (H, W, Hp, Wp, w, sh, sw)
    if key not in _BIAS_CACHE:
        region = _region_ids_np(H, W, Hp, Wp, w, sh, sw)
        q_real = region[:, :, None] >= 0
        k_real = region[:, None, :] >= 0
        same = region[:, :, None] == region[:, None, :]
        allowed = np.where(q_real, same & q_real & k_real, np.broadcast_to(k_real, same.shape))
        _BIAS_CACHE[key] = np.where(allowed, 0.0, MASK_NEG).astype(np.float32)
    return torch.tensor(_BIAS_CACHE[key])


def _masked_attention(q, k, v, bias2d: torch.Tensor, T: int) -> torch.Tensor:
    """softmax(q k^T / sqrt(d) + bias) v for window attention.

    q, k, v: (B*nW, heads, T*S, d); bias2d: (nW, S, S), shared by every frame
    pair in a full-temporal window, since token order is (t, h, w):
    bias[(t,s),(t',s')] = bias2d[s, s']. The bias is broadcast-added to a
    (.., T, S, T, S) view of the scores, so no (T*S)^2 bias is materialized,
    and nothing in the graph depends on fully-masked rows (none exist).
    """
    BW, nh, N, d = q.shape
    nW, S, _ = bias2d.shape
    scores = torch.matmul(q * (d ** -0.5), k.transpose(-1, -2))
    scores = scores.view(BW // nW, nW, nh, T, S, T, S) + bias2d.view(1, nW, 1, 1, S, 1, S)
    probs = torch.softmax(scores.view(BW, nh, N, N), dim=-1)
    return torch.matmul(probs, v)


def _swin_attn2d_masked(attn_mod, x: torch.Tensor, bias2d: torch.Tensor) -> torch.Tensor:
    """SwinAttention2D.forward for a pre-padded, pre-shifted grid, with region
    isolation from a (num_windows, S, S) bias. Keeps SDPA 4-D for the exporter."""
    B, H, W, C = x.shape
    m = attn_mod
    w = m.window_size
    assert H % w == 0 and W % w == 0, "caller pads to whole windows"
    N = w * w
    nW = (H // w) * (W // w)
    if m.qkv_dw is not None:
        y = m.qkv_dw(m.qkv.qkv(x))
        windows, _ = blocks.window_partition_2d(y, w)
        qkv = windows.reshape(B * nW, N, 3, m.num_heads, m.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(dim=0)
    else:
        windows, _ = blocks.window_partition_2d(x, w)
        q, k, v = m.qkv(windows.reshape(B * nW, N, C))
    gate_windows, _ = blocks.window_partition_2d(x, w)
    gate_tokens = gate_windows.reshape(B * nW, N, C)
    if m.qk_norm:
        q = m.q_norm(q)
        k = m.k_norm(k)
    cos, sin = m.rope.get_cos_sin(w, w, device=x.device)
    q = blocks.apply_rotary(q, cos, sin)
    k = blocks.apply_rotary(k, cos, sin)
    attn = _masked_attention(q, k, v, bias2d, 1)
    attn = m.attn_gate(gate_tokens, attn)
    attn = attn.transpose(1, 2).reshape(B * nW, N, C)
    attn = m.proj(attn)
    return blocks.window_unpartition_2d(attn.reshape(B * nW, w, w, C), w, (H, W), B)


def _swin2d_forward(self, x: torch.Tensor) -> torch.Tensor:
    sh, sw = self.shift_h, self.shift_w
    shortcut = x
    x = self.norm1(x)
    if sh or sw:
        H, W = x.shape[1], x.shape[2]
        w = self.window_size
        Hp = math.ceil(H / w) * w
        Wp = math.ceil(W / w) * w
        if Hp != H or Wp != W:
            x = F.pad(x, (0, 0, 0, Wp - W, 0, Hp - H))
        bias = shifted_window_bias(H, W, Hp, Wp, w, sh, sw).to(device=x.device, dtype=x.dtype)
        x = blocks._cyclic_shift_2d(x, (-sh, -sw))
        x = _swin_attn2d_masked(self.attn, x, bias)
        x = blocks._cyclic_shift_2d(x, (sh, sw))
        x = x[:, :H, :W, :].contiguous()
    else:
        x = self.attn(x)
    x = shortcut + x
    if self.local_mix is not None:
        x = x + self.local_mix(x)
    shortcut = x
    x = self.norm2(x)
    x = self.mlp(x)
    return shortcut + x


def _swin_attn3d_masked(attn_mod, x: torch.Tensor, t_stride: float,
                        bias2d: torch.Tensor) -> torch.Tensor:
    """SwinAttention3D.forward (non-chunked path) for a pre-padded, pre-shifted
    grid, with region isolation from a spatial (num_windows, S, S) bias."""
    B, T, H, W, C = x.shape
    m = attn_mod
    H_w, W_w = m.window_H, m.window_W
    assert H % H_w == 0 and W % W_w == 0, "caller pads to whole windows"
    S = H_w * W_w
    N = T * S
    nW = (H // H_w) * (W // W_w)
    if m.qkv_dw is not None:
        y = m.qkv_dw(m.qkv.qkv(x))
        windows, _ = blocks.window_partition_3d(y, T, H_w, W_w)
        qkv = windows.reshape(B * nW, N, 3, m.num_heads, m.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(dim=0)
    else:
        windows, _ = blocks.window_partition_3d(x, T, H_w, W_w)
        q, k, v = m.qkv(windows.reshape(B * nW, N, C))
    gate_windows, _ = blocks.window_partition_3d(x, T, H_w, W_w)
    gate_tokens = gate_windows.reshape(B * nW, N, C)
    if m.qk_norm:
        q = m.q_norm(q)
        k = m.k_norm(k)
    cos, sin = m.rope.get_cos_sin(T, H_w, W_w, device=x.device, t_stride=t_stride)
    q = blocks.apply_rotary(q, cos, sin)
    k = blocks.apply_rotary(k, cos, sin)
    attn = _masked_attention(q, k, v, bias2d, T)
    attn = m.attn_gate(gate_tokens, attn)
    attn = attn.transpose(1, 2).reshape(B * nW, N, C)
    attn = m.proj(attn)
    out_w = attn.reshape(B * nW, T, H_w, W_w, C)
    return blocks.window_unpartition_3d(out_w, T, H_w, W_w, (T, H, W), B)


def _swin3d_forward(self, x: torch.Tensor, t_stride: float = 1.0,
                    adaln_cond: torch.Tensor | None = None) -> torch.Tensor:
    if self.adaln is not None and adaln_cond is None:
        adaln_cond = x.new_zeros(x.shape[0], x.shape[1])
    if self.adaln is not None:
        shift1, scale1, gate1, shift2, scale2, gate2 = self.adaln(adaln_cond, x)
    else:
        shift1 = scale1 = gate1 = shift2 = scale2 = gate2 = None
    sh, sw = self.shift_h, self.shift_w
    shortcut = x
    x = self.norm1(x)
    if shift1 is not None:
        x = x * (1.0 + scale1) + shift1
    if sh or sw:
        T, H, W = x.shape[1], x.shape[2], x.shape[3]
        H_w, W_w = self.window_H, self.window_W
        Hp = math.ceil(H / H_w) * H_w
        Wp = math.ceil(W / W_w) * W_w
        if Hp != H or Wp != W:
            x = F.pad(x, (0, 0, 0, Wp - W, 0, Hp - H))
        bias2d = shifted_window_bias(H, W, Hp, Wp, H_w, sh, sw).to(device=x.device, dtype=x.dtype)
        x = blocks._cyclic_shift_3d(x, (0, -sh, -sw))
        x = _swin_attn3d_masked(self.attn, x, t_stride, bias2d)
        x = blocks._cyclic_shift_3d(x, (0, sh, sw))
        x = x[:, :, :H, :W, :].contiguous()
    else:
        x = self.attn(x, t_stride=t_stride)
    x = shortcut + (x * gate1 if gate1 is not None else x)
    x = self._local_mix_frames(x)
    shortcut = x
    x = self.norm2(x)
    if shift2 is not None:
        x = x * (1.0 + scale2) + shift2
    x = self.mlp(x)
    return shortcut + (x * gate2 if gate2 is not None else x)


def install() -> None:
    blocks.SwinBlock2D.forward = _swin2d_forward
    blocks.SwinBlock3D.forward = _swin3d_forward


def uninstall() -> None:
    blocks.SwinBlock2D.forward = _ORIG["SwinBlock2D.forward"]
    blocks.SwinBlock3D.forward = _ORIG["SwinBlock3D.forward"]


@torch.no_grad()
def check_equivalence(model, H: int, W: int, T: int, device: str = "cuda") -> dict:
    """Max |patched - original| for every shifted Swin block at the given shape."""
    torch.manual_seed(0)
    res = {}
    for name, mod in model.named_modules():
        if isinstance(mod, blocks.SwinBlock2D) and (mod.shift_h or mod.shift_w):
            x = torch.randn(1, H // 16, W // 16, mod.dim, device=device)
            ref = _ORIG["SwinBlock2D.forward"](mod, x)
            res[name] = float((_swin2d_forward(mod, x) - ref).abs().max())
        elif isinstance(mod, blocks.SwinBlock3D) and (mod.shift_h or mod.shift_w):
            x = torch.randn(1, T, H // 16, W // 16, mod.dim, device=device)
            cond = torch.rand(1, T, device=device) if mod.adaln is not None else None
            ref = _ORIG["SwinBlock3D.forward"](mod, x, 1.0, cond)
            res[name] = float((_swin3d_forward(mod, x, 1.0, cond) - ref).abs().max())
    return res


# ---------------------------------------------------------------------------
# Convolutions as one MatMul + shifted sums
#
# ORT WebGPU runs large dense GEMMs at ~15-17 TFLOPS (fp16, RTX 3090) but
# overlapping strided convs far slower: ConvTranspose(k=32, s=16) depatchify
# ~15 ms, Conv(k=8, s=4) ~3.7 TFLOPS. These rewrites are exact reorderings of
# the same sums, so they change only which kernels run. (As MatMuls, their
# weights are also quantized in optimize.py's q8/q4 variants.)
# ---------------------------------------------------------------------------


def _space_to_depth(x: torch.Tensor, s: int) -> torch.Tensor:
    """(B, C, H, W) -> (B, H/s, W/s, C*s*s), feature order (c, ry, rx)."""
    B, C, H, W = x.shape
    if s == 1:
        return x.permute(0, 2, 3, 1)
    x = x.reshape(B, C, H // s, s, W // s, s).permute(0, 2, 4, 1, 3, 5)
    return x.reshape(B, H // s, W // s, C * s * s)


def _conv_mm_forward(self, x: torch.Tensor) -> torch.Tensor:
    """Conv2d with kernel m*s, stride s, as space-to-depth + one MatMul to all
    m*m kernel-block partial sums + a shifted sum over the blocks:
    y[o] = sum_q Z_q[o + q], Z_q[t] = sum_r x_pad[t*s + r] W[q*s + r]."""
    s, m, p = self._mm_s, self._mm_m, self._mm_p
    B, C, H, W = x.shape
    Ho, Wo = (H + 2 * p - m * s) // s + 1, (W + 2 * p - m * s) // s + 1
    Hp, Wp = (Ho + m - 1) * s, (Wo + m - 1) * s
    x = F.pad(x, (p, Wp - W - p, p, Hp - H - p))
    z = torch.matmul(_space_to_depth(x, s), self._mm_weight)     # (B, Ho+m-1, Wo+m-1, m*m*out)
    out = self._mm_weight.shape[1] // (m * m)
    z = z.reshape(B, Ho + m - 1, Wo + m - 1, m * m, out)
    y = None
    for qy in range(m):
        for qx in range(m):
            part = z[:, qy:qy + Ho, qx:qx + Wo, qy * m + qx]
            y = part if y is None else y + part
    y = y + self._mm_bias
    return y.permute(0, 3, 1, 2)


def _deconv_mm_forward(self, x: torch.Tensor) -> torch.Tensor:
    """ConvTranspose2d(kernel 2s, stride s, pad s/2) as one MatMul to every
    token's 2s x 2s patch + col2im: quadrant (qy, qx) lands qy/qx tokens later,
    so y = sum_q shift(Z_q, (q*s - p))."""
    s, p = self._mm_s, self._mm_p
    B, C, H, W = x.shape
    out = self._mm_weight.shape[1] // (4 * s * s)
    z = torch.matmul(x.flatten(2).transpose(1, 2), self._mm_weight)      # (B, HW, 4*out*s*s)
    z = z.reshape(B, H, W, 4, out, s, s).permute(0, 3, 4, 1, 5, 2, 6)   # (B, 4, out, H, s, W, s)
    z = z.reshape(B, 4, out, H * s, W * s)
    y = None
    for q in range(4):
        top, left = (q // 2) * s - p, (q % 2) * s - p
        img = F.pad(z[:, q], (max(left, 0), max(-left, 0), max(top, 0), max(-top, 0)))
        y0, x0 = max(-top, 0), max(-left, 0)
        img = img[:, :, y0:y0 + H * s, x0:x0 + W * s]
        y = img if y is None else y + img
    return y + self._mm_bias.view(1, -1, 1, 1)


def _prepare_conv_mm(conv: torch.nn.Conv2d) -> bool:
    k, s, p = conv.kernel_size[0], conv.stride[0], conv.padding[0]
    if (conv.kernel_size[0] != conv.kernel_size[1] or conv.stride[0] != conv.stride[1]
            or conv.padding[0] != conv.padding[1] or conv.groups != 1 or conv.dilation != (1, 1)
            or k % s != 0 or k == s):
        return False
    m = k // s
    w = conv.weight.detach()                                     # (out, C, m*s, m*s)
    out, C = w.shape[:2]
    w = w.reshape(out, C, m, s, m, s).permute(1, 3, 5, 2, 4, 0)  # (C, ry, rx, qy, qx, out)
    conv._mm_s, conv._mm_m, conv._mm_p = s, m, p
    conv.register_buffer("_mm_weight", w.reshape(C * s * s, m * m * out).contiguous(), persistent=False)
    b = conv.bias.detach() if conv.bias is not None else torch.zeros(out)
    conv.register_buffer("_mm_bias", b.clone(), persistent=False)
    conv.forward = types.MethodType(_conv_mm_forward, conv)
    return True


def _prepare_deconv_mm(dc: torch.nn.ConvTranspose2d) -> bool:
    k, s, p = dc.kernel_size[0], dc.stride[0], dc.padding[0]
    if k != 2 * s or p != s // 2 or dc.groups != 1 or dc.output_padding != (0, 0):
        return False
    w = dc.weight.detach()                                       # (C, out, 2s, 2s)
    C, out = w.shape[:2]
    w = w.reshape(C, out, 2, s, 2, s).permute(0, 2, 4, 1, 3, 5)  # (C, qy, qx, out, ry, rx)
    dc._mm_s, dc._mm_p = s, p
    dc.register_buffer("_mm_weight", w.reshape(C, 4 * out * s * s).contiguous(), persistent=False)
    b = dc.bias.detach() if dc.bias is not None else torch.zeros(out)
    dc.register_buffer("_mm_bias", b.clone(), persistent=False)
    dc.forward = types.MethodType(_deconv_mm_forward, dc)
    return True


CONV_REWRITE_NAMES = ("down", "mid_down", "smooth")  # Downsample4 (k8 s4), Upsample4ICNR 3x3


def install_conv_rewrites(model) -> list[str]:
    """Swap overlapping strided convs/deconvs for MatMul forms, in place.
    Returns the names of the rewritten modules."""
    done = []
    for name, mod in model.named_modules():
        leaf = name.rsplit(".", 1)[-1]
        if isinstance(mod, torch.nn.ConvTranspose2d) and leaf == "deconv":
            if _prepare_deconv_mm(mod):
                done.append(name)
        elif isinstance(mod, torch.nn.Conv2d) and any(f".{n}." in f".{name}." for n in CONV_REWRITE_NAMES):
            if _prepare_conv_mm(mod):
                done.append(name)
    return done
