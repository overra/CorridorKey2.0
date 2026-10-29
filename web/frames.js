// Frame packing and result rendering for the CorridorKey web runtime.
//
// The model works on a fixed preset (e.g. 896x512). A source frame is scaled
// to fit inside it, centered, and the border filled with BORDER_REFLECT_101
// exactly like video_io.center_pad_to_shape, then packed as a (3, H, W)
// float32 sRGB tensor. Results are cropped back to the fitted region.

function reflect101(i, n) {
  if (n === 1) return 0;
  for (;;) {
    if (i < 0) i = -i;
    else if (i >= n) i = 2 * n - 2 - i;
    else return i;
  }
}

export class FramePacker {
  constructor(W, H, srcW, srcH) {
    const scale = Math.min(W / srcW, H / srcH, 1);
    this.W = W;
    this.H = H;
    this.w = Math.max(1, Math.round(srcW * scale));
    this.h = Math.max(1, Math.round(srcH * scale));
    this.padL = (W - this.w) >> 1;
    this.padT = (H - this.h) >> 1;
    this.canvas = new OffscreenCanvas(this.w, this.h);
    this.ctx = this.canvas.getContext('2d', { willReadFrequently: true });
    this.ctx.imageSmoothingQuality = 'high';
    this.xs = Int32Array.from({ length: W }, (_, x) => reflect101(x - this.padL, this.w));
    this.ys = Int32Array.from({ length: H }, (_, y) => reflect101(y - this.padT, this.h));
  }

  /** Draw `source` scaled to the fitted size and return its RGBA8 pixels. */
  draw(source) {
    this.ctx.clearRect(0, 0, this.w, this.h);
    this.ctx.drawImage(source, 0, 0, this.w, this.h);
    return this.ctx.getImageData(0, 0, this.w, this.h).data;
  }

  /** Draw `source` (video, image, bitmap, canvas) and pack it as (3, H, W) float32. */
  pack(source) {
    return this.packRGBA(this.draw(source));
  }

  /** Pack an RGBA8 array of the fitted size (w*h*4). */
  packRGBA(rgba) {
    const { W, H, w, xs, ys } = this;
    const plane = W * H;
    const out = new Float32Array(3 * plane);
    for (let y = 0; y < H; y++) {
      const row = ys[y] * w;
      for (let x = 0; x < W; x++) {
        const i = (row + xs[x]) * 4;
        const o = y * W + x;
        out[o] = rgba[i] / 255;
        out[plane + o] = rgba[i + 1] / 255;
        out[2 * plane + o] = rgba[i + 2] / 255;
      }
    }
    return out;
  }

  /** Pack a matte (white = subject) drawn from `source` as (1, H, W). Like the
   * CUDA runtime's hint reader, an image with transparency contributes its
   * alpha channel, otherwise its first (red) channel. */
  packMatte(source) {
    const rgba = this.draw(source);
    let channel = 0;
    for (let i = 3; i < rgba.length; i += 4) if (rgba[i] !== 255) { channel = 3; break; }
    const { W, H, w, xs, ys } = this;
    const out = new Float32Array(W * H);
    for (let y = 0; y < H; y++) {
      const row = ys[y] * w;
      for (let x = 0; x < W; x++) out[y * W + x] = rgba[(row + xs[x]) * 4 + channel] / 255;
    }
    return out;
  }

  /** Crop padded (1|3, H, W) float results to the fitted region as 8-bit. */
  crop8(alpha, fg) {
    const { W, H, w, h, padL, padT } = this;
    const plane = W * H;
    const a8 = new Uint8Array(w * h);
    const f8 = new Uint8Array(3 * w * h);
    const q = (v) => (v <= 0 ? 0 : v >= 1 ? 255 : (v * 255 + 0.5) | 0);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const s = (y + padT) * W + (x + padL);
        const o = y * w + x;
        a8[o] = q(alpha[s]);
        f8[3 * o] = q(fg[s]);
        f8[3 * o + 1] = q(fg[plane + s]);
        f8[3 * o + 2] = q(fg[2 * plane + s]);
      }
    }
    return { alpha: a8, fg: f8 };
  }
}

// sRGB <-> linear lookup tables (12-bit) for compositing.
const LUT = 4096;
const SRGB_TO_LIN = Float32Array.from({ length: LUT }, (_, i) => {
  const c = i / (LUT - 1);
  return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
});
const LIN_TO_SRGB8 = Uint8Array.from({ length: LUT }, (_, i) => {
  const c = i / (LUT - 1);
  return Math.round(255 * (c <= 0.0031308 ? c * 12.92 : 1.055 * c ** (1 / 2.4) - 0.055));
});
const toLin = (c) => SRGB_TO_LIN[Math.min(LUT - 1, Math.max(0, (c * (LUT - 1) + 0.5) | 0))];
const toSrgb8 = (c) => LIN_TO_SRGB8[Math.min(LUT - 1, Math.max(0, (c * (LUT - 1) + 0.5) | 0))];

/**
 * Render a cropped 8-bit result (FramePacker.crop8) as ImageData.
 * mode: 'alpha' | 'fg' | 'composite' (despilled FG over a checkerboard or
 * `bg` sRGB triple, blended in linear light like the CUDA runtime's preview).
 */
export function renderResult(packer, result, mode, { despill = 0.5, bg = null } = {}) {
  const { w, h } = packer;
  const { alpha, fg } = result;
  const img = new ImageData(w, h);
  const d = img.data;
  for (let o = 0; o < w * h; o++) {
    const p = 4 * o;
    if (mode === 'alpha') {
      d[p] = d[p + 1] = d[p + 2] = alpha[o];
    } else if (mode === 'fg') {
      d[p] = fg[3 * o]; d[p + 1] = fg[3 * o + 1]; d[p + 2] = fg[3 * o + 2];
    } else {
      let r = fg[3 * o] / 255, g = fg[3 * o + 1] / 255, b = fg[3 * o + 2] / 255;
      const spill = Math.max(g - (r + b) * 0.5, 0);
      r += despill * spill * 0.5;
      g -= despill * spill;
      b += despill * spill * 0.5;
      const a = alpha[o] / 255;
      const x = o % w, y = (o / w) | 0;
      const c = ((x >> 4) + (y >> 4)) & 1 ? 0.6 : 0.4;
      const br = bg ? bg[0] : c, bgc = bg ? bg[1] : c, bb = bg ? bg[2] : c;
      d[p] = toSrgb8(toLin(r) * a + toLin(br) * (1 - a));
      d[p + 1] = toSrgb8(toLin(g) * a + toLin(bgc) * (1 - a));
      d[p + 2] = toSrgb8(toLin(b) * a + toLin(bb) * (1 - a));
    }
    d[p + 3] = 255;
  }
  return img;
}
