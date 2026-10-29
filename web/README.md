# CorridorKey 2.0 in the browser (experimental)

CorridorKey 2.0 runs end to end in the browser on WebGPU: the three frozen
foundation models (C-RADIOv4-SO400M, MoGe-2 ViT-B, RVM-small) and both
CorridorKey stages, exported to ONNX, executed by ONNX Runtime Web's WebGPU
execution provider, and loaded through transformers.js. Nothing leaves the
user's machine.

It is a working port, not a replacement for the CUDA runtime: it processes a
clip at a fixed model resolution (896×512 for 16:9 footage), a few frames per
second on a desktop GPU.

## Results

RTX 3090, Chrome 145 (Linux, Vulkan), ONNX Runtime Web 1.31.0-dev.20260914
(the build transformers.js 4.3.1 pins), 896×512 preset, 4-frame Hann windows
with 50% overlap (the CUDA runtime's temporal scheme), outputs read back to
the page. ms/frame is the median of four sessions, each the best of three
24-frame runs (fp16 ranged 221–232 ms, int8 304–318 ms). Accuracy is the
alpha matte against a true fp32 PyTorch reference (TF32 disabled) on one
window. The raw benchmark outputs are in [results/](results/).

| Variant | Download | ms/frame | fps | Alpha error, real clip: mean / >1/255 / >4/255 | Synthetic hair test: mean / >1/255 / >4/255 |
|---|---|---|---|---|---|
| fp16 | 2.41 GB | 223 | 4.5 | 2.9e-4 / 1.68% / 0% | 4.0e-4 / 2.71% / 0.35% |
| int8 weights, fp32 math | 1.49 GB | 310 | 3.2 | 1.4e-4 / 0.15% / 0% | 1.9e-4 / 0.96% / 0% |
| PyTorch bf16 autocast on CUDA (reference) | | 63* | 16* | 1.1e-4 / 0.21% / 0% | 2.1e-4 / 1.47% / 0.06% |

">1/255" is the share of alpha pixels that differ from the fp32 reference by
more than one 8-bit level. The real clip is
[Pexels video 30400393](https://www.pexels.com/video/young-man-expressively-speaking-on-green-screen-30400393/)
(frames 120–123, scaled to cover 896×512 and center-cropped); the synthetic test has dense
hair strands and soft edges. *The CUDA timing is GPU-only model time for one
8-frame window without overlap (no upload, readback, or second stage-1
pass), so it is not like for like with the browser rows; its accuracy is
bf16 autocast of the same PyTorch modules. Over a whole 14-frame clip,
the browser's streamed output (overlapping windows, taper blending, carry
hints) tracks the CUDA runtime's own temporal path with a flat per-frame
mean difference (3.3e-4 for fp16, 1.9e-4 for int8), with and without a
first-frame hint matte.

Other resolution presets (same setup, one session each; downloads barely
change, since the weights do not depend on resolution):

| Preset | fp16: ms/frame (fps) | int8 weights, fp32 math: ms/frame (fps) |
|---|---|---|
| 640×384 | 154 (6.5) | 195 (5.1) |
| 896×512 | 223 (4.5) | 310 (3.2) |
| 1280×768 (≈720p) | 456 (2.2) | 820 (1.2) |

Turning off overlapping windows in the demo runs stage 1 once per frame
instead of twice: 190 ms/frame (5.3 fps) at 896×512 fp16. Without overlap no
carry hint passes between windows, so a first-frame matte only guides the
first 4 frames (as with `--hann_stride` equal to `--hann_chunk` on CUDA), and
window seams are less smooth.

Above 896×512, use fp16: the fp32-math variant's attention tensors double in
size. 1280×768 is about the ceiling for the current graphs: their attention is
unfused, and stage 1's largest attention-score tensor is already 0.9 GB in
fp16 (1.8 GB in fp32), against a 2 GB WebGPU buffer limit on Windows. 1080p
would need fused (flash) attention.

Speedups that moved the fp16 number from ~275 to ~223 ms/frame: replacing
transposed/strided convolutions with MatMuls (largest single win: the stage-2
depatchify head went from 15.6 ms to 1.8 ms), RotaryEmbedding and RMSNorm
fusion, and NCHW session layout.

Notes:

- fp16 accuracy is limited by ONNX Runtime Web 1.31: its fp16 MatMul/Gemm and
  Softmax kernels accumulate in fp16. In the pinned build the
  `enableMatmulFp32Accumulation` EP option only covers MatMulNBits (turning it
  on for a q8f16 variant did not help); upstream extended it to MatMul/Gemm
  after 1.31 (onnxruntime PR #33057). With a release that includes it, pass
  `epOptions: { enableMatmulFp32Accumulation: true }` to `CorridorKey.load`;
  its effect here is not measured yet.
- The int8 variant (int8 weights, including the rewritten conv/deconv
  weights, with fp32 math) puts fewer alpha pixels off by more than one 8-bit
  level than bf16 autocast does (0.15% vs 0.21% on the real clip, 0.96% vs
  1.47% synthetic) at a comparable mean error (1.4e-4 vs 1.1e-4 real). It is
  ~39% slower than fp16 at 896×512 despite the smaller download: its
  activations are fp32.
- 4-bit weight quantization (round-to-nearest, block 32) is not usable as
  is: C-RADIO's MLP output weights have extreme outliers (max/std up to 232),
  and the mean alpha error rises about 30-fold (5.7e-3 on the synthetic
  test).

## Try it

From the repository root, after the normal setup in the main README:

```
pip install onnx onnxscript onnxruntime-gpu onnxconverter-common
python onnx_export/export.py --height 512 --width 896 --frames 4 --out web/models/896x512
python onnx_export/optimize.py --models web/models/896x512 --variant fp16 --fuse --tag _fused
python onnx_export/optimize.py --models web/models/896x512 --variant q8 --fuse --tag _fused
python onnx_export/blend_graph.py --models web/models/896x512
cd web && npm install && node serve.mjs
```

Open http://localhost:8787/, load a model, drop a clip, and key it. The
export takes a few minutes on CPU. The fp32 intermediate graphs in `raw/`
need ~5 GB of disk per preset; the browser only loads the variant files, so
`raw/` can be deleted once the variants are built (rebuilding variants and
the fp32 parity check need it).

To check accuracy and speed in a real browser (needs an NVIDIA GPU with CUDA
for the PyTorch references, and Linux with X11 or Xwayland for the bench):

```
python onnx_export/parity.py --models web/models/896x512 --write-golden
cd web && node bench.mjs --dtype fp16_fused
cd web && node bench.mjs --dtype q8_fused
```

`bench.mjs` launches Chrome headed with an off-screen window on the X display
in `$BENCH_DISPLAY` (default `:1`; use `BENCH_DISPLAY=:0` on a plain X11
session) because headless Chrome on Linux only exposes the SwiftShader CPU
adapter. `--exe <path>` picks another Chrome or Chromium binary, `--out`
saves the results JSON, and `--stride 4` measures non-overlapping windows.

The real-clip and whole-clip checks (`clip.mp4` is the Pexels clip above):

```
python onnx_export/parity.py --models web/models/896x512 --write-golden --golden-name golden_real --clip clip.mp4 --start 120
python onnx_export/seq_reference.py --models web/models/896x512 --clip clip.mp4 --start 120 --frames 14
python onnx_export/seq_reference.py --models web/models/896x512 --clip clip.mp4 --start 120 --frames 14 --name golden_seq_hint --hint-from golden_seq
cd web && node bench.mjs --dtype fp16_fused --golden golden_real
cd web && node bench.mjs --dtype fp16_fused --seq golden_seq_hint
```

`seq_reference.py` runs the CUDA runtime's own `hann_stage1_stream` (bf16)
on the clip, with `--hint-from` adding a first-frame matte; the browser's
streamed output is compared frame by frame.

### Browser requirements

- WebGPU. Chrome and Edge on Windows and macOS have it by default. On Linux,
  Chrome enables it only for some GPUs; the measurements here used
  `--enable-unsafe-webgpu --ignore-gpu-blocklist --enable-features=Vulkan
  --ozone-platform=x11` (the last one under Wayland).
- The fp16 variant needs the `shader-f16` feature. On Linux with the NVIDIA
  driver, Chrome 145 does not expose it even on driver 615.71 (Dawn enables it
  by default for drivers ≥ 615.71 only since a late-September 2026 change);
  until your Chrome includes that change, start it with
  `--enable-dawn-features=vulkan_enable_f16_on_nvidia`. The int8 variant
  works without it, and the demo preselects it when `shader-f16` is missing.
- GPU memory (peak above idle, sampled with `nvidia-smi` every 0.5 s during
  a benchmark run): about 4.7 GB for fp16 at 896×512, 5.9 GB for the int8
  variant, and 9.0 GB for fp16 at 1280×768.
- Downloads are 1.5–2.4 GB. transformers.js keeps them in the Cache API, in a
  store named after the preset's `build_id`, so rebuilding a preset replaces
  the cached copy instead of serving stale graphs.

## How it works

Five static-shape graphs per resolution preset, plus a tiny blend graph:

| Graph | Input | Output |
|---|---|---|
| `cradio` | padded sRGB frame | C-RADIO features at /64 |
| `moge` | padded sRGB frame | MoGe features at /64 |
| `rvm` | frame + recurrent state | RVM features at /64, next state |
| `stage1` | T frames, hints, features | T batons at /16, T carry hints |
| `stage2` | frame, two batons, blend weight | alpha, foreground |

Each graph folds in its own preprocessing (area/bilinear resizes as constant
MatMuls, sRGB to linear to the asinh model encoding, feature normalization)
and postprocessing, so the page uploads one frame tensor and every
intermediate stays on the GPU: features, RVM state, batons and carry hints go
from one ONNX Runtime session to the next as GPU buffers on one shared
`GPUDevice`. `corridorkey.js` reproduces `corridorkey/runtime.py`'s temporal
logic: overlapping windows, Hann taper weights, the running baton average,
and the aux-alpha carry hint for frames shared with the next window. One
difference: a clip shorter than the window (4 frames) repeats its last frame
to fill the static window, where the CUDA runtime runs a shorter window.

transformers.js loads each graph (`AutoModel.from_pretrained` with
`subfolder` and `model_file_name`): it resolves the files, fetches chunked
external data, caches it, and reports progress. A preset folder is laid out
as a Hugging Face model repo (`config.json` at the root, graphs under
`<component>/onnx/`).

### Export changes

The exported graphs compute the same function as the PyTorch model; each
change below was checked against the original modules.

- Shifted-window attention uses precomputed additive masks instead of the
  CUDA path's per-region SDPA calls, which index with `nonzero().tolist()`
  and cannot be traced.
- RVM is exported as a single recurrent step.
- MoGe's same-size antialiased resize (an identity, and unsupported by the
  WebGPU EP) is skipped.
- Overlapping strided convolutions and the transposed-convolution depatchify
  heads become one MatMul plus shifted sums. ORT's WebGPU GEMM reaches ~15–17
  TFLOPS here, while its ConvTranspose took 15.6 ms for the stage-2 head
  (1.8 ms after the rewrite).
- RoPE becomes `com.microsoft::RotaryEmbedding`, RMSNorm becomes
  `SimplifiedLayerNormalization`, and the sessions run in NCHW layout.

## Files

- `onnx_export/`: export wrappers and patches, the variant optimizer, the
  PyTorch-vs-ONNX parity check, and op microbenchmarks.
- `web/corridorkey.js`: the browser runtime.
- `web/index.html`, `web/app.js`, `web/frames.js`: the demo page.
- `web/bench.html`, `web/bench.js`, `web/bench.mjs`: the parity and throughput benchmark.
- `web/micro.html`, `web/micro.js`: single-op WebGPU timings.
- `web/results/`: the benchmark outputs behind the numbers above.
