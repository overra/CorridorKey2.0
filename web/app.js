// CorridorKey 2.0 browser demo: key a video file on WebGPU.

import { CorridorKey, checkWebGPU, configureRuntime } from './corridorkey.js';
import { FramePacker, renderResult } from './frames.js';

const $ = (id) => document.getElementById(id);
const ui = {
  preset: $('preset'), variant: $('variant'), load: $('load'), loadStatus: $('load-status'),
  file: $('file'), hint: $('hint'), fps: $('fps'), maxFrames: $('max-frames'), overlap: $('overlap'),
  run: $('run'), stop: $('stop'), runStatus: $('run-status'), view: $('view'), despill: $('despill'),
  scrub: $('scrub'), frameLabel: $('frame-label'), canvas: $('out'), save: $('save'),
  drop: $('drop'),
};
const video = document.createElement('video');
video.muted = true;
video.playsInline = true;

let gpu = null;         // {device, info} from checkWebGPU, reused across loads
let ck = null;
let packer = null;
let results = [];       // per frame: FramePacker.crop8() output (8-bit, cropped)
let videoUrl = null;
let videoReady = false;
let running = false;
let stopRequested = false;

configureRuntime({
  wasmBase: new URL('node_modules/onnxruntime-web/dist/', location.href).href,
  modelBase: new URL('models/', location.href).href,
});

const fmtGB = (b) => `${(b / 1e9).toFixed(2)} GB`;
const updateRun = () => { ui.run.disabled = running || !ck || !videoReady; };

async function loadPresets() {
  const presets = await (await fetch('models/presets.json')).json().catch(() => ['896x512']);
  ui.preset.replaceChildren(...presets.map((p) => new Option(p, p)));
  if (presets.includes('896x512')) ui.preset.value = '896x512';
  await loadVariants();
}

// Variants offered in the demo, best first; others in config.json are for benchmarking.
const VARIANT_LABELS = {
  fp16_fused: 'fp16, fastest',
  q8_fused: 'int8, fewest errors',
};

async function loadVariants() {
  const meta = (await (await fetch(`models/${ui.preset.value}/config.json`, { cache: 'no-store' })).json()).corridorkey;
  const options = Object.entries(VARIANT_LABELS)
    .filter(([v]) => meta.variants?.[v])
    .map(([v, label]) => {
      const bytes = Object.values(meta.variants[v]).reduce((a, c) => a + c.bytes, 0);
      return new Option(`${label} (${fmtGB(bytes)})`, v);
    });
  ui.variant.replaceChildren(...options);
  // Without shader-f16 (e.g. Chrome on Linux with the NVIDIA driver) only the
  // fp32-math variant can run; preselect it instead of failing on load.
  const adapter = await navigator.gpu?.requestAdapter({ powerPreference: 'high-performance' });
  if (adapter && !adapter.features.has('shader-f16') && meta.variants?.q8_fused) {
    ui.variant.value = 'q8_fused';
    ui.loadStatus.textContent = 'This GPU/browser has no shader-f16, so the fp16 models cannot run; using the int8 variant.';
  }
}

ui.preset.addEventListener('change', loadVariants);

ui.load.addEventListener('click', async () => {
  ui.load.disabled = true;
  const progress = new Map();
  try {
    const dtype = ui.variant.value;
    if (!gpu) gpu = await checkWebGPU(dtype);
    else if (dtype.includes('16') && !gpu.info.f16) throw new Error('this GPU adapter has no shader-f16; pick the int8 variant');
    const a = gpu.info;
    await ck?.dispose();  // release the previous model's GPU memory first
    ck = null;
    updateRun();
    ui.loadStatus.textContent = `GPU: ${a.vendor} ${a.architecture}${a.f16 ? ', shader-f16' : ''}. Loading…`;
    const t0 = performance.now();
    ck = await CorridorKey.load(ui.preset.value, {
      device: gpu.device,
      dtype,
      onProgress: (p) => {
        if (p.status === 'progress' && p.total) {
          progress.set(`${p.component}/${p.file}`, [p.loaded, p.total]);
          let done = 0, total = 0;
          for (const [l, t] of progress.values()) { done += l; total += t; }
          ui.loadStatus.textContent = `Downloading ${fmtGB(done)} / ${fmtGB(total)}…`;
        }
      },
    });
    ui.loadStatus.textContent = `Loaded ${ui.preset.value} ${dtype} in ${((performance.now() - t0) / 1000).toFixed(1)} s.`;
  } catch (e) {
    ui.loadStatus.textContent = `Load failed: ${e.message ?? e}`;
    throw e;
  } finally {
    ui.load.disabled = false;
    updateRun();
  }
});

function setVideo(file) {
  if (running) return;
  if (videoUrl) URL.revokeObjectURL(videoUrl);
  videoReady = false;
  updateRun();
  videoUrl = URL.createObjectURL(file);
  video.src = videoUrl;
  video.onloadedmetadata = () => {
    videoReady = Number.isFinite(video.duration) && video.duration > 0;
    ui.runStatus.textContent = videoReady
      ? `${file.name}: ${video.videoWidth}x${video.videoHeight}, ${video.duration.toFixed(2)} s`
      : `${file.name}: unknown duration; this file cannot be seeked frame by frame.`;
    ui.drop.classList.add('has-file');
    updateRun();
  };
  video.onerror = () => {
    videoReady = false;
    ui.runStatus.textContent = `${file.name}: this browser cannot decode the file (${video.error?.message || 'unsupported format'}).`;
    updateRun();
  };
}

ui.file.addEventListener('change', () => ui.file.files[0] && setVideo(ui.file.files[0]));
ui.drop.addEventListener('dragover', (e) => { e.preventDefault(); ui.drop.classList.add('over'); });
ui.drop.addEventListener('dragleave', () => ui.drop.classList.remove('over'));
ui.drop.addEventListener('drop', (e) => {
  e.preventDefault();
  ui.drop.classList.remove('over');
  const f = [...e.dataTransfer.files].find((x) => x.type.startsWith('video/'));
  if (f) setVideo(f);
});

function seek(time, timeoutMs = 10000) {
  return new Promise((resolve, reject) => {
    const done = (fn) => (ev) => {
      clearTimeout(timer);
      video.removeEventListener('seeked', ok);
      video.removeEventListener('error', fail);
      video.removeEventListener('emptied', fail);
      fn(ev);
    };
    const ok = done(() => resolve());
    const fail = done(() => reject(new Error('video seek failed')));
    const timer = setTimeout(done(() => reject(new Error('video seek timed out'))), timeoutMs);
    video.addEventListener('seeked', ok);
    video.addEventListener('error', fail);
    video.addEventListener('emptied', fail);
    video.currentTime = time;
  });
}

async function loadHint() {
  const f = ui.hint.files[0];
  if (!f) return null;
  // Raw values, like the CUDA runtime's hint reader: no premultiplication or
  // color management; packMatte uses the alpha channel of mattes that have one.
  const bmp = await createImageBitmap(f, { premultiplyAlpha: 'none', colorSpaceConversion: 'none' });
  if (Math.abs(bmp.width / bmp.height - video.videoWidth / video.videoHeight) > 0.01) {
    throw new Error(`first-frame matte is ${bmp.width}x${bmp.height}, but the clip is ${video.videoWidth}x${video.videoHeight}`);
  }
  return packer.packMatte(bmp);
}

function setInputsLocked(locked) {
  for (const el of [ui.file, ui.hint, ui.preset, ui.variant, ui.load, ui.fps, ui.maxFrames, ui.overlap]) el.disabled = locked;
  ui.drop.classList.toggle('locked', locked);
}

ui.run.addEventListener('click', async () => {
  running = true;
  stopRequested = false;
  updateRun();
  setInputsLocked(true);
  ui.stop.disabled = false;
  const pending = new Set();
  let readbackError = null;
  try {
    const fps = Number(ui.fps.value) || 24;
    const limit = Number(ui.maxFrames.value) || Infinity;
    const n = Math.min(Math.floor(video.duration * fps), limit);
    if (!Number.isFinite(n) || n < 1) throw new Error('set a frame rate and a frame limit that give at least one frame');
    packer = new FramePacker(ck.W, ck.H, video.videoWidth, video.videoHeight);
    ui.canvas.width = packer.w;
    ui.canvas.height = packer.h;
    results = new Array(n);
    ui.scrub.max = n - 1;
    const initialHint = await loadHint();
    const stride = ui.overlap.checked ? null : ck.T;
    if (initialHint && stride) {
      ui.runStatus.textContent = 'Note: without overlapping windows the first-frame matte only guides the first window.';
    }
    const t0 = performance.now();
    let done = 0;
    // Read results back while the GPU keeps working on the next frames; a few
    // in flight is enough to hide the readback and the CPU-side rendering.
    const collect = async (t, out) => {
      const [a, f] = await Promise.allSettled([out.alpha.getData(), out.fg.getData()]);
      out.alpha.dispose();
      out.fg.dispose();
      if (a.status === 'rejected' || f.status === 'rejected') {
        readbackError ??= (a.reason ?? f.reason);
        return;
      }
      results[t] = packer.crop8(a.value, f.value);
      done++;
      ui.scrub.value = t;
      show(t);
      const ms = (performance.now() - t0) / done;
      ui.runStatus.textContent = `Frame ${done}/${n} · ${ms.toFixed(0)} ms/frame (${(1000 / ms).toFixed(2)} fps)`;
    };
    await ck.process(n, async (t) => {
      if (stopRequested) throw new Error('stopped');
      await seek(Math.min((t + 0.5) / fps, video.duration - 1e-3));
      if (stopRequested) throw new Error('stopped');
      return packer.pack(video);
    }, async (t, out) => {
      if (readbackError) throw readbackError;
      const job = collect(t, out).finally(() => pending.delete(job));
      pending.add(job);
      if (pending.size > 3) await Promise.race(pending);
    }, { initialHint, stride });
    await Promise.all(pending);
    if (readbackError) throw readbackError;
  } catch (e) {
    await Promise.allSettled(pending);
    if (e.message === 'stopped') ui.runStatus.textContent += ' · stopped';
    else { ui.runStatus.textContent = `Error: ${e.message ?? e}`; throw e; }
  } finally {
    running = false;
    setInputsLocked(false);
    ui.stop.disabled = true;
    ui.save.disabled = !results.some(Boolean);
    updateRun();
  }
});

ui.stop.addEventListener('click', () => { stopRequested = true; });

function show(t) {
  const r = results[t];
  if (!r || !packer) return;
  ui.frameLabel.textContent = `frame ${t}`;
  const img = renderResult(packer, r, ui.view.value, { despill: Number(ui.despill.value) });
  ui.canvas.getContext('2d').putImageData(img, 0, 0);
}

ui.scrub.addEventListener('input', () => show(Number(ui.scrub.value)));
ui.view.addEventListener('change', () => show(Number(ui.scrub.value)));
ui.despill.addEventListener('input', () => show(Number(ui.scrub.value)));
ui.save.addEventListener('click', () => {
  ui.canvas.toBlob((b) => {
    const a = document.createElement('a');
    a.href = URL.createObjectURL(b);
    a.download = `corridorkey_${ui.view.value}_${String(ui.scrub.value).padStart(5, '0')}.png`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  });
});

loadPresets();
