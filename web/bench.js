// Benchmark + parity harness for the CorridorKey web runtime.
//
// bench.html?preset=896x512&dtype=fp16&frames=24
//
// 1. Parity: runs the golden window (onnx_export/parity.py --write-golden)
//    graph by graph and compares every intermediate with the PyTorch fp32
//    reference.
// 2. Throughput: streams `frames` frames through CorridorKey.process()
//    (Hann windows, carry hints) and reports ms/frame, then repeats with a GPU
//    sync after each session run for a per-stage breakdown.
// Results land in window.__bench (read by bench.mjs) and on the page.

import * as ort from 'onnxruntime-web/webgpu';
import { CorridorKey, checkWebGPU, configureRuntime } from './corridorkey.js';

const q = new URLSearchParams(location.search);
const preset = q.get('preset') ?? '896x512';
const dtype = q.get('dtype') ?? 'fp16';
const streamFrames = Number(q.get('frames') ?? 24);
const dtypes = JSON.parse(q.get('dtypes') ?? '{}');
const epOptions = JSON.parse(q.get('ep') ?? '{}');
const goldenName = q.get('golden') ?? 'golden';
const seqName = q.get('seq');  // e.g. golden_seq from onnx_export/seq_reference.py
const stride = q.get('stride') ? Number(q.get('stride')) : null;  // Hann stride (default T/2)
const logEl = document.getElementById('log');
const log = (...a) => {
  const s = a.map((x) => (typeof x === 'string' ? x : JSON.stringify(x))).join(' ');
  console.log(s);
  logEl.textContent += s + '\n';
};

async function fetchBin(name, manifest) {
  const buf = await (await fetch(`models/${preset}/${goldenName}/${name}.bin`)).arrayBuffer();
  return { data: new Float32Array(buf), dims: manifest[name] };
}

function compare(got, ref) {
  let max = 0, sum = 0, nan = false, over1 = 0, over4 = 0;
  for (let i = 0; i < ref.length; i++) {
    const v = got[i];
    if (Number.isNaN(v)) { nan = true; continue; }
    const d = Math.abs(v - ref[i]);
    if (d > max) max = d;
    sum += d;
    if (d > 1 / 255) over1++;
    if (d > 4 / 255) over4++;
  }
  // over_1/255, over_4/255: share of values off by more than 1 or 4 8-bit levels.
  return { max_abs: max, mean_abs: sum / ref.length, nan, over1: over1 / ref.length, over4: over4 / ref.length };
}

const slice = (t, i) => {
  const per = t.data.length / t.dims[0];
  return t.data.subarray(i * per, (i + 1) * per);
};

async function parity(ck, manifest) {
  const T = ck.T;
  const g = {};
  for (const k of Object.keys(manifest)) g[k] = await fetchBin(k, manifest);
  const res = {};
  const acc = (key, got, ref) => {
    const c = compare(got, ref);
    const r = (res[key] ??= { max_abs: 0, mean_abs: 0, over1: 0, over4: 0, nan: false, n: 0 });
    r.max_abs = Math.max(r.max_abs, c.max_abs);
    r.mean_abs += c.mean_abs;
    r.over1 += c.over1;
    r.over4 += c.over4;
    r.nan ||= c.nan;
    r.n += 1;
  };
  const frames = [];
  let state = ck.gpuTensor([1, ck.meta.rvm_tokens, ck.meta.rvm_dim]);
  const feats = [];
  for (let t = 0; t < T; t++) {
    frames.push(ck.uploadFrame(slice(g.frames, t)));
    const f = await ck.foundations(frames[t], state);
    state.dispose();
    state = f.state;
    feats.push(f);
    acc('cradio', await f.cradio.getData(), slice(g.cradio, t));
    acc('moge', await f.moge.getData(), slice(g.moge, t));
    acc('rvm', await f.rvm.getData(), slice(g.rvm, t));
  }
  acc('rvm_state', await state.getData(), g.rvm_state.data);
  state.dispose();
  const feeds = { hint_quality: new ort.Tensor('float32', new Float32Array(T).fill(-1), [1, T]) };
  for (let t = 0; t < T; t++) {
    Object.assign(feeds, {
      [`frame_${t}`]: frames[t], [`hint_${t}`]: ck.zeroHint,
      [`cradio_${t}`]: feats[t].cradio, [`moge_${t}`]: feats[t].moge, [`rvm_${t}`]: feats[t].rvm,
    });
  }
  const s1 = await ck.run('stage1', feeds);
  const zero = new ort.Tensor('float32', new Float32Array([0]), [1]);
  for (let t = 0; t < T; t++) {
    const baton = s1[`baton_${t}`];
    acc('baton', await baton.getData(), slice(g.baton, t));
    acc('carry', await s1[`carry_${t}`].getData(), slice(g.carry, t));
    const out = await ck.run('stage2', { frame: frames[t], baton_a: baton, baton_b: baton, blend: zero });
    acc('alpha', await out.alpha.getData(), slice(g.alpha, t));
    acc('fg', await out.fg.getData(), slice(g.fg, t));
    out.alpha.dispose(); out.fg.dispose();
    baton.dispose(); s1[`carry_${t}`].dispose();
    frames[t].dispose();
    feats[t].cradio.dispose(); feats[t].moge.dispose(); feats[t].rvm.dispose();
  }
  // Isolated: each stage fed the PyTorch reference inputs, so its own error
  // is separated from what upstream stages hand it.
  const upload = (arr, dims) => {
    const t = ck.gpuTensor(dims);
    ck.device.queue.writeBuffer(t.gpuBuffer, 0, arr.buffer, arr.byteOffset, arr.byteLength);
    return t;
  };
  const h64 = ck.H / 64, w64 = ck.W / 64;
  const iso = { hint_quality: feeds.hint_quality };
  const owned = [];
  for (let t = 0; t < T; t++) {
    const fr = upload(slice(g.frames, t), [1, 3, ck.H, ck.W]);
    const c = upload(slice(g.cradio, t), [1, ck.meta.cradio_dim, h64, w64]);
    const m = upload(slice(g.moge, t), [1, ck.meta.moge_dim, h64, w64]);
    const r = upload(slice(g.rvm, t), [1, ck.meta.rvm_dim, h64, w64]);
    owned.push(fr, c, m, r);
    Object.assign(iso, { [`frame_${t}`]: fr, [`hint_${t}`]: ck.zeroHint, [`cradio_${t}`]: c, [`moge_${t}`]: m, [`rvm_${t}`]: r });
  }
  const s1iso = await ck.run('stage1', iso);
  for (let t = 0; t < T; t++) {
    acc('baton(isolated)', await s1iso[`baton_${t}`].getData(), slice(g.baton, t));
    s1iso[`baton_${t}`].dispose(); s1iso[`carry_${t}`].dispose();
    const b = upload(slice(g.baton, t), [1, ck.meta.baton_dim, ck.H / 16, ck.W / 16]);
    const out = await ck.run('stage2', { frame: iso[`frame_${t}`], baton_a: b, baton_b: b, blend: zero });
    acc('alpha(isolated)', await out.alpha.getData(), slice(g.alpha, t));
    out.alpha.dispose(); out.fg.dispose(); b.dispose();
  }
  owned.forEach((x) => x.dispose());
  for (const r of Object.values(res)) { r.mean_abs /= r.n; r.over1 /= r.n; r.over4 /= r.n; delete r.n; }
  return { res, frames: g.frames };
}

async function throughput(ck, golden, n, sync) {
  ck.resetTimings();
  ck.syncTimings = sync;
  const T = golden.dims[0];
  let downloaded = 0;
  const t0 = performance.now();
  await ck.process(n, (t) => slice(golden, t % T), async (t, out) => {
    // Pull results back like a real consumer would (display/encode).
    await out.alpha.getData();
    await out.fg.getData();
    out.alpha.dispose(); out.fg.dispose();
    downloaded++;
  }, { stride });
  const wall = performance.now() - t0;
  ck.syncTimings = false;
  const stages = Object.fromEntries(Object.entries(ck.timings).map(([k, v]) => [k, { calls: v.calls, ms_per_call: +(v.ms / v.calls).toFixed(2), ms_per_frame: +(v.ms / n).toFixed(2) }]));
  return { frames: n, downloaded, wall_ms: +wall.toFixed(1), ms_per_frame: +(wall / n).toFixed(2), fps: +(1000 * n / wall).toFixed(3), stages };
}

// Whole-clip check of CorridorKey.process() (Hann windows, taper blending,
// carry hints) against the CUDA runtime's own temporal path.
async function sequence(ck) {
  const manifest = await (await fetch(`models/${preset}/${seqName}/manifest.json`)).json();
  const get = async (k) => ({ data: new Float32Array(await (await fetch(`models/${preset}/${seqName}/${k}.bin`)).arrayBuffer()), dims: manifest[k] });
  const frames = await get('frames');
  const alpha = await get('alpha');
  const hint = manifest.hint ? (await get('hint')).data : null;
  const n = frames.dims[0];
  const perFrame = [];
  await ck.process(n, (t) => slice(frames, t), async (t, out) => {
    const a = await out.alpha.getData();
    out.alpha.dispose(); out.fg.dispose();
    const c = compare(a, slice(alpha, t));
    perFrame[t] = { t, max_abs: +c.max_abs.toFixed(4), mean_abs: +c.mean_abs.toExponential(2), over4: +c.over4.toExponential(2) };
  }, { stride: stride ?? manifest.stride, initialHint: hint, hintQuality: manifest.hint_quality ?? 0.95 });
  return perFrame;
}

async function main() {
  const result = { preset, dtype, dtypes, epOptions, golden: goldenName, stride, userAgent: navigator.userAgent };
  try {
    configureRuntime({ wasmBase: new URL('node_modules/onnxruntime-web/dist/', location.href).href, modelBase: new URL('models/', location.href).href, browserCache: false });
    const gpu = await checkWebGPU(dtype, dtypes);
    result.adapter = gpu.info;
    log('adapter', result.adapter);
    const t0 = performance.now();
    const ck = await CorridorKey.load(preset, { device: gpu.device, dtype, dtypes, epOptions, onProgress: (p) => { if (p.status === 'done') log('loaded', p.component, p.file ?? ''); } });
    result.load_ms = +(performance.now() - t0).toFixed(0);
    result.load_ms_by_component = Object.fromEntries(Object.entries(ck.loadTimes).map(([k, v]) => [k, Math.round(v)]));
    log('models loaded in', result.load_ms, 'ms', result.load_ms_by_component);
    if (seqName) {
      result.sequence = await sequence(ck);
      log('sequence vs CUDA runtime', result.sequence);
      window.__bench = result;
      return;
    }
    const manifest = await (await fetch(`models/${preset}/${goldenName}/manifest.json`)).json();
    const p0 = performance.now();
    const { res, frames } = await parity(ck, manifest);
    result.first_window_ms = +(performance.now() - p0).toFixed(0);  // includes shader compilation
    result.parity = res;
    log('parity vs PyTorch fp32', res);
    await throughput(ck, frames, ck.T * 2, false);  // warm-up
    // Best of 3: a shared desktop GPU adds +-15% run-to-run noise.
    const runs = [];
    for (let r = 0; r < 3; r++) runs.push(await throughput(ck, frames, streamFrames, false));
    result.throughput_runs_ms = runs.map((r) => r.ms_per_frame);
    result.throughput = runs.reduce((a, b) => (b.ms_per_frame < a.ms_per_frame ? b : a));
    log('throughput', result.throughput, 'runs', result.throughput_runs_ms);
    result.profile = await throughput(ck, frames, Math.min(streamFrames, 12), true);
    log('per-stage (GPU-synced)', result.profile);
  } catch (e) {
    result.error = String(e?.stack ?? e);
    log('ERROR', result.error);
  }
  window.__bench = result;
}

main();
