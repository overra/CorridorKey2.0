// CorridorKey 2.0 in the browser.
//
// Five static-shape ONNX graphs (C-RADIO, MoGe, RVM, stage 1, stage 2) run on
// ONNX Runtime Web's WebGPU execution provider. transformers.js loads them
// (dtype-suffixed files, chunked external data, session options) and we
// drive the underlying ORT sessions directly so every intermediate tensor --
// foundation features, RVM state, batons, carry hints -- stays on the GPU.
//
// Temporal processing mirrors corridorkey/runtime.py: stage 1 runs on
// overlapping Hann windows of T frames, batons of overlapping windows are
// combined with the same taper weights, and each window's aux alpha is
// carried as the hint for the frames it shares with the next window.

import * as ort from 'onnxruntime-web/webgpu';
import { AutoModel, env } from '@huggingface/transformers';

export const COMPONENTS = ['cradio', 'moge', 'rvm', 'stage1', 'stage2'];
const CARRY_HINT_QUALITY = 0.95;

/** Window start frames, as video_io.window_starts. */
export function windowStarts(total, chunk, stride) {
  if (total <= chunk) return [0];
  const starts = [];
  for (let s = 0; s <= total - chunk; s += stride) starts.push(s);
  if (starts[starts.length - 1] !== total - chunk) starts.push(total - chunk);
  return starts;
}

/** Hann taper weights for one window, as video_io.taper_weights. */
export function taperWeights(Tc, overlap, isFirst, isLast) {
  const w = new Float32Array(Tc).fill(1);
  if (overlap <= 0 || Tc <= 1) return w;
  const n = Math.min(overlap, Tc);
  const ramp = Array.from({ length: n }, (_, i) => 0.5 * (1 - Math.cos(Math.PI * (i + 1) / (overlap + 1))));
  if (!isFirst) for (let i = 0; i < n; i++) w[i] = ramp[i];
  if (!isLast) for (let i = 0; i < n; i++) w[Tc - n + i] = ramp[n - 1 - i];
  return w;
}

/**
 * Configure ONNX Runtime Web and transformers.js for local, self-hosted models.
 * Must run before the first model load.
 */
export function configureRuntime({ wasmBase, modelBase, browserCache = true }) {
  ort.env.wasm.wasmPaths = wasmBase;
  ort.env.webgpu.powerPreference = 'high-performance';
  env.allowRemoteModels = false;
  env.allowLocalModels = true;
  env.localModelPath = modelBase;
  // transformers.js keeps fetched model files in the Cache API so a reload does
  // not download ~1.5-2.4 GB again (storage quota permitting). It never
  // revalidates cached files, so CorridorKey.load names the cache after the
  // preset's build_id (see cacheStoreFor).
  env.useBrowserCache = browserCache;
}

const CACHE_PREFIX = 'corridorkey-';

/** Point transformers.js at a Cache API store for this exact model build and
 * delete stores left by earlier builds of the same preset. */
async function cacheStoreFor(preset, buildId) {
  if (!env.useBrowserCache || typeof caches === 'undefined') return;
  const name = `${CACHE_PREFIX}${preset}-${buildId ?? 'unversioned'}`;
  env.cacheKey = name;
  for (const key of await caches.keys()) {
    if (key.startsWith(`${CACHE_PREFIX}${preset}-`) && key !== name) await caches.delete(key);
  }
}

// Device features the graphs may use. Experimental ones that flags such as
// --enable-unsafe-webgpu expose (e.g. chromium-experimental-subgroup-matrix)
// are deliberately not requested: ORT then selects kernels for them, and the
// subgroup-matrix MatMulNBits shader fails to compile on NVIDIA/Vulkan.
const DEVICE_FEATURES = ['shader-f16', 'subgroups', 'timestamp-query'];
const DEVICE_LIMITS = [
  'maxBufferSize', 'maxStorageBufferBindingSize', 'maxComputeWorkgroupStorageSize',
  'maxComputeInvocationsPerWorkgroup', 'maxComputeWorkgroupSizeX', 'maxComputeWorkgroupSizeY',
  'maxComputeWorkgroupSizeZ', 'maxComputeWorkgroupsPerDimension', 'maxStorageBuffersPerShaderStage',
];

/**
 * Create the GPUDevice ONNX Runtime will use (before the first session) and
 * fail early with an actionable message when the fp16 graphs cannot run.
 */
export async function checkWebGPU(dtype, dtypes = {}) {
  if (!navigator.gpu) throw new Error('WebGPU is not available in this browser.');
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('No WebGPU adapter available.');
  const info = adapter.info ?? {};
  const needsF16 = [dtype, ...Object.values(dtypes)].some((d) => d.includes('16'));
  if (needsF16 && !adapter.features.has('shader-f16')) {
    const hint = info.vendor === 'nvidia'
      ? ' On Linux with the NVIDIA driver, start Chrome with --enable-dawn-features=vulkan_enable_f16_on_nvidia.'
      : '';
    throw new Error(`This GPU adapter does not expose shader-f16, which the ${dtype} models need.${hint}`);
  }
  const requiredLimits = Object.fromEntries(DEVICE_LIMITS.map((k) => [k, adapter.limits[k]]));
  const device = await adapter.requestDevice({
    requiredFeatures: DEVICE_FEATURES.filter((f) => adapter.features.has(f)),
    requiredLimits,
  });
  return {
    device,
    info: {
      vendor: info.vendor, architecture: info.architecture, description: info.description,
      f16: adapter.features.has('shader-f16'),
      maxBufferSize: adapter.limits.maxBufferSize,
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
    },
  };
}

export class CorridorKey {
  /**
   * @param {string} preset  model folder under env.localModelPath, e.g. '896x512'.
   *   One folder is one transformers.js model repo: config.json at its root
   *   and each graph at <component>/onnx/model_<variant>.onnx.
   * @param {{device: GPUDevice, dtype?: string, dtypes?: Record<string,string>,
   *          epOptions?: object, onProgress?: Function}} opts
   *   device: from checkWebGPU(); every session shares it so GPU tensors can
   *   flow between them. dtypes: per-component variant overrides.
   *   epOptions: extra WebGPU EP options.
   */
  static async load(preset, { device, dtype = 'fp16_fused', dtypes = {}, epOptions = {}, onProgress } = {}) {
    if (typeof GPUDevice === 'undefined' || !(device instanceof GPUDevice)) {
      throw new TypeError('CorridorKey.load needs the GPUDevice from checkWebGPU().');
    }
    const res = await fetch(`${env.localModelPath}${preset}/config.json`, { cache: 'no-store' });
    if (!res.ok) throw new Error(`No model preset "${preset}" (${res.status}).`);
    const config = await res.json();
    const meta = config.corridorkey;
    for (const v of new Set([dtype, ...Object.values(dtypes)])) {
      if (!meta.variants?.[v]) {
        throw new Error(`Preset ${preset} has no "${v}" variant; built: ${Object.keys(meta.variants ?? {}).join(', ')}.`);
      }
    }
    await cacheStoreFor(preset, meta.build_id);
    const sessionOptions = {
      // NCHW: the native EP defaults to NHWC and wraps every Conv in
      // Transposes; these graphs are NCHW throughout (~5% faster measured).
      executionProviders: [{ name: 'webgpu', device, preferredLayout: 'NCHW', ...epOptions }],
      graphOptimizationLevel: 'all',
      preferredOutputLocation: 'gpu-buffer',
    };
    const sessions = {};
    const loadTimes = {};
    try {
      for (const name of [...COMPONENTS, 'blend']) {
        const t0 = performance.now();
        // Every variant is stored as model_<variant>.onnx; dtype 'fp32' keeps
        // transformers.js from appending its own dtype suffix. The blend graph
        // is plain model.onnx: config.json's external-data entries are keyed by
        // file name across all subfolders, and it has none.
        const variant = dtypes[name] ?? dtype;
        const model = await AutoModel.from_pretrained(preset, {
          subfolder: `${name}/onnx`,
          dtype: 'fp32',
          model_file_name: name === 'blend' ? 'model' : `model_${variant}`,
          device: 'webgpu',
          session_options: sessionOptions,
          progress_callback: onProgress ? (p) => onProgress({ component: name, ...p }) : undefined,
        });
        sessions[name] = model.sessions.model;
        loadTimes[name] = performance.now() - t0;
      }
      return new CorridorKey(meta, sessions, device, loadTimes);
    } catch (e) {
      // ORT keeps sessions (and their GPU weights) alive until released.
      await Promise.allSettled(Object.values(sessions).map((s) => s.release()));
      throw e;
    }
  }

  /** Release every session and buffer this pipeline owns. */
  async dispose() {
    this.zeroHint.dispose();
    await Promise.allSettled(Object.values(this.sessions).map((s) => s.release()));
    this.sessions = {};
  }

  constructor(meta, sessions, device, loadTimes) {
    this.meta = meta;
    this.H = meta.height;
    this.W = meta.width;
    this.T = meta.frames;
    this.sessions = sessions;
    this.device = device;
    this.loadTimes = loadTimes;
    this.timings = {};
    this.syncTimings = false;  // await GPU completion after each run (profiling)
    this.zeroHint = this.gpuTensor([1, 1, this.H, this.W]);
  }

  /** A zero-initialized float32 GPU tensor owned by this pipeline. */
  gpuTensor(dims) {
    const size = dims.reduce((a, b) => a * b, 1) * 4;
    const buffer = this.device.createBuffer({
      size: Math.ceil(size / 16) * 16,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    return ort.Tensor.fromGpuBuffer(buffer, { dataType: 'float32', dims, dispose: () => buffer.destroy() });
  }

  /** Upload a float32 array as a new GPU tensor. WebGPU reports a wrong-size
   * write only asynchronously, so check the length up front. */
  upload(data, dims) {
    const n = dims.reduce((a, b) => a * b, 1);
    if (!(data instanceof Float32Array) || data.length !== n) {
      throw new TypeError(`expected a Float32Array of ${n} values for [${dims}], got ${data?.constructor?.name} of ${data?.length}`);
    }
    const t = this.gpuTensor(dims);
    this.device.queue.writeBuffer(t.gpuBuffer, 0, data.buffer, data.byteOffset, data.byteLength);
    return t;
  }

  /** Upload a (3, H, W) float32 sRGB frame (padded to the preset size). */
  uploadFrame(chw) {
    return this.upload(chw, [1, 3, this.H, this.W]);
  }

  async run(name, feeds) {
    const t0 = performance.now();
    const out = await this.sessions[name].run(feeds);
    if (this.syncTimings) await this.device.queue.onSubmittedWorkDone();
    const s = (this.timings[name] ??= { calls: 0, ms: 0 });
    s.calls += 1;
    s.ms += performance.now() - t0;
    return out;
  }

  resetTimings() {
    this.timings = {};
  }

  /** Foundation features for one frame; advances the RVM recurrent state. */
  async foundations(frame, state) {
    const done = [];
    try {
      const { cradio } = await this.run('cradio', { frames: frame });
      done.push(cradio);
      const { moge } = await this.run('moge', { frames: frame });
      done.push(moge);
      const { rvm, state_out } = await this.run('rvm', { frame, state });
      return { cradio, moge, rvm, state: state_out };
    } catch (e) {
      done.forEach((t) => t.dispose());
      throw e;
    }
  }

  /**
   * Key a clip. `getFrame(t)` returns a (3, H, W) Float32Array for frame t;
   * `onResult(t, {alpha, fg})` receives GPU tensors (the callback must
   * dispose them). Optional `initialHint` is a (1, H, W) Float32Array matte
   * for frame 0 with `hintQuality` trust.
   */
  async process(numFrames, getFrame, onResult, { initialHint = null, hintQuality = 0.95, carryHint = true, stride = null } = {}) {
    const T = this.T;
    const S = stride ?? Math.max(1, T >> 1);
    if (!Number.isInteger(numFrames) || numFrames < 1) throw new RangeError(`numFrames must be a positive integer, got ${numFrames}`);
    if (!Number.isInteger(S) || S < 1 || S > T) throw new RangeError(`stride must be an integer in [1, ${T}], got ${S}`);
    // Clips shorter than T repeat their last frame to fill the static T-frame
    // window (the CUDA runtime runs a shorter window instead).
    const total = Math.max(numFrames, T);
    const starts = windowStarts(total, T, S);
    const overlap = Math.max(0, T - S);
    const lastWindow = new Int32Array(total);
    starts.forEach((s, i) => { for (let t = s; t < s + T; t++) lastWindow[t] = i; });

    const frames = new Map();   // t -> GPU frame tensor
    const feats = new Map();    // t -> {cradio, moge, rvm}
    const acc = new Map();      // t -> {baton, w}
    const carry = new Map();    // t -> carry hint tensor
    const pending = new Set();  // stage outputs not yet handed off or stored
    let state = null;
    let hint0 = null;
    let loaded = 0;
    const blendScalar = (x) => new ort.Tensor('float32', new Float32Array([x]), [1]);

    try {
      state = this.gpuTensor([1, this.meta.rvm_tokens, this.meta.rvm_dim]);
      if (initialHint) hint0 = this.upload(initialHint, [1, 1, this.H, this.W]);
      for (let i = 0; i < starts.length; i++) {
        const t0 = starts[i];
        while (loaded < t0 + T) {
          const src = Math.min(loaded, numFrames - 1);
          const frame = this.uploadFrame(await getFrame(src));
          frames.set(loaded, frame);
          const f = await this.foundations(frame, state);
          state.dispose();
          state = f.state;
          feats.set(loaded, f);
          loaded++;
        }

        const feeds = {};
        const hq = new Float32Array(T);
        let anyHint = false;
        for (let j = 0; j < T; j++) {
          const t = t0 + j;
          feeds[`frame_${j}`] = frames.get(t);
          feeds[`cradio_${j}`] = feats.get(t).cradio;
          feeds[`moge_${j}`] = feats.get(t).moge;
          feeds[`rvm_${j}`] = feats.get(t).rvm;
          let h = this.zeroHint;
          if (carry.has(t)) { h = carry.get(t); hq[j] = CARRY_HINT_QUALITY; anyHint = true; }
          if (hint0 && t === 0) { h = hint0; hq[j] = hintQuality; anyHint = true; }
          feeds[`hint_${j}`] = h;
        }
        if (!anyHint) hq.fill(-1);
        feeds.hint_quality = new ort.Tensor('float32', hq, [1, T]);
        const out = await this.run('stage1', feeds);
        Object.values(out).forEach((t) => pending.add(t));

        const w = taperWeights(T, overlap, i === 0, i === starts.length - 1);
        const nextStart = i + 1 < starts.length ? starts[i + 1] : total;
        for (let j = 0; j < T; j++) {
          const t = t0 + j;
          const baton = out[`baton_${j}`];
          const prev = acc.get(t);
          if (lastWindow[t] === i) {
            const res = prev
              ? await this.run('stage2', { frame: frames.get(t), baton_a: prev.baton, baton_b: baton, blend: blendScalar(w[j] / (prev.w + w[j])) })
              : await this.run('stage2', { frame: frames.get(t), baton_a: baton, baton_b: baton, blend: blendScalar(0) });
            pending.add(res.alpha).add(res.fg);
            if (prev) { prev.baton.dispose(); acc.delete(t); }
            baton.dispose();
            pending.delete(baton);
            pending.delete(res.alpha);
            pending.delete(res.fg);
            if (t < numFrames) {
              try {
                await onResult(t, res);
              } catch (e) {
                res.alpha.dispose(); res.fg.dispose();
                throw e;
              }
            } else {
              res.alpha.dispose(); res.fg.dispose();
            }
          } else if (prev) {
            const { out: mixed } = await this.run('blend', { a: prev.baton, b: baton, t: blendScalar(w[j] / (prev.w + w[j])) });
            prev.baton.dispose();
            baton.dispose();
            pending.delete(baton);
            acc.set(t, { baton: mixed, w: prev.w + w[j] });
          } else {
            acc.set(t, { baton, w: w[j] });
            pending.delete(baton);
          }
          const c = out[`carry_${j}`];
          pending.delete(c);
          if (carryHint) {
            carry.get(t)?.dispose();
            carry.set(t, c);
          } else {
            c.dispose();
          }
        }
        for (const t of [...carry.keys()]) if (t < nextStart) { carry.get(t).dispose(); carry.delete(t); }
        for (const t of [...frames.keys()]) {
          if (t < nextStart) {
            frames.get(t).dispose();
            frames.delete(t);
            const f = feats.get(t);
            f.cradio.dispose(); f.moge.dispose(); f.rvm.dispose();
            feats.delete(t);
          }
        }
      }
    } finally {
      state?.dispose();
      hint0?.dispose();
      for (const t of pending) t.dispose();
      for (const m of [frames, carry]) for (const v of m.values()) v.dispose();
      for (const f of feats.values()) { f.cradio.dispose(); f.moge.dispose(); f.rvm.dispose(); }
      for (const a of acc.values()) a.baton.dispose();
    }
  }
}
