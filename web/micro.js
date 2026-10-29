// Single-op WebGPU timings for graphs from onnx_export/micro_graphs.py.
// micro.html?cases=a,b (default: all in models/micro/manifest.json)

import * as ort from 'onnxruntime-web/webgpu';
import { checkWebGPU } from './corridorkey.js';

const q = new URLSearchParams(location.search);
const logEl = document.getElementById('log');
const log = (...a) => {
  const s = a.map((x) => (typeof x === 'string' ? x : JSON.stringify(x))).join(' ');
  console.log(s);
  logEl.textContent += s + '\n';
};

async function timeCase(name, spec, device) {
  const session = await ort.InferenceSession.create(`models/micro/${name}/onnx/model.onnx`, {
    executionProviders: [{ name: 'webgpu', device }],
    preferredOutputLocation: 'gpu-buffer',
    graphOptimizationLevel: 'all',
  });
  // GPU-resident inputs, written once, so timings exclude uploads.
  const feeds = {};
  const buffers = [];
  for (const [k, dims] of Object.entries(spec.inputs)) {
    const n = dims.reduce((a, b) => a * b, 1);
    const data = new Uint16Array(n + (n & 1));
    for (let i = 0; i < n; i++) data[i] = 0x3c00 ^ (i & 0x3ff);  // fp16 values around 1
    const buf = device.createBuffer({ size: Math.ceil(data.byteLength / 16) * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    device.queue.writeBuffer(buf, 0, data);
    buffers.push(buf);
    feeds[k] = ort.Tensor.fromGpuBuffer(buf, { dataType: 'float16', dims });
  }
  // Back-to-back runs with one sync at the end, so the per-run submit/sync
  // round trip (~2.5 ms here) is amortized and kernel time dominates.
  const burst = async (n) => {
    const outs = [];
    for (let i = 0; i < n; i++) outs.push(await session.run(feeds));
    await device.queue.onSubmittedWorkDone();
    for (const out of outs) for (const t of Object.values(out)) t.dispose();
  };
  await burst(3);
  const n = 30;
  const t0 = performance.now();
  await burst(n);
  const ms = (performance.now() - t0) / n;
  await session.release();
  buffers.forEach((b) => b.destroy());
  return { ms: +ms.toFixed(3), tflops: spec.flops ? +(spec.flops / ms / 1e9).toFixed(2) : undefined };
}

async function main() {
  const result = {};
  try {
    const { device, info } = await checkWebGPU('fp16');
    result.adapter = info;
    const manifest = await (await fetch('models/micro/manifest.json')).json();
    const cases = q.get('cases') ? q.get('cases').split(',') : Object.keys(manifest);
    result.cases = {};
    for (const name of cases) {
      result.cases[name] = await timeCase(name, manifest[name], device);
      log(name, result.cases[name]);
    }
  } catch (e) {
    result.error = String(e?.stack ?? e);
    log('ERROR', result.error);
  }
  window.__bench = result;
}

main();
