// Run bench.html in Chrome with WebGPU and print the results as JSON.
//
//   node bench.mjs --preset 896x512 --dtype fp16 [--frames 24] [--out results.json]
//                  [--dtypes '{"stage2":"q8"}'] [--ep '{"enableMatmulFp32Accumulation":true}']
//   node bench.mjs --page micro.html [--query cases=a,b]   (op microbenchmarks)
//
// Headless Chrome on Linux only exposes the SwiftShader (CPU) WebGPU adapter,
// so this launches a headed browser with an off-screen window on the X
// display in $BENCH_DISPLAY (default :1, i.e. Xwayland), with the flags that
// expose the discrete GPU and, on NVIDIA, shader-f16.

import { chromium } from 'playwright-core';
import { writeFileSync } from 'node:fs';
import { startServer } from './serve.mjs';

const argv = process.argv.slice(2);
const opt = (name, dflt) => {
  const i = argv.indexOf(`--${name}`);
  return i >= 0 ? argv[i + 1] : dflt;
};
const preset = opt('preset', '896x512');
const dtype = opt('dtype', 'fp16');
const frames = opt('frames', '24');
const dtypes = opt('dtypes', '{}');
const ep = opt('ep', '{}');
const golden = opt('golden', 'golden');
const seq = opt('seq', '');
const stride = opt('stride', '');
const exe = opt('exe', '/usr/bin/google-chrome-stable');
const out = opt('out', null);
const timeoutMin = Number(opt('timeout-min', '12'));
const port = Number(opt('port', '8799'));
const pageName = opt('page', 'bench.html');
const extra = opt('query', '');

const server = await startServer(port);
const browser = await chromium.launch({
  executablePath: exe,
  headless: false,
  env: { ...process.env, DISPLAY: process.env.BENCH_DISPLAY ?? ':1' },
  args: [
    '--enable-unsafe-webgpu',
    '--ignore-gpu-blocklist',
    '--enable-features=Vulkan',
    '--ozone-platform=x11',
    '--enable-dawn-features=vulkan_enable_f16_on_nvidia',
    '--window-position=-2400,-2400',
    '--window-size=900,700',
  ],
});
let result;
try {
  const page = await browser.newPage();
  page.on('console', (m) => process.stderr.write(`[page] ${m.text().slice(0, 400)}\n`));
  page.on('pageerror', (e) => process.stderr.write(`[pageerror] ${e}\n`));
  const qs = new URLSearchParams({ preset, dtype, frames, dtypes, ep, golden, ...(seq ? { seq } : {}), ...(stride ? { stride } : {}) });
  await page.goto(`http://localhost:${port}/${pageName}?${qs}${extra ? '&' + extra : ''}`);
  await page.waitForFunction(() => window.__bench !== undefined, null, { timeout: timeoutMin * 60e3, polling: 1000 });
  result = await page.evaluate(() => window.__bench);
} finally {
  await browser.close();
  server.close();
}
const json = JSON.stringify(result, null, 1);
if (out) writeFileSync(out, json);
console.log(json);
process.exit(result?.error ? 1 : 0);
