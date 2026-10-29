// Static dev server for the CorridorKey web demo.
//
// Serves this directory (pages, node_modules, models/) with the cross-origin
// isolation headers multithreaded WASM needs, and correct MIME types.
//
//   node serve.mjs [--port 8787]

import { createReadStream, statSync } from 'node:fs';
import { createServer } from 'node:http';
import { extname, join, normalize, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = resolve(fileURLToPath(new URL('.', import.meta.url)));
const portArg = process.argv.indexOf('--port');
const PORT = portArg > 0 ? Number(process.argv[portArg + 1]) : 8787;

const TYPES = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.wasm': 'application/wasm',
  '.onnx': 'application/octet-stream',
  '.bin': 'application/octet-stream',
  '.png': 'image/png',
  '.mp4': 'video/mp4',
  '.webm': 'video/webm',
  '.map': 'application/json',
};

export function startServer(port = PORT) {
  const server = createServer((req, res) => {
    const url = new URL(req.url, 'http://localhost');
    let path = normalize(join(ROOT, decodeURIComponent(url.pathname)));
    if (path !== ROOT && !path.startsWith(ROOT + sep)) {
      res.writeHead(403).end();
      return;
    }
    let st;
    try {
      st = statSync(path);
      if (st.isDirectory()) {
        path = join(path, 'index.html');
        st = statSync(path);
      }
    } catch {
      res.writeHead(404).end('not found');
      return;
    }
    const ext = extname(path).toLowerCase();
    const type = TYPES[ext] ?? 'application/octet-stream';  // .onnx_data chunks too
    res.writeHead(200, {
      'Content-Type': type,
      'Content-Length': st.size,
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
      'Cross-Origin-Resource-Policy': 'same-origin',
      'Cache-Control': ext === '.html' || ext === '.js' || ext === '.mjs' ? 'no-cache' : 'max-age=3600',
    });
    if (req.method === 'HEAD') {
      res.end();
      return;
    }
    createReadStream(path).pipe(res);
  });
  return new Promise((ok) => server.listen(port, '127.0.0.1', () => ok(server)));
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  await startServer(PORT);
  console.log(`CorridorKey web demo: http://localhost:${PORT}/`);
}
