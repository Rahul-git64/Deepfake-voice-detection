import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PORT = Number(process.env.PORT || 5173);
const BACKEND_HOST = '127.0.0.1';
const BACKEND_PORT = 8080;

const server = http.createServer((req, res) => {
  const url = req.url || '/';

  // Proxy API and sample requests to FastAPI on port 8080
  if (url.startsWith('/predict') || url.startsWith('/health') || url.startsWith('/samples') || url.startsWith('/legacy') || url.startsWith('/docs')) {
    const proxyReq = http.request(
      {
        hostname: BACKEND_HOST,
        port: BACKEND_PORT,
        path: url,
        method: req.method,
        headers: req.headers,
      },
      (proxyRes) => {
        res.writeHead(proxyRes.statusCode || 200, proxyRes.headers);
        proxyRes.pipe(res, { end: true });
      }
    );
    proxyReq.on('error', (err) => {
      res.writeHead(502, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ detail: `Backend proxy error: ${err.message}` }));
    });
    req.pipe(proxyReq, { end: true });
    return;
  }

  // Serve Verity frontend index.html
  const indexPath = path.join(__dirname, 'index.html');
  fs.readFile(indexPath, 'utf8', (err, content) => {
    if (err) {
      res.writeHead(500, { 'Content-Type': 'text/plain' });
      res.end('Error loading frontend/index.html');
      return;
    }
    res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' });
    res.end(content);
  });
});

server.listen(PORT, '127.0.0.1', () => {
  console.log(`Verity Frontend Dev Host running at http://localhost:${PORT} (proxying API to http://127.0.0.1:${BACKEND_PORT})`);
});
