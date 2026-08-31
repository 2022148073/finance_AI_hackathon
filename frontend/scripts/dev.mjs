import "./build.mjs";
import { createReadStream, existsSync, statSync } from "node:fs";
import { createServer } from "node:http";
import { extname, isAbsolute, join, normalize, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";


const frontendDir = resolve(fileURLToPath(new URL("..", import.meta.url)));
const distDir = resolve(frontendDir, "dist");
const port = Number(process.env.PORT ?? 5173);
const mimeTypes = {
  ".css": "text/css; charset=utf-8",
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".svg": "image/svg+xml",
};

createServer((request, response) => {
  let requested = "/";
  try {
    requested = decodeURIComponent((request.url ?? "/").split("?")[0]);
  } catch {
    response.writeHead(400).end("Bad Request");
    return;
  }
  const relativeRequest = normalize(requested).replace(/^[/\\]+/, "");
  let target = resolve(join(distDir, relativeRequest || "index.html"));
  const pathFromDist = relative(distDir, target);
  if (
    pathFromDist.startsWith("..") ||
    isAbsolute(pathFromDist) ||
    !existsSync(target) ||
    statSync(target).isDirectory()
  ) {
    target = resolve(distDir, "index.html");
  }
  response.setHeader("Content-Type", mimeTypes[extname(target)] ?? "application/octet-stream");
  response.setHeader("X-Content-Type-Options", "nosniff");
  response.setHeader("X-Frame-Options", "DENY");
  response.setHeader("Referrer-Policy", "no-referrer");
  response.setHeader(
    "Content-Security-Policy",
    "default-src 'self'; connect-src 'self' http://127.0.0.1:8000 http://localhost:8000; img-src 'self'; style-src 'self'; script-src 'self'",
  );
  response.setHeader("Cache-Control", "no-store");
  createReadStream(target).pipe(response);
}).listen(port, "127.0.0.1", () => {
  console.log(`Frontend: http://127.0.0.1:${port}`);
});
