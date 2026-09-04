import { build } from "esbuild";
import { cp, mkdir, readFile, rm, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";


const scriptDir = dirname(fileURLToPath(import.meta.url));
const frontendDir = resolve(scriptDir, "..");
const distDir = resolve(frontendDir, "dist");
const assetsDir = resolve(distDir, "assets");
const publicDir = resolve(frontendDir, "public");

await rm(distDir, { recursive: true, force: true });
await mkdir(assetsDir, { recursive: true });
await cp(publicDir, distDir, { recursive: true });

const sourceHtml = await readFile(resolve(frontendDir, "index.html"), "utf8");
const productionHtml = sourceHtml
  .replace(
    '<script type="module" src="/src/main.jsx"></script>',
    '<script type="module" src="/assets/app.js"></script>',
  )
  .replace(
    "</head>",
    '    <link rel="stylesheet" href="/assets/app.css" />\n  </head>',
  );
await writeFile(resolve(distDir, "index.html"), productionHtml, "utf8");

await build({
  entryPoints: [resolve(frontendDir, "src", "main.jsx")],
  outfile: resolve(assetsDir, "app.js"),
  bundle: true,
  minify: true,
  sourcemap: false,
  target: ["es2020"],
  jsx: "automatic",
  legalComments: "none",
});

console.log(`Frontend build completed: ${distDir}`);
