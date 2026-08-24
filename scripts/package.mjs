import { cp, mkdir, readFile, rm, writeFile, access } from "node:fs/promises";
import { resolve } from "node:path";

const root = resolve(import.meta.dirname, "..");
const destination = resolve(root, "dist", "filteredresearch-extension");
await rm(destination, { recursive: true, force: true });
await mkdir(destination, { recursive: true });

for (const entry of [
  "manifest.json",
  "src",
  "assets",
  "LICENSE",
  "PRIVACY.md",
  "COMPLIANCE.md",
]) {
  await cp(resolve(root, entry), resolve(destination, entry), { recursive: true });
}

// Store artwork and the icon master live in assets/ for convenience but must not
// ship inside the extension; they are dead weight in every user's download.
for (const extra of ["promo-small-440x280.jpg", "promo-marquee-1400x560.jpg", "icon-source.svg"]) {
  await rm(resolve(destination, "assets", extra), { force: true });
}

// The sentence-embedding model, if it has been fetched. It is not committed --
// run `npm run fetch-model` first -- and the extension degrades to lexical
// relevance ranking without it, so a package built without the weights is a
// smaller, working extension rather than a broken one.
let modelBytes = 0;
try {
  await access(resolve(root, "vendor", "Xenova", "all-MiniLM-L6-v2", "onnx", "model_quantized.onnx"));
  await cp(resolve(root, "vendor"), resolve(destination, "vendor"), { recursive: true });
  const { statSync } = await import("node:fs");
  modelBytes = statSync(resolve(destination, "vendor", "Xenova", "all-MiniLM-L6-v2", "onnx", "model_quantized.onnx")).size;
  process.stdout.write(`Included embedding model (${(modelBytes / 1048576).toFixed(1)} MB)\n`);
} catch {
  process.stdout.write("No embedding model found; packaging lexical-only build (npm run fetch-model to include it)\n");
}

const manifestPath = resolve(destination, "manifest.json");
const manifest = JSON.parse(await readFile(manifestPath, "utf8"));
await writeFile(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`, "utf8");

process.stdout.write(`Staged load-unpacked extension at ${destination}\n`);
