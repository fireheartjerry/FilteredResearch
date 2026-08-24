// node scripts/fetch-model.mjs
//
// Downloads the sentence-embedding model the relevance ranker uses. Weights are
// NOT committed: they are fetched once here and packaged by scripts/package.mjs.
// Nothing is downloaded at runtime -- the extension ships the files it needs and
// makes no network call for them.

import { mkdirSync, writeFileSync, existsSync, statSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
// Laid out as <vendor>/<repo>/ because that is where the loader looks:
// it resolves localModelPath + the model id.
const TARGET = join(HERE, "..", "vendor", "Xenova", "all-MiniLM-L6-v2");
const REPOSITORY = "Xenova/all-MiniLM-L6-v2";
const BASE = `https://huggingface.co/${REPOSITORY}/resolve/main/`;
const FILES = ["config.json", "tokenizer.json", "tokenizer_config.json", "onnx/model_quantized.onnx"];

// The inference runtime ships too. It is a build-time copy out of node_modules
// rather than a committed blob, and it is what lets the extension run the model
// without fetching anything: Manifest V3 forbids remote code, and this is local
// code by the time Chrome sees it.
async function vendorRuntime() {
  const { cp, mkdir } = await import("node:fs/promises");
  const source = join(HERE, "..", "node_modules", "@xenova", "transformers");
  const destination = join(TARGET, "..", "..", "transformers");
  if (!existsSync(source)) {
    console.log("  transformers runtime not installed; run: npm install @xenova/transformers");
    console.log("  (the extension falls back to lexical relevance ranking without it)");
    return;
  }
  // Only what actually runs. The published package carries source maps, an
  // unminified build and four WASM variants; shipping all of it costs 44 MB for
  // roughly 9 MB of use. The single-threaded non-SIMD binary is the one a service
  // worker can rely on.
  await mkdir(join(destination, "dist"), { recursive: true });
  const needed = ["transformers.min.js", "ort-wasm.wasm", "ort-wasm-simd.wasm"];
  for (const file of needed) {
    const from = join(source, "dist", file);
    if (existsSync(from)) await cp(from, join(destination, "dist", file));
  }
  await cp(join(source, "package.json"), join(destination, "package.json"));
  console.log("  vendored the inference runtime (minified build, 2 WASM binaries)");
}

async function main() {
  mkdirSync(join(TARGET, "onnx"), { recursive: true });
  for (const file of FILES) {
    const destination = join(TARGET, file);
    if (existsSync(destination) && statSync(destination).size > 0) {
      console.log(`  have ${file}`);
      continue;
    }
    const response = await fetch(BASE + file);
    if (!response.ok) throw new Error(`${file}: HTTP ${response.status}`);
    const bytes = Buffer.from(await response.arrayBuffer());
    writeFileSync(destination, bytes);
    console.log(`  ${file} ${(bytes.length / 1048576).toFixed(2)} MB`);
  }
  await vendorRuntime();
  console.log(`\nModel ready in vendor/${REPOSITORY}, runtime in vendor/transformers.`);
}

main().catch((error) => {
  console.error(error.message);
  process.exit(1);
});
