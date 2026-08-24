// npm run bench -- wall-clock and peak memory for a production-shaped pass.
// Offline: builds a 10,000-candidate corpus by resampling the cached fixtures,
// so the text statistics stay realistic rather than degenerating into synthetic
// noise that every algorithm finds trivially easy.

import { writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { loadCorpus, baselineAdapter, currentAdapter } from "./lib/adapters.mjs";
import { seededRandom } from "./lib/metrics.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const TARGET_CANDIDATES = Number(process.env.FR_BENCH_N || 10_000);
const PEERS = 3_200;

function grow(works, target, random) {
  const out = [];
  let counter = 0;
  while (out.length < target) {
    const source = works[Math.floor(random() * works.length)];
    counter += 1;
    // Distinct ids, identical text: the scorer must do the same amount of work
    // it would on a real corpus of this size.
    out.push({ ...source, id: `${source.id}_b${counter}` });
  }
  return out;
}

async function measure(adapter, candidates, peers, authors) {
  if (global.gc) global.gc();
  const before = process.memoryUsage().rss;
  let peak = before;
  const watch = setInterval(() => {
    peak = Math.max(peak, process.memoryUsage().rss);
  }, 25);
  const started = process.hrtime.bigint();
  const scored = adapter.scoreBatch(candidates, peers, authors);
  const elapsed = Number(process.hrtime.bigint() - started) / 1e6;
  clearInterval(watch);
  peak = Math.max(peak, process.memoryUsage().rss);
  return {
    name: adapter.name,
    wallClockMs: Math.round(elapsed),
    msPerThousand: Math.round((elapsed / candidates.length) * 1000),
    peakRssMb: Math.round(peak / 1024 / 1024),
    deltaRssMb: Math.round((peak - before) / 1024 / 1024),
    scored: scored.length,
  };
}

async function main() {
  const corpus = loadCorpus();
  const random = seededRandom(4242);
  const candidates = grow(corpus.candidates, TARGET_CANDIDATES, random);
  const peers = grow(corpus.peers, PEERS, random);
  console.log(`Benchmarking ${candidates.length} candidates against ${peers.length} peers...\n`);

  const results = [];
  results.push(await measure(await baselineAdapter(), candidates, peers, corpus.authors));
  results.push(await measure(await currentAdapter(), candidates, peers, corpus.authors));

  const [baseline, current] = results;
  const ratio = current.wallClockMs / Math.max(1, baseline.wallClockMs);
  const memoryRatio = current.peakRssMb / Math.max(1, baseline.peakRssMb);

  for (const result of results) {
    console.log(`${result.name.padEnd(16)} ${String(result.wallClockMs).padStart(7)} ms   peak RSS ${result.peakRssMb} MB`);
  }
  console.log(`\n  wall-clock ratio vs baseline: ${ratio.toFixed(2)}x`);
  console.log(`  peak memory ratio:            ${memoryRatio.toFixed(2)}x\n`);

  const report = { generatedAt: new Date().toISOString(), candidates: candidates.length, peers: peers.length, results, ratio: Number(ratio.toFixed(3)), memoryRatio: Number(memoryRatio.toFixed(3)) };
  writeFileSync(join(HERE, "bench.json"), `${JSON.stringify(report, null, 2)}\n`);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
