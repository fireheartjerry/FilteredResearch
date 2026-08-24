// npm run bench -- wall-clock and peak memory for a production-shaped pass.
//
// Offline: builds a 10,000-candidate corpus by resampling the cached fixtures, so
// the text statistics stay realistic rather than degenerating into synthetic
// noise that every algorithm finds trivially easy.
//
// Each ranker is measured in its OWN process. Resident memory never shrinks once
// the OS has handed it over, so running both in one process charged the second
// ranker for everything the first had already allocated -- which made whichever
// ran second look roughly twice as hungry as it was, regardless of which one it
// was.

import { execFileSync } from "node:child_process";
import { writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { loadCorpus } from "./lib/adapters.mjs";
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
    // Distinct ids, identical text: the scorer must do the same amount of work it
    // would on a real corpus of this size.
    out.push({ ...source, id: `${source.id}_b${counter}` });
  }
  return out;
}

// The child: build the corpus, score it once, report. Kept in this file so there
// is one place to read rather than two.
async function measureInThisProcess(which) {
  const corpus = loadCorpus();
  const random = seededRandom(4242);
  const candidates = grow(corpus.candidates, TARGET_CANDIDATES, random);
  const peers = grow(corpus.peers, PEERS, random);
  const module = which === "baseline"
    ? await import("./baseline/scoring.js")
    : await import("../src/shared/scoring.js");

  if (global.gc) global.gc();
  let peak = process.memoryUsage().rss;
  const watch = setInterval(() => {
    peak = Math.max(peak, process.memoryUsage().rss);
  }, 25);
  const started = process.hrtime.bigint();
  const scored = module.scoreBatch(candidates, peers, corpus.authors);
  const elapsed = Number(process.hrtime.bigint() - started) / 1e6;
  clearInterval(watch);
  peak = Math.max(peak, process.memoryUsage().rss);

  process.stdout.write(`__RESULT__${JSON.stringify({
    name: which === "baseline" ? "v1.0.1 baseline" : "current",
    wallClockMs: Math.round(elapsed),
    peakRssMb: Math.round(peak / 1024 / 1024),
    scored: scored.length,
  })}\n`);
}

function measureInChild(which) {
  const output = execFileSync(
    process.execPath,
    ["--expose-gc", join(HERE, "bench.mjs"), which],
    { encoding: "utf8", timeout: 1_800_000, maxBuffer: 32 * 1024 * 1024 },
  );
  const line = output.split("\n").find((row) => row.startsWith("__RESULT__"));
  if (!line) throw new Error(`bench child produced no result:\n${output.slice(0, 400)}`);
  return JSON.parse(line.slice("__RESULT__".length));
}

async function main() {
  const which = process.argv[2];
  if (which === "baseline" || which === "current") {
    await measureInThisProcess(which);
    return;
  }

  console.log(`Benchmarking ${TARGET_CANDIDATES} candidates against ${PEERS} peers, one process each...\n`);
  const results = [measureInChild("baseline"), measureInChild("current")];
  const [baseline, current] = results;
  const ratio = current.wallClockMs / Math.max(1, baseline.wallClockMs);
  const memoryRatio = current.peakRssMb / Math.max(1, baseline.peakRssMb);

  for (const result of results) {
    console.log(`${result.name.padEnd(16)} ${String(result.wallClockMs).padStart(7)} ms   peak RSS ${result.peakRssMb} MB`);
  }
  console.log(`\n  wall-clock ratio vs baseline: ${ratio.toFixed(2)}x`);
  console.log(`  peak memory ratio:            ${memoryRatio.toFixed(2)}x\n`);

  writeFileSync(join(HERE, "bench.json"), `${JSON.stringify({
    generatedAt: new Date().toISOString(),
    candidates: TARGET_CANDIDATES,
    peers: PEERS,
    isolatedProcesses: true,
    results,
    ratio: Number(ratio.toFixed(3)),
    memoryRatio: Number(memoryRatio.toFixed(3)),
  }, null, 2)}\n`);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
