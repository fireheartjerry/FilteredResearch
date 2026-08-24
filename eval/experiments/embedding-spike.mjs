// Feasibility spike: would a BUNDLED local embedding model clear the relevance
// bar that five bag-of-words method families could not?
//
// This commits nothing. It measures whether the ~10 points behind C3 are real
// before anyone pays 22 MB of extension size and an inference dependency for them.
// Model weights live outside the repo; nothing here ships.

import { loadCorpus } from "../lib/adapters.mjs";
import { buildRelevanceQueries } from "../lib/labels.mjs";
import { ndcgAt, mean } from "../lib/metrics.mjs";
import { buildRelevanceIndex, expandQuery, withFeedback, scoreEntry } from "../../src/shared/relevance.js";

// Point FR_SPIKE_DIR at a directory holding `node_modules/@xenova/transformers`
// and `models/Xenova/all-MiniLM-L6-v2/` (config.json, tokenizer.json,
// tokenizer_config.json, onnx/model_quantized.onnx from Hugging Face).
// Deliberately outside the repo: none of it ships, and none of it is committed.
const SPIKE_DIR = process.env.FR_SPIKE_DIR;
if (!SPIKE_DIR) {
  console.log("Set FR_SPIKE_DIR to a directory with @xenova/transformers and the MiniLM weights.");
  console.log("See docs/criteria/algorithm-criteria.md, criterion C3, for what this measured.");
  process.exit(0);
}
const NORMALISED = SPIKE_DIR.split("\\").join("/");
const SPIKE = NORMALISED.startsWith("file:") ? NORMALISED : `file:///${NORMALISED}`;
const transformers = await import(`${SPIKE}/node_modules/@xenova/transformers/src/transformers.js`);
transformers.env.allowLocalModels = true;
transformers.env.allowRemoteModels = true;
transformers.env.localModelPath = `${NORMALISED}/models/`;
transformers.env.cacheDir = `${NORMALISED}/hfcache`;

const corpus = loadCorpus();
const relevanceCorpus = loadCorpus({ forRelevance: true });
const splitOf = new Map(corpus.rawCandidates.map((w) => [w.id, w.split]));
const docs = relevanceCorpus.candidates.filter((w) => splitOf.get(w.id) === "test");
const ids = new Set(docs.map((w) => w.id));
const queries = buildRelevanceQueries(corpus.rawCandidates, { minimumRelevant: 10, maximumQueries: 60 })
  .map((e) => { const g = new Map([...e.gains].filter(([id]) => ids.has(id))); return { ...e, gains: g, relevantCount: [...g.values()].filter((x) => x >= 2).length }; })
  .filter((e) => e.relevantCount >= 6);

console.log(`${docs.length} documents, ${queries.length} queries`);

const pipe = await transformers.pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2", { quantized: true });
const textOf = (w) => `${w.title || ""}. ${String(w.abstract || "").slice(0, 900)}`;

const started = Date.now();
const vectors = [];
const BATCH = 32;
for (let i = 0; i < docs.length; i += BATCH) {
  const out = await pipe(docs.slice(i, i + BATCH).map(textOf), { pooling: "mean", normalize: true });
  vectors.push(...out.tolist());
  if (i % 320 === 0) process.stdout.write(`\r  embedded ${i}/${docs.length}`);
}
console.log(`\r  embedded ${docs.length} docs in ${((Date.now() - started) / 1000).toFixed(1)}s`);

const queryVectors = (await pipe(queries.map((q) => q.query), { pooling: "mean", normalize: true })).tolist();
const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };

// lexical reference, exactly what ships today
const index = buildRelevanceIndex(docs, { space: null });
const lexicalRank = queries.map((entry) => {
  const expanded = withFeedback(index, expandQuery(entry.query, index));
  const scored = index.entries.map((e) => ({ id: e.work.id, s: scoreEntry(e, expanded, index).rank, raw: scoreEntry(e, expanded, index).raw }));
  const map = new Map(scored.map((x) => [x.id, x.raw > 0 ? x.s : -50]));
  return map;
});

for (const lambda of [0, 0.25, 0.5, 0.75, 1]) {
  const per = queries.map((entry, qi) => {
    const lex = lexicalRank[qi];
    const ranked = docs
      .map((w, di) => ({ w, s: (1 - lambda) * (lex.get(w.id) ?? -50) + lambda * 12 * dot(queryVectors[qi], vectors[di]) }))
      .sort((a, b) => b.s - a.s)
      .map((x) => x.w);
    return ndcgAt(ranked, (w) => entry.gains.get(w.id) || 0, 20);
  });
  const label = lambda === 0 ? "lexical only (ships today)" : lambda === 1 ? "embeddings only" : `hybrid lambda=${lambda}`;
  console.log(`${label.padEnd(30)} nDCG@20 ${mean(per).toFixed(4)}`);
}
