// MiniLM is a general-purpose sentence encoder. Retrieval-specific models of the
// same size are usually better at exactly this task, so before settling on the
// 22 MB already paid for, the alternatives get measured on the same queries.
import { loadCorpus } from "../lib/adapters.mjs";
import { buildRelevanceQueries } from "../lib/labels.mjs";
import { ndcgAt, mean } from "../lib/metrics.mjs";

const corpus = loadCorpus();
const relevanceCorpus = loadCorpus({ forRelevance: true });
const splitOf = new Map(corpus.rawCandidates.map((w) => [w.id, w.split]));
const docs = relevanceCorpus.candidates.filter((w) => splitOf.get(w.id) === "test");
const ids = new Set(docs.map((w) => w.id));
const queries = buildRelevanceQueries(corpus.rawCandidates, { minimumRelevant: 10, maximumQueries: 60 })
  .map((e) => { const g = new Map([...e.gains].filter(([id]) => ids.has(id))); return { ...e, gains: g, relevantCount: [...g.values()].filter((x) => x >= 2).length }; })
  .filter((e) => e.relevantCount >= 6);

const t = await import("@xenova/transformers");
t.env.allowRemoteModels = true;
t.env.allowLocalModels = false;
t.env.cacheDir = process.env.FR_MODEL_CACHE || "./vendor/.cache";

const text = (w) => `${w.title || ""}. ${String(w.abstract || "").slice(0, 900)}`;
const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };

// Some retrieval models expect an instruction prefix on the query side.
const MODELS = [
  ["Xenova/all-MiniLM-L6-v2", (q) => q],
  ["Xenova/bge-small-en-v1.5", (q) => `Represent this sentence for searching relevant passages: ${q}`],
  ["Xenova/gte-small", (q) => q],
  ["Xenova/e5-small-v2", (q) => `query: ${q}`],
];

for (const [id, queryPrefix] of MODELS) {
  try {
    const started = Date.now();
    const pipe = await t.pipeline("feature-extraction", id, { quantized: true });
    const vecs = [];
    for (let i = 0; i < docs.length; i += 32) {
      const batch = docs.slice(i, i + 32).map((w) => (id.includes("e5") ? `passage: ${text(w)}` : text(w)));
      vecs.push(...(await pipe(batch, { pooling: "mean", normalize: true })).tolist());
    }
    const qv = (await pipe(queries.map((q) => queryPrefix(q.query)), { pooling: "mean", normalize: true })).tolist();
    const per = queries.map((entry, qi) => {
      const ranked = docs.map((w, di) => ({ w, s: dot(qv[qi], vecs[di]) })).sort((a, b) => b.s - a.s).map((x) => x.w);
      return ndcgAt(ranked, (w) => entry.gains.get(w.id) || 0, 20);
    });
    console.log(`${id.padEnd(30)} nDCG@20 ${mean(per).toFixed(4)}   ${((Date.now() - started) / 1000).toFixed(0)}s`);
  } catch (error) {
    console.log(`${id.padEnd(30)} unavailable: ${String(error.message).slice(0, 60)}`);
  }
}
