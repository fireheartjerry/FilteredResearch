// Can the embedding model be DISTILLED into something an extension can ship?
//
// A contextual model reaches relevance nDCG@20 0.638 but costs 22 MB of weights,
// a WASM runtime and 27.7 ms per document. This tests the cheap alternative the
// constraints explicitly allow: embed the vocabulary once, offline, ship the term
// vectors, and build a document vector at runtime as an IDF-weighted average --
// which is exactly what src/shared/semantic.js already does with far worse
// vectors. No runtime dependency, no async, ~5 MB.

import { loadCorpus } from "../lib/adapters.mjs";
import { buildRelevanceQueries } from "../lib/labels.mjs";
import { ndcgAt, mean } from "../lib/metrics.mjs";
import { buildRelevanceIndex, expandQuery, withFeedback, scoreEntry } from "../../src/shared/relevance.js";
import { contentTerms, textOf, buildLexicon, bm25Weights, tokenize } from "../../src/shared/text.js";

const SPIKE_DIR = process.env.FR_SPIKE_DIR;
if (!SPIKE_DIR) { console.log("Set FR_SPIKE_DIR."); process.exit(0); }
const N = SPIKE_DIR.split("\\").join("/");
const t = await import(`file:///${N}/node_modules/@xenova/transformers/src/transformers.js`);
t.env.allowLocalModels = true; t.env.allowRemoteModels = true;
t.env.localModelPath = `${N}/models/`; t.env.cacheDir = `${N}/hfcache`;

const corpus = loadCorpus();
const relevanceCorpus = loadCorpus({ forRelevance: true });
const splitOf = new Map(corpus.rawCandidates.map((w) => [w.id, w.split]));
const docs = relevanceCorpus.candidates.filter((w) => splitOf.get(w.id) === "test");
const ids = new Set(docs.map((w) => w.id));
const queries = buildRelevanceQueries(corpus.rawCandidates, { minimumRelevant: 10, maximumQueries: 60 })
  .map((e) => { const g = new Map([...e.gains].filter(([id]) => ids.has(id))); return { ...e, gains: g, relevantCount: [...g.values()].filter((x) => x >= 2).length }; })
  .filter((e) => e.relevantCount >= 6);

const docTerms = docs.map((w) => contentTerms(textOf(w)));
const lexicon = buildLexicon(docTerms);
const averageLength = docTerms.reduce((s, x) => s + x.length, 0) / docTerms.length;

// Vocabulary to distil: unigrams the corpus actually uses.
const vocabulary = [...lexicon.ids.keys()].filter((term) => !term.includes(" "));
console.log(`${docs.length} docs, ${queries.length} queries, ${vocabulary.length} vocabulary terms`);

const pipe = await t.pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2", { quantized: true });
const started = Date.now();
const termVectors = new Map();
const BATCH = 256;
for (let i = 0; i < vocabulary.length; i += BATCH) {
  const slice = vocabulary.slice(i, i + BATCH);
  const out = (await pipe(slice, { pooling: "mean", normalize: true })).tolist();
  slice.forEach((term, j) => termVectors.set(term, Float32Array.from(out[j])));
  if (i % 2560 === 0) process.stdout.write(`\r  distilled ${i}/${vocabulary.length}`);
}
console.log(`\r  distilled ${vocabulary.length} terms in ${((Date.now() - started) / 1000).toFixed(1)}s`);
console.log(`  shipped size at int8: ${((vocabulary.length * 384) / 1048576).toFixed(1)} MB`);

const DIM = 384;
function documentVector(terms) {
  const out = new Float32Array(DIM);
  const { ids: wids, weights } = bm25Weights(terms, lexicon, averageLength);
  const byId = new Map([...lexicon.ids].map(([term, id]) => [id, term]));
  for (let i = 0; i < wids.length; i += 1) {
    const vec = termVectors.get(byId.get(wids[i]));
    if (!vec) continue;
    for (let d = 0; d < DIM; d += 1) out[d] += weights[i] * vec[d];
  }
  let n = 0; for (let d = 0; d < DIM; d += 1) n += out[d] * out[d];
  n = Math.sqrt(n); if (n) for (let d = 0; d < DIM; d += 1) out[d] /= n;
  return { v: out, n };
}
const docVecs = docTerms.map(documentVector);
const dot = (a, b) => { if (!a.n || !b.n) return 0; let s = 0; for (let d = 0; d < DIM; d += 1) s += a.v[d] * b.v[d]; return s; };

const index = buildRelevanceIndex(docs, { space: null });
const lexRank = queries.map((entry) => {
  const expanded = withFeedback(index, expandQuery(entry.query, index));
  return new Map(index.entries.map((e) => { const r = scoreEntry(e, expanded, index); return [e.work.id, r.raw > 0 ? r.rank : -50]; }));
});
const queryVecs = queries.map((q) => documentVector(contentTerms(q.query)));

for (const lambda of [0, 0.25, 0.5, 0.75, 1]) {
  const per = queries.map((entry, qi) => {
    const lex = lexRank[qi];
    const ranked = docs
      .map((w, di) => ({ w, s: (1 - lambda) * (lex.get(w.id) ?? -50) + lambda * 12 * dot(queryVecs[qi], docVecs[di]) }))
      .sort((a, b) => b.s - a.s).map((x) => x.w);
    return ndcgAt(ranked, (w) => entry.gains.get(w.id) || 0, 20);
  });
  const label = lambda === 0 ? "lexical only (ships today)" : lambda === 1 ? "distilled vectors only" : `hybrid lambda=${lambda}`;
  console.log(`${label.padEnd(30)} nDCG@20 ${mean(per).toFixed(4)}`);
}
