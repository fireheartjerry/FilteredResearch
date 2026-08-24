// Does a stronger semantic space (PPMI, AUC 0.73 vs random indexing's 0.67) close
// the relevance gap toward the 0.659 an LLM reaches on the same pool?
import { loadCorpus } from "../lib/adapters.mjs";
import { buildRelevanceQueries } from "../lib/labels.mjs";
import { ndcgAt, mean } from "../lib/metrics.mjs";
import { buildRelevanceIndex, expandQuery, withFeedback, scoreEntry } from "../../src/shared/relevance.js";
import { contentTerms, textOf, buildLexicon, tokenize } from "../../src/shared/text.js";
import { documentVector, removeCommonComponent, semanticSimilarity } from "../../src/shared/semantic.js";

const corpus = loadCorpus();
const relevanceCorpus = loadCorpus({ forRelevance: true });
const splitOf = new Map(corpus.rawCandidates.map((w) => [w.id, w.split]));
const docs = relevanceCorpus.candidates.filter((w) => splitOf.get(w.id) === "tune");
const ids = new Set(docs.map((w) => w.id));
const queries = buildRelevanceQueries(corpus.rawCandidates, { minimumRelevant: 10, maximumQueries: 60 })
  .map((e) => { const g = new Map([...e.gains].filter(([id]) => ids.has(id))); return { ...e, gains: g, relevantCount: [...g.values()].filter((x) => x >= 2).length }; })
  .filter((e) => e.relevantCount >= 6);

const terms = docs.map((w) => contentTerms(textOf(w)));
const lex = buildLexicon(terms);
const avg = terms.reduce((s, t) => s + t.length, 0) / terms.length;

const DIM = 128, WIN = 4;
const vocab = new Map();
for (const t of lex.ids.keys()) if (!t.includes(" ")) vocab.set(t, vocab.size);
const V = vocab.size;
const single = new Float64Array(V);
const pairs = new Map();
let totalPairs = 0;
for (const w of docs) {
  const toks = tokenize(textOf(w), 400).filter((t) => !t.includes("-")).map((t) => vocab.get(t) ?? -1);
  for (let p = 0; p < toks.length; p++) {
    if (toks[p] < 0) continue;
    single[toks[p]] += 1;
    for (let q = Math.max(0, p - WIN); q <= Math.min(toks.length - 1, p + WIN); q++) {
      if (q === p || toks[q] < 0) continue;
      const key = toks[p] * V + toks[q];
      pairs.set(key, (pairs.get(key) || 0) + 1); totalPairs += 1;
    }
  }
}
function hash(n, salt) { let h = (n ^ salt) >>> 0; h = Math.imul(h, 2246822519) >>> 0; h ^= h >>> 13; h = Math.imul(h, 3266489917) >>> 0; return h >>> 0; }
let vectors = new Float32Array(V * DIM);
for (let i = 0; i < V * DIM; i++) vectors[i] = ((hash(i, 0x9e3779b9) & 1023) / 511.5) - 1;
function project(input) {
  const out = new Float32Array(V * DIM);
  for (const [key, count] of pairs) {
    const a = Math.floor(key / V), b = key % V;
    const pmi = Math.log((count * totalPairs) / Math.max(1, single[a] * single[b]));
    if (!(pmi > 0)) continue;
    const ab = a * DIM, bb = b * DIM;
    for (let d = 0; d < DIM; d++) out[ab + d] += pmi * input[bb + d];
  }
  for (let i = 0; i < V; i++) { const b2 = i * DIM; let n = 0; for (let d = 0; d < DIM; d++) n += out[b2 + d] ** 2; n = Math.sqrt(n); if (n) for (let d = 0; d < DIM; d++) out[b2 + d] /= n; }
  return out;
}
vectors = project(vectors); vectors = project(vectors);
const indexByLexiconId = new Int32Array(lex.idfById.length).fill(-1);
for (const [t, lid] of lex.ids) { const v = vocab.get(t); if (v !== undefined) indexByLexiconId[lid] = v; }
const space = { vocabulary: vocab, indexByLexiconId, vectors, dimensions: DIM, empty: false };
const docVecs = docs.map((w, i) => documentVector(terms[i], space, lex, avg));
removeCommonComponent(docVecs, DIM);
const vecById = new Map(docs.map((w, i) => [w.id, docVecs[i]]));

const index = buildRelevanceIndex(docs, { space: null });
for (const lambda of [0, 0.1, 0.2, 0.35, 0.5, 0.7]) {
  const per = queries.map((entry) => {
    const expanded = withFeedback(index, expandQuery(entry.query, index));
    const qv = documentVector(contentTerms(entry.query), space, lex, avg);
    const ranked = index.entries.map((e) => {
      const r = scoreEntry(e, expanded, index);
      const sem = semanticSimilarity(qv, vecById.get(e.work.id) || { norm: 0 });
      return { e, s: (1 - lambda) * (r.raw > 0 ? r.rank : -50) + lambda * 10 * sem };
    }).sort((a, b) => b.s - a.s).map((x) => x.e.work);
    return ndcgAt(ranked, (w) => entry.gains.get(w.id) || 0, 20);
  });
  console.log(`lambda ${lambda.toFixed(2)}   nDCG@20 ${mean(per).toFixed(4)}`);
}
