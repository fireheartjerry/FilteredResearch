// Can a better local space close the semantic gap? Random indexing gets AUC 0.63
// on same-topic discrimination. PPMI + a power-iteration subspace refinement is
// the next rung that still fits in an extension.
import { loadCorpus, loadFixture } from "../lib/adapters.mjs";
import { contentTerms, textOf, buildLexicon, bm25Weights, tokenize } from "../../src/shared/text.js";
import { buildSemanticSpace, documentVector, removeCommonComponent, tokensForSpace, semanticSimilarity } from "../../src/shared/semantic.js";
import { auc } from "../lib/metrics.mjs";

const c = loadCorpus();
const works = [...c.peers.slice(0, 2000), ...c.candidates.slice(0, 1200)];
const docs = works.map((w) => contentTerms(textOf(w)));
const lex = buildLexicon(docs);
const avg = docs.reduce((s, d) => s + d.length, 0) / docs.length;
const raw = loadFixture("candidates");
const topicOf = new Map(raw.map((w) => [w.id, w.topicId]));
const topicFor = (w) => topicOf.get(w.id) || w.subfieldId;

function evaluate(name, vecs) {
  const pos = [], neg = [];
  for (let i = 0; i < works.length; i++) {
    const ti = topicFor(works[i]);
    for (let k = 0; k < 6; k++) {
      const j = (i * 7919 + k * 104729) % works.length;
      if (i === j) continue;
      const tj = topicFor(works[j]);
      const s = semanticSimilarity(vecs[i], vecs[j]);
      if (ti && tj && ti === tj) pos.push(s); else neg.push(s);
    }
  }
  console.log(`${name.padEnd(34)} AUC ${auc(pos, neg).toFixed(4)}`);
  return auc(pos, neg);
}

// current
const spaceA = buildSemanticSpace((function* () { for (const w of works) yield tokensForSpace(textOf(w)); })(), lex, { documentCount: works.length });
const vecsA = works.map((w, i) => documentVector(docs[i], spaceA, lex, avg));
removeCommonComponent(vecsA, spaceA.dimensions);
evaluate("random indexing (shipped)", vecsA);

// --- PPMI over an explicit co-occurrence table, projected down -------------
const DIM = 128, WIN = Number(process.env.WIN || 4);
const VOCAB_CAP = Number(process.env.VOCAB_CAP || 100000);
const MIN_PAIR = Number(process.env.MIN_PAIR || 1);
const vocab = new Map();
{
  const unigrams = [...lex.ids.keys()].filter((t) => !t.includes(" "));
  unigrams.sort((a, b) => (lex.documentFrequency.get(b) || 0) - (lex.documentFrequency.get(a) || 0) || a.localeCompare(b));
  for (const t of unigrams.slice(0, VOCAB_CAP)) vocab.set(t, vocab.size);
}
const V = vocab.size;
const single = new Float64Array(V);
const pairs = new Map();
let totalPairs = 0;
for (const w of works) {
  const toks = tokenize(textOf(w), 400).filter((t) => !t.includes("-")).map((t) => vocab.get(t) ?? -1);
  for (let p = 0; p < toks.length; p++) {
    if (toks[p] < 0) continue;
    single[toks[p]] += 1;
    for (let q = Math.max(0, p - WIN); q <= Math.min(toks.length - 1, p + WIN); q++) {
      if (q === p || toks[q] < 0) continue;
      const key = toks[p] * V + toks[q];
      pairs.set(key, (pairs.get(key) || 0) + 1);
      totalPairs += 1;
    }
  }
}
if (MIN_PAIR > 1) { for (const [k, v] of pairs) if (v < MIN_PAIR) pairs.delete(k); }
console.log(`  vocab ${V}, distinct pairs ${pairs.size}, window ${WIN}, minPair ${MIN_PAIR}`);

// Random projection of the PPMI rows: Y = PPMI x Omega, then one power iteration
// through the same matrix to sharpen the subspace.
function hash(n, salt) { let h = (n ^ salt) >>> 0; h = Math.imul(h, 2246822519) >>> 0; h ^= h >>> 13; h = Math.imul(h, 3266489917) >>> 0; return h >>> 0; }
const omega = new Float32Array(V * DIM);
for (let i = 0; i < V * DIM; i++) omega[i] = ((hash(i, 0x9e3779b9) & 1023) / 511.5) - 1;

function project(input) {
  const out = new Float32Array(V * DIM);
  for (const [key, count] of pairs) {
    const a = Math.floor(key / V), b = key % V;
    const pmi = Math.log((count * totalPairs) / Math.max(1, single[a] * single[b]));
    if (!(pmi > 0)) continue;
    const wgt = pmi;
    const ab = a * DIM, bb = b * DIM;
    for (let d = 0; d < DIM; d++) out[ab + d] += wgt * input[bb + d];
  }
  for (let i = 0; i < V; i++) {
    const base = i * DIM;
    let n = 0; for (let d = 0; d < DIM; d++) n += out[base + d] ** 2;
    n = Math.sqrt(n); if (n) for (let d = 0; d < DIM; d++) out[base + d] /= n;
  }
  return out;
}

let vectors = project(omega);
for (const iterations of [1, 2, 3]) {
  const spaceB = { vocabulary: vocab, indexByLexiconId: spaceA.indexByLexiconId, vectors, dimensions: DIM, empty: false };
  const vecsB = works.map((w, i) => documentVector(docs[i], spaceB, lex, avg));
  removeCommonComponent(vecsB, DIM);
  evaluate(`ppmi projection, ${iterations} pass`, vecsB);
  vectors = project(vectors);
}
