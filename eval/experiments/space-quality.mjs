import { loadCorpus, loadFixture } from "../lib/adapters.mjs";
import { contentTerms, textOf, buildLexicon, bm25Weights, cosineOfWeighted, tokenize } from "../../src/shared/text.js";
import { auc } from "../lib/metrics.mjs";

const c = loadCorpus();
const works = [...c.peers.slice(0, 1500), ...c.candidates.slice(0, 800)];
const docs = works.map((w) => contentTerms(textOf(w)));
const lex = buildLexicon(docs);
const avg = docs.reduce((s, d) => s + d.length, 0) / docs.length;
const raw = loadFixture("candidates");
const topicOf = new Map(raw.map((w) => [w.id, w.topicId]));
const topicFor = (w) => topicOf.get(w.id) || w.subfieldId;

function evaluate(name, simFn) {
  const pos = [], neg = [];
  for (let i = 0; i < works.length; i++) {
    const ti = topicFor(works[i]);
    for (let k = 0; k < 6; k++) {
      const j = (i * 7919 + k * 104729) % works.length;
      if (i === j) continue;
      const tj = topicFor(works[j]);
      const s = simFn(i, j);
      if (ti && tj && ti === tj) pos.push(s); else neg.push(s);
    }
  }
  const m = (a) => a.reduce((x, y) => x + y, 0) / Math.max(1, a.length);
  console.log(`${name.padEnd(34)} AUC ${auc(pos, neg).toFixed(4)}  same ${m(pos).toFixed(3)}  diff ${m(neg).toFixed(3)}`);
}

const bm = docs.map((d) => bm25Weights(d, lex, avg));
evaluate("BM25 lexical cosine", (i, j) => cosineOfWeighted(bm[i], bm[j]));

// --- random indexing variants ---
const DIM = 128, NZ = 8, WIN = 4;
function hashTerm(t, salt) { let h = 2166136261 ^ salt; for (let i = 0; i < t.length; i++) { h ^= t.charCodeAt(i); h = Math.imul(h, 16777619) >>> 0; } return h >>> 0; }
function sig(t) { const p = new Int16Array(NZ), s = new Int8Array(NZ); for (let i = 0; i < NZ; i++) { const h = hashTerm(t, i * 2654435761); p[i] = h % DIM; s[i] = (h >>> 16) & 1 ? 1 : -1; } return { p, s }; }

function buildSpace({ idfWeightContext }) {
  const vocab = new Map();
  for (const t of lex.idf.keys()) if (!t.includes(" ")) vocab.set(t, vocab.size);
  const V = vocab.size, vec = new Float32Array(V * DIM);
  const sigs = []; for (const [t, i] of vocab) sigs[i] = sig(t);
  const toks = works.map((w) => tokenize(textOf(w), 400).filter((t) => !t.includes("-")));
  for (const tk of toks) {
    const m = tk.map((t) => { const i = vocab.get(t); return i === undefined ? -1 : i; });
    for (let p = 0; p < m.length; p++) {
      const tgt = m[p]; if (tgt < 0) continue;
      const base = tgt * DIM;
      for (let q = Math.max(0, p - WIN); q <= Math.min(m.length - 1, p + WIN); q++) {
        if (q === p) continue;
        const ctx = m[q]; if (ctx < 0) continue;
        let w = 1 / Math.abs(q - p);
        if (idfWeightContext) w *= Math.min(3, lex.idf.get(tk[q]) || 1);
        const { p: pos, s: sgn } = sigs[ctx];
        for (let z = 0; z < NZ; z++) vec[base + pos[z]] += sgn[z] * w;
      }
    }
  }
  for (let i = 0; i < V; i++) { const b = i * DIM; let n = 0; for (let z = 0; z < DIM; z++) n += vec[b + z] ** 2; n = Math.sqrt(n); if (n) for (let z = 0; z < DIM; z++) vec[b + z] /= n; }
  return { vocab, vec };
}

function docVectors(space, { center }) {
  const out = works.map((_, i) => {
    const v = new Float32Array(DIM);
    for (const [t, w] of bm[i].weights) { const idx = space.vocab.get(t); if (idx === undefined) continue; const b = idx * DIM; for (let z = 0; z < DIM; z++) v[z] += w * space.vec[b + z]; }
    let n = 0; for (let z = 0; z < DIM; z++) n += v[z] ** 2; n = Math.sqrt(n); if (n) for (let z = 0; z < DIM; z++) v[z] /= n;
    return { v, n };
  });
  if (center) {
    const mu = new Float32Array(DIM);
    for (const d of out) if (d.n) for (let z = 0; z < DIM; z++) mu[z] += d.v[z];
    let mn = 0; for (let z = 0; z < DIM; z++) mn += mu[z] ** 2; mn = Math.sqrt(mn) || 1;
    for (let z = 0; z < DIM; z++) mu[z] /= mn;
    for (const d of out) {
      if (!d.n) continue;
      let dot = 0; for (let z = 0; z < DIM; z++) dot += d.v[z] * mu[z];
      for (let z = 0; z < DIM; z++) d.v[z] -= dot * mu[z];
      let n = 0; for (let z = 0; z < DIM; z++) n += d.v[z] ** 2; n = Math.sqrt(n); d.n = n; if (n) for (let z = 0; z < DIM; z++) d.v[z] /= n;
    }
  }
  return out;
}
const cos = (a, b) => { if (!a.n || !b.n) return 0; let d = 0; for (let z = 0; z < DIM; z++) d += a.v[z] * b.v[z]; return d; };

for (const idfw of [false, true]) for (const center of [false, true]) {
  const sp = buildSpace({ idfWeightContext: idfw });
  const dv = docVectors(sp, { center });
  evaluate(`RI idf=${idfw?1:0} center=${center?1:0}`, (i, j) => cos(dv[i], dv[j]));
}
