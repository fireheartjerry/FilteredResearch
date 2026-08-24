// How well can "this is consolidation work" be detected from structure alone --
// no keyword list, no document-type field? Fitted on the tuning split only.
import { loadCorpus } from "../lib/adapters.mjs";
import { scoreBatch } from "../../src/shared/scoring.js";
import { auc } from "../lib/metrics.mjs";

const corpus = loadCorpus();
const scored = scoreBatch(corpus.candidates, corpus.peers, corpus.authors);
const byId = new Map(scored.map((w) => [w.id, w]));
const tune = corpus.rawCandidates.filter((w) => w.split === "tune" && byId.has(w.id));
const test = corpus.rawCandidates.filter((w) => w.split === "test" && byId.has(w.id));

const FEATURES = ["referenceCount", "emergentDensity", "centrality", "teamSize", "topicConcentration",
  "canonShare", "unfamiliarReferenceFraction", "crowding", "coupledPeerFraction", "genericness"];

function featuresOf(w) {
  const e = byId.get(w.id).noveltyEvidence;
  return FEATURES.map((f) => {
    const v = e[f];
    return Number.isFinite(v) ? v : 0;
  });
}
// rank-normalise each feature over the pooled corpus so scales are comparable
const all = [...tune, ...test];
const raw = all.map(featuresOf);
const sortedCols = FEATURES.map((_, i) => raw.map((r) => r[i]).sort((a, b) => a - b));
function rank(v, col) { let lo = 0, hi = col.length; while (lo < hi) { const m = (lo + hi) >> 1; if (col[m] < v) lo = m + 1; else hi = m; } return lo / Math.max(1, col.length); }
const X = new Map(all.map((w, i) => [w.id, raw[i].map((v, c) => rank(v, sortedCols[c]))]));
const y = new Map(all.map((w) => [w.id, w.workType === "review" ? 1 : 0]));

// logistic regression, plain gradient descent
let beta = new Array(FEATURES.length + 1).fill(0);
const trainSet = tune;
for (let epoch = 0; epoch < 4000; epoch++) {
  const grad = new Array(beta.length).fill(0);
  for (const w of trainSet) {
    const x = [1, ...X.get(w.id)];
    let z = 0; for (let i = 0; i < beta.length; i++) z += beta[i] * x[i];
    const p = 1 / (1 + Math.exp(-z));
    const err = p - y.get(w.id);
    for (let i = 0; i < beta.length; i++) grad[i] += err * x[i];
  }
  for (let i = 0; i < beta.length; i++) beta[i] -= (0.35 / trainSet.length) * grad[i] + 0.0002 * beta[i];
}
const predict = (w) => { const x = [1, ...X.get(w.id)]; let z = 0; for (let i = 0; i < beta.length; i++) z += beta[i] * x[i]; return 1 / (1 + Math.exp(-z)); };

for (const [name, set] of [["tune", tune], ["test", test]]) {
  const r = set.filter((w) => w.workType === "review").map(predict);
  const a = set.filter((w) => w.workType !== "review").map(predict);
  console.log(`${name}: AUC(review > article) = ${auc(r, a).toFixed(4)}   reviews ${r.length} articles ${a.length}`);
}
console.log("\ncoefficients (rank-space):");
console.log(`  intercept ${beta[0].toFixed(3)}`);
FEATURES.forEach((f, i) => console.log(`  ${f.padEnd(30)} ${beta[i + 1].toFixed(3)}`));
