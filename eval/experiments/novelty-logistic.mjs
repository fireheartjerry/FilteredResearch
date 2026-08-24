// The fusion is a weighted sum of per-signal ranks, fitted by coordinate ascent
// on a blended objective. A logistic fit targets the separation criterion
// directly instead. Fitted on TUNE, reported on TEST, so overfitting shows.
import { loadCorpus } from "../lib/adapters.mjs";
import { buildLabels, disruptionClasses, gainFromLabel } from "../lib/labels.mjs";
import { auc, ndcgAt } from "../lib/metrics.mjs";
import { scoreBatch, NOVELTY_WEIGHTS } from "../../src/shared/scoring.js";

const corpus = loadCorpus();
const labels = buildLabels(corpus.rawCandidates);
const scored = scoreBatch(corpus.candidates, corpus.peers, corpus.authors);
const byId = new Map(scored.map((w) => [w.id, w]));
const keys = NOVELTY_WEIGHTS.map((e) => e.key);

const ranksOf = (id) => {
  const map = new Map(byId.get(id).noveltyEvidence.signalContributions.map((c) => [c.signal, c.rank]));
  // Missing signals sit at the middle, which is what the renormalised sum does.
  return keys.map((k) => (map.has(k) ? map.get(k) : 0.5));
};

function split(name) {
  const set = corpus.rawCandidates.filter((w) => w.split === name && byId.has(w.id));
  const classes = disruptionClasses(set, labels);
  return {
    set,
    labelled: set.filter((w) => labels.get(w.id)?.noveltyLabel !== null),
    pos: classes.positives.filter((w) => w.workType !== "review"),
    neg: classes.negatives.filter((w) => w.workType !== "review"),
  };
}
const tune = split("tune");
const test = split("test");
console.log(`tune ${tune.pos.length}+/${tune.neg.length}-   test ${test.pos.length}+/${test.neg.length}-`);

// Logistic regression with L2, targeting the separation the criterion reads.
let beta = new Array(keys.length + 1).fill(0);
const rows = [...tune.pos.map((w) => [ranksOf(w.id), 1]), ...tune.neg.map((w) => [ranksOf(w.id), 0])];
for (let epoch = 0; epoch < 6000; epoch += 1) {
  const grad = new Array(beta.length).fill(0);
  for (const [features, y] of rows) {
    const x = [1, ...features];
    let z = 0;
    for (let i = 0; i < beta.length; i += 1) z += beta[i] * x[i];
    const p = 1 / (1 + Math.exp(-z));
    for (let i = 0; i < beta.length; i += 1) grad[i] += (p - y) * x[i];
  }
  for (let i = 0; i < beta.length; i += 1) beta[i] -= (0.4 / rows.length) * grad[i] + 0.001 * beta[i];
}
const predict = (id) => { const x = [1, ...ranksOf(id)]; let z = 0; for (let i = 0; i < beta.length; i += 1) z += beta[i] * x[i]; return z; };

for (const [name, s] of [["tune", tune], ["test", test]]) {
  const sep = auc(s.pos.map((w) => predict(w.id)), s.neg.map((w) => predict(w.id)));
  const ranked = [...s.labelled].sort((a, b) => predict(b.id) - predict(a.id));
  const nd = ndcgAt(ranked, (w) => gainFromLabel(labels.get(w.id).noveltyLabel), 50);
  const shippedSep = auc(s.pos.map((w) => byId.get(w.id).noveltyScore), s.neg.map((w) => byId.get(w.id).noveltyScore));
  const shippedNd = ndcgAt([...s.labelled].sort((a, b) => byId.get(b.id).noveltyScore - byId.get(a.id).noveltyScore), (w) => gainFromLabel(labels.get(w.id).noveltyLabel), 50);
  console.log(`${name}:  logistic AUC ${sep.toFixed(4)} nDCG ${nd.toFixed(4)}   |   shipped AUC ${shippedSep.toFixed(4)} nDCG ${shippedNd.toFixed(4)}`);
}
console.log("\ncoefficients:");
keys.forEach((k, i) => console.log(`  ${k.padEnd(30)} ${beta[i + 1].toFixed(3)}`));
