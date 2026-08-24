// How hard should the consolidation penalty press? Swept on the tuning split.
import { loadCorpus } from "../lib/adapters.mjs";
import { scoreBatch } from "../../src/shared/scoring.js";
import { buildLabels, gainFromLabel, disruptionClasses } from "../lib/labels.mjs";
import { auc, ndcgAt, quantile } from "../lib/metrics.mjs";

const corpus = loadCorpus();
const labels = buildLabels(corpus.rawCandidates);
console.log("weight   nDCG@50    AUC   objective   reviewGap");
for (const weight of [0, 8, 12, 16, 20, 25, 30, 36]) {
  const scored = scoreBatch(corpus.candidates, corpus.peers, corpus.authors, { structuralConsolidationWeight: weight });
  const byId = new Map(scored.map((w) => [w.id, w]));
  const tune = corpus.rawCandidates.filter((w) => w.split === "tune" && byId.has(w.id));
  const labelled = tune.filter((w) => labels.get(w.id)?.noveltyLabel !== null);
  const gain = (w) => gainFromLabel(labels.get(w.id).noveltyLabel);
  const order = [...labelled].sort((a, b) => byId.get(b.id).noveltyScore - byId.get(a.id).noveltyScore);
  const nd = ndcgAt(order, gain, 50);
  const cl = disruptionClasses(tune, labels);
  const ar = auc(cl.positives.map((w) => byId.get(w.id).noveltyScore), cl.negatives.map((w) => byId.get(w.id).noveltyScore));
  const rs = tune.filter((w) => w.workType === "review").map((w) => byId.get(w.id).noveltyScore).sort((a, b) => a - b);
  const as = tune.filter((w) => w.workType !== "review").map((w) => byId.get(w.id).noveltyScore).sort((a, b) => a - b);
  const gap = quantile(as, 0.5) - quantile(rs, 0.5);
  console.log(`${String(weight).padStart(5)}   ${nd.toFixed(4)}  ${ar.toFixed(4)}    ${((nd + ar) / 2).toFixed(4)}   ${gap.toFixed(2)}`);
}
