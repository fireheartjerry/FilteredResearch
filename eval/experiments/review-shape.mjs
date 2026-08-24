import { loadCorpus } from "../lib/adapters.mjs";
import { scoreBatch } from "../../src/shared/scoring.js";
import { auc, mean, quantile } from "../lib/metrics.mjs";

const corpus = loadCorpus();
const scored = scoreBatch(corpus.candidates, corpus.peers, corpus.authors);
const byId = new Map(scored.map((w) => [w.id, w]));
const reviews = corpus.rawCandidates.filter((w) => w.workType === "review" && byId.has(w.id));
const articles = corpus.rawCandidates.filter((w) => w.workType !== "review" && byId.has(w.id));
console.log(`reviews ${reviews.length}  articles ${articles.length}\n`);
const fields = ["referenceCount", "canonShare", "centrality", "emergentDensity", "crowding",
  "unseenPairFraction", "pairSupport", "unfamiliarReferenceFraction", "strongestCoupling",
  "coupledPeerFraction", "teamSize", "fusedValue", "consolidationPrior"];
console.log("field                          reviewMean  articleMean   AUC(article>review)");
for (const f of fields) {
  const r = reviews.map((w) => byId.get(w.id).noveltyEvidence[f]).filter(Number.isFinite);
  const a = articles.map((w) => byId.get(w.id).noveltyEvidence[f]).filter(Number.isFinite);
  if (r.length < 20 || a.length < 20) { console.log(`${f.padEnd(30)} (sparse ${r.length}/${a.length})`); continue; }
  console.log(`${f.padEnd(30)} ${mean(r).toFixed(4).padStart(10)} ${mean(a).toFixed(4).padStart(12)} ${auc(a, r).toFixed(4).padStart(10)}`);
}
const rs = reviews.map((w) => byId.get(w.id).noveltyScore).sort((x, y) => x - y);
const as = articles.map((w) => byId.get(w.id).noveltyScore).sort((x, y) => x - y);
console.log(`\nnoveltyScore  review median ${quantile(rs, 0.5).toFixed(2)}  article median ${quantile(as, 0.5).toFixed(2)}  gap ${(quantile(as, 0.5) - quantile(rs, 0.5)).toFixed(2)}`);
console.log(`AUC(article > review) on final score: ${auc(articles.map((w) => byId.get(w.id).noveltyScore), reviews.map((w) => byId.get(w.id).noveltyScore)).toFixed(4)}`);
