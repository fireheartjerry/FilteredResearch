// The consolidation penalty is driven by where a paper's shape probability sits
// among the batch's. That makes the penalty depend on which OTHER papers are in
// the index, which is the route by which a score drifts as a corpus grows.
// A fixed reference distribution, measured once on the tuning split and shipped
// as constants, removes the dependence entirely.
import { loadCorpus } from "../lib/adapters.mjs";
import { scoreBatch } from "../../src/shared/scoring.js";
import { quantile } from "../lib/metrics.mjs";

const corpus = loadCorpus();
const rawById = new Map(corpus.rawCandidates.map((w) => [w.id, w]));
const tune = corpus.candidates.filter((w) => rawById.get(w.id)?.split === "tune");
const scored = scoreBatch(tune, corpus.peers, corpus.authors);
const shapes = scored
  .map((w) => w.noveltyEvidence.consolidationShape)
  .filter((v) => Number.isFinite(v))
  .sort((a, b) => a - b);

console.log(`${shapes.length} shape values on the tuning split`);
const points = [];
for (let i = 0; i <= 20; i += 1) points.push(Number(quantile(shapes, i / 20).toFixed(5)));
console.log("SHAPE_REFERENCE (21 quantiles, 0 to 1 in 0.05 steps):");
console.log(JSON.stringify(points));
