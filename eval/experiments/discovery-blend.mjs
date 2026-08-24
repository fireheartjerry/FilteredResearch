// The "balanced" viewing order blends novelty and authorship. Swept on tune,
// reporting whether each blend beats BOTH of its parts -- the property C4 asks
// for -- so the choice is the most novelty-led blend that still qualifies.
import { loadCorpus } from "../lib/adapters.mjs";
import { scoreBatch } from "../../src/shared/scoring.js";
import { buildLabels, gainFromLabel } from "../lib/labels.mjs";
import { ndcgAt } from "../lib/metrics.mjs";

const corpus = loadCorpus();
const labels = buildLabels(corpus.rawCandidates);
const scored = scoreBatch(corpus.candidates, corpus.peers, corpus.authors);
const byId = new Map(scored.map((w) => [w.id, w]));

for (const split of ["tune", "test"]) {
  const set = corpus.rawCandidates.filter((w) => w.split === split && byId.has(w.id));
  const gain = (w) => gainFromLabel(labels.get(w.id).combinedLabel);
  const order = (fn) => [...set].sort((a, b) => fn(byId.get(b.id)) - fn(byId.get(a.id)));
  const nAlone = ndcgAt(order((w) => w.noveltyScore), gain, 50);
  const aAlone = ndcgAt(order((w) => w.researcherScore), gain, 50);
  console.log(`\n${split}: novelty alone ${nAlone.toFixed(4)}   authorship alone ${aAlone.toFixed(4)}`);
  console.log("  share  nDCG@50   beatsBoth");
  for (const share of [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.62]) {
    const v = ndcgAt(order((w) => share * w.noveltyScore + (1 - share) * w.researcherScore), gain, 50);
    console.log(`   ${share.toFixed(2)}  ${v.toFixed(4)}   ${v > nAlone && v > aAlone ? "yes" : "no"}`);
  }
}
