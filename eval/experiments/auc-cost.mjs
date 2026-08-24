// The fused signal separates better than the finished score does. Something
// between them costs AUC. This measures each stage on the TEST split so the
// culprit is identified rather than guessed at.
import { loadCorpus } from "../lib/adapters.mjs";
import { buildLabels, disruptionClasses } from "../lib/labels.mjs";
import { auc, quantile } from "../lib/metrics.mjs";
import { scoreBatch } from "../../src/shared/scoring.js";

const corpus = loadCorpus();
const labels = buildLabels(corpus.rawCandidates);

function measure(label, options) {
  const scored = scoreBatch(corpus.candidates, corpus.peers, corpus.authors, options);
  const byId = new Map(scored.map((w) => [w.id, w]));
  const set = corpus.rawCandidates.filter((w) => w.split === "test" && byId.has(w.id));
  const classes = disruptionClasses(set, labels);
  const pos = classes.positives.filter((w) => w.workType !== "review");
  const neg = classes.negatives.filter((w) => w.workType !== "review");
  const of = (key) => (w) => byId.get(w.id).noveltyEvidence[key];

  const stages = {
    fused: auc(pos.map(of("fusedValue")), neg.map(of("fusedValue"))),
    afterCohortRank: auc(pos.map(of("calibratedValue")), neg.map(of("calibratedValue"))),
    rawAfterPenalty: auc(pos.map(of("rawScore")), neg.map(of("rawScore"))),
    finalScore: auc(pos.map((w) => byId.get(w.id).noveltyScore), neg.map((w) => byId.get(w.id).noveltyScore)),
  };
  const reviews = set.filter((w) => w.workType === "review").map((w) => byId.get(w.id).noveltyScore).sort((a, b) => a - b);
  const articles = set.filter((w) => w.workType !== "review").map((w) => byId.get(w.id).noveltyScore).sort((a, b) => a - b);
  const gap = quantile(articles, 0.5) - quantile(reviews, 0.5);
  console.log(`${label.padEnd(34)} fused ${stages.fused.toFixed(4)}  cohortRank ${stages.afterCohortRank.toFixed(4)}  afterPenalty ${stages.rawAfterPenalty.toFixed(4)}  final ${stages.finalScore.toFixed(4)}  reviewGap ${gap.toFixed(1)}`);
}

measure("shipped", {});
for (const w of [0, 10, 20, 28, 34]) measure(`structural consolidation = ${w}`, { structuralConsolidationWeight: w });
