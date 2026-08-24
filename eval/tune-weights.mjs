// node eval/tune-weights.mjs
//
// Fits the novelty fusion weights and the discovery blend. Runs ONLY on the
// tuning split; the split assignment is seeded in the fixture builder and the
// reported numbers come from the disjoint test split, so nothing fitted here can
// leak into a reported result.
//
// It also prints each signal's correlation with the label on its own, which is
// the diagnostic that matters: a signal with no individual correlation and no
// complementarity is dead weight, and a signal with the wrong sign is worse.

import { writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { loadCorpus } from "./lib/adapters.mjs";
import { buildLabels, gainFromLabel, disruptionClasses } from "./lib/labels.mjs";
import { spearman, auc, ndcgAt, mean } from "./lib/metrics.mjs";
import { scoreBatch, NOVELTY_WEIGHTS } from "../src/shared/scoring.js";

const HERE = dirname(fileURLToPath(import.meta.url));
const SPLIT = "tune";

const SIGNALS = [
  ["crowding", true],
  ["centrality", true],
  ["emergentDensity", false],
  ["unseenPairFraction", false],
  ["pairSupport", true],
  ["canonShare", true],
  ["unfamiliarReferenceFraction", false],
  ["referenceCount", true],
  ["strongestCoupling", true],
  ["coupledPeerFraction", true],
  ["teamSize", true],
];

function round(value, places = 4) {
  return Number.isFinite(value) ? Number(value.toFixed(places)) : null;
}

function main() {
  const corpus = loadCorpus();
  const labels = buildLabels(corpus.rawCandidates);
  const scored = scoreBatch(corpus.candidates, corpus.peers, corpus.authors);
  const byId = new Map(scored.map((work) => [work.id, work]));
  const tune = corpus.rawCandidates.filter((work) => work.split === SPLIT && byId.has(work.id));

  const labelled = tune.filter((work) => labels.get(work.id)?.noveltyLabel !== null);
  const target = labelled.map((work) => labels.get(work.id).noveltyLabel);
  const classes = disruptionClasses(tune, labels);

  console.log(`Tuning split: ${tune.length} papers, ${labelled.length} with a disruption label`);
  console.log(`Separation task: ${classes.positives.length} disruptive vs ${classes.negatives.length} derivative/review\n`);
  console.log("Per-signal, measured alone (positive Spearman = the signal as oriented predicts novelty):\n");
  console.log("  signal                          n   spearman     AUC   review-gap");

  const diagnostics = [];
  for (const [key, invert] of SIGNALS) {
    const present = labelled.filter((work) => Number.isFinite(byId.get(work.id).noveltyEvidence[key]));
    if (present.length < 40) {
      console.log(`  ${key.padEnd(30)} ${String(present.length).padStart(4)}   (too few records)`);
      continue;
    }
    const sign = invert ? -1 : 1;
    const values = present.map((work) => sign * byId.get(work.id).noveltyEvidence[key]);
    const targets = present.map((work) => labels.get(work.id).noveltyLabel);
    const rho = spearman(values, targets);
    const separation = auc(
      classes.positives.filter((work) => Number.isFinite(byId.get(work.id).noveltyEvidence[key])).map((work) => sign * byId.get(work.id).noveltyEvidence[key]),
      classes.negatives.filter((work) => Number.isFinite(byId.get(work.id).noveltyEvidence[key])).map((work) => sign * byId.get(work.id).noveltyEvidence[key]),
    );
    const reviews = tune.filter((work) => work.workType === "review" && Number.isFinite(byId.get(work.id).noveltyEvidence[key]));
    const articles = tune.filter((work) => work.workType !== "review" && Number.isFinite(byId.get(work.id).noveltyEvidence[key]));
    const gap = mean(articles.map((work) => sign * byId.get(work.id).noveltyEvidence[key])) -
      mean(reviews.map((work) => sign * byId.get(work.id).noveltyEvidence[key]));
    diagnostics.push({ signal: key, invert, n: present.length, spearman: round(rho), auc: round(separation), reviewGap: round(gap, 5) });
    console.log(
      `  ${key.padEnd(30)} ${String(present.length).padStart(4)}   ${String(round(rho)).padStart(8)}  ${String(round(separation)).padStart(6)}   ${String(round(gap, 4)).padStart(10)}`,
    );
  }

  // ---- weight search --------------------------------------------------------
  // The per-signal ranks are independent of the weights, so one scoring pass
  // produces everything the search needs and the whole sweep costs milliseconds.
  // nDCG and AUC depend only on ordering, so searching the raw weighted sum is
  // exactly equivalent to searching the calibrated score.
  const ranksFor = (work) => {
    const out = new Map();
    for (const entry of byId.get(work.id).noveltyEvidence.signalContributions) out.set(entry.signal, entry.rank);
    return out;
  };
  const rankCache = new Map();
  for (const work of tune) rankCache.set(work.id, ranksFor(work));

  const keys = NOVELTY_WEIGHTS.map((entry) => entry.key);
  const noveltyGain = (work) => gainFromLabel(labels.get(work.id).noveltyLabel);
  function evaluateWeights(weights) {
    const score = (work) => {
      const ranks = rankCache.get(work.id);
      let total = 0;
      let available = 0;
      for (let index = 0; index < keys.length; index += 1) {
        const rank = ranks.get(keys[index]);
        if (rank === undefined) continue;
        total += weights[index] * rank;
        available += weights[index];
      }
      return available > 0 ? total / available : 0.5;
    };
    const ordered = [...labelled].sort((left, right) => score(right) - score(left));
    const ndcg = ndcgAt(ordered, noveltyGain, 50);
    const separation = auc(
      classes.positives.map(score),
      classes.negatives.map(score),
    );
    // Both halves of C1 matter, so the objective is their mean rather than
    // whichever one happens to be easier to move.
    return { ndcg, auc: separation, objective: (ndcg + separation) / 2 };
  }

  let weights = NOVELTY_WEIGHTS.map((entry) => entry.weight);
  let best = evaluateWeights(weights);
  console.log(`\nWeight search (coordinate ascent on the tuning split)`);
  console.log(`  start   objective ${round(best.objective)}  (nDCG@50 ${round(best.ndcg)}, AUC ${round(best.auc)})`);
  const grid = [0, 0.02, 0.05, 0.08, 0.12, 0.16, 0.2, 0.26, 0.32, 0.4];
  for (let pass = 0; pass < 6; pass += 1) {
    let improved = false;
    for (let index = 0; index < weights.length; index += 1) {
      const original = weights[index];
      let bestValue = original;
      for (const candidate of grid) {
        if (candidate === original) continue;
        weights[index] = candidate;
        const result = evaluateWeights(weights);
        if (result.objective > best.objective + 1e-6) {
          best = result;
          bestValue = candidate;
          improved = true;
        }
      }
      weights[index] = bestValue;
    }
    if (!improved) break;
  }
  const sum = weights.reduce((total, value) => total + value, 0) || 1;
  const normalised = weights.map((value) => Number((value / sum).toFixed(3)));
  console.log(`  tuned   objective ${round(best.objective)}  (nDCG@50 ${round(best.ndcg)}, AUC ${round(best.auc)})`);
  console.log(`\n  fitted weights:`);
  for (let index = 0; index < keys.length; index += 1) {
    if (normalised[index] > 0) console.log(`    ${keys[index].padEnd(30)} ${normalised[index]}`);
  }
  const dropped = keys.filter((_, index) => normalised[index] === 0);
  if (dropped.length) console.log(`    (dropped: ${dropped.join(", ")})`);

  // Discovery blend: how much novelty versus authorship best predicts which
  // papers were actually worth surfacing.
  console.log("\nDiscovery blend against the combined label (citation percentile):\n");
  const combinedGain = (work) => gainFromLabel(labels.get(work.id).combinedLabel);
  let bestBlend = null;
  for (let novelty = 0; novelty <= 1.001; novelty += 0.05) {
    const order = [...tune].sort((left, right) => {
      const score = (work) => novelty * byId.get(work.id).noveltyScore + (1 - novelty) * byId.get(work.id).researcherScore;
      return score(right) - score(left);
    });
    const value = ndcgAt(order, combinedGain, 50);
    if (!bestBlend || value > bestBlend.ndcg) bestBlend = { novelty: Number(novelty.toFixed(2)), ndcg: round(value) };
  }
  console.log(`  best novelty share ${bestBlend.novelty}  ->  nDCG@50 ${bestBlend.ndcg}`);

  const report = {
    generatedAt: new Date().toISOString(),
    split: SPLIT,
    papers: tune.length,
    labelled: labelled.length,
    currentWeights: NOVELTY_WEIGHTS,
    perSignal: diagnostics,
    bestDiscoveryBlend: bestBlend,
    fittedWeights: Object.fromEntries(keys.map((key, index) => [key, normalised[index]])),
    fittedObjective: { ndcg: round(best.ndcg), auc: round(best.auc), objective: round(best.objective) },
    note: "Fitted on the tuning split only. Reported metrics always come from the disjoint test split.",
  };
  writeFileSync(join(HERE, "tuning.json"), `${JSON.stringify(report, null, 2)}\n`);
  console.log(`\n  wrote ${join(HERE, "tuning.json")}\n`);
}

main();
