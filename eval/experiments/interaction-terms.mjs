// Uzzi et al. (2013) do not claim atypical combinations alone predict impact.
// They claim the combination does: work that is conventional at its median AND
// carries a tail of unusual pairings. That is an interaction, and a weighted sum
// of ranks cannot express one however the weights are set. Tested here, along
// with venue commonality, the last untried metadata field.
import { loadCorpus } from "../lib/adapters.mjs";
import { buildLabels, disruptionClasses } from "../lib/labels.mjs";
import { auc, spearman } from "../lib/metrics.mjs";
import { scoreBatch } from "../../src/shared/scoring.js";

const corpus = loadCorpus();
const labels = buildLabels(corpus.rawCandidates);
const scored = scoreBatch(corpus.candidates, corpus.peers, corpus.authors);
const byId = new Map(scored.map((w) => [w.id, w]));
const rawById = new Map(corpus.rawCandidates.map((w) => [w.id, w]));

// Venue commonality: how much of the corpus shares this paper's source.
const bySource = new Map();
for (const w of corpus.rawCandidates) if (w.sourceId) bySource.set(w.sourceId, (bySource.get(w.sourceId) || 0) + 1);

const ranksOf = (id) => new Map(byId.get(id).noveltyEvidence.signalContributions.map((c) => [c.signal, c.rank]));

const FEATURES = {
  // The Uzzi shape: conventional at the median, unusual in the tail.
  conventionalWithTail: (id) => {
    const r = ranksOf(id);
    const conventional = 1 - (r.get("pairSupport") ?? 0.5);   // un-invert: high support = conventional
    const tail = r.get("unseenPairFraction") ?? 0.5;
    return conventional * tail;
  },
  atypicalAndUnfamiliar: (id) => {
    const r = ranksOf(id);
    return (r.get("unseenPairFraction") ?? 0.5) * (r.get("unfamiliarReferenceFraction") ?? 0.5);
  },
  newTermsInNewCombination: (id) => {
    const r = ranksOf(id);
    return (r.get("emergentDensity") ?? 0.5) * (r.get("unseenPairFraction") ?? 0.5);
  },
  focusedButUnconventional: (id) => {
    const r = ranksOf(id);
    return (r.get("topicConcentration") ?? 0.5) * (r.get("canonShare") ?? 0.5);
  },
  venueCommonality: (id) => {
    const w = rawById.get(id);
    return w?.sourceId ? bySource.get(w.sourceId) || 0 : null;
  },
};

for (const name of ["tune", "test"]) {
  const set = corpus.rawCandidates.filter((w) => w.split === name && byId.has(w.id));
  const labelled = set.filter((w) => labels.get(w.id)?.noveltyLabel !== null);
  const classes = disruptionClasses(set, labels);
  const pos = classes.positives.filter((w) => w.workType !== "review");
  const neg = classes.negatives.filter((w) => w.workType !== "review");
  console.log(`\n${name}:`);
  for (const [label, fn] of Object.entries(FEATURES)) {
    const present = labelled.filter((w) => Number.isFinite(fn(w.id)));
    if (present.length < 60) { console.log(`  ${label.padEnd(28)} too few`); continue; }
    const rho = spearman(present.map((w) => fn(w.id)), present.map((w) => labels.get(w.id).noveltyLabel));
    const sep = auc(
      pos.filter((w) => Number.isFinite(fn(w.id))).map((w) => fn(w.id)),
      neg.filter((w) => Number.isFinite(fn(w.id))).map((w) => fn(w.id)),
    );
    console.log(`  ${label.padEnd(28)} spearman ${rho.toFixed(4).padStart(8)}   AUC ${sep.toFixed(4)}`);
  }
}
