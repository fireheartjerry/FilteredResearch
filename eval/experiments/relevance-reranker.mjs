// Can a learned reranker beat hand-weighted BM25 on this task?
//
// Four hand-built method families have been tried and capped around nDCG@20 0.42.
// This fits a small linear model over retrieval features instead -- the one thing
// not yet attempted. Fitted on the TUNING split, evaluated on tune and test
// separately so overfitting is visible rather than assumed away.

import { loadCorpus } from "../lib/adapters.mjs";
import { buildRelevanceQueries } from "../lib/labels.mjs";
import { ndcgAt, mean } from "../lib/metrics.mjs";
import { buildRelevanceIndex, expandQuery, withFeedback, scoreEntry } from "../../src/shared/relevance.js";

const corpus = loadCorpus();
const relevanceCorpus = loadCorpus({ forRelevance: true });
const splitOf = new Map(corpus.rawCandidates.map((w) => [w.id, w.split]));

function queriesFor(split) {
  const ids = new Set(corpus.rawCandidates.filter((w) => w.split === split).map((w) => w.id));
  return buildRelevanceQueries(corpus.rawCandidates, { minimumRelevant: 10, maximumQueries: 60 })
    .map((entry) => {
      const gains = new Map([...entry.gains].filter(([id]) => ids.has(id)));
      return { ...entry, gains, relevantCount: [...gains.values()].filter((g) => g >= 2).length };
    })
    .filter((entry) => entry.relevantCount >= 6);
}

const docsFor = (split) => relevanceCorpus.candidates.filter((w) => splitOf.get(w.id) === split);

// Features that a linear model can combine but a fixed formula cannot weigh
// against each other without being told how.
function featuresFor(entry, expanded, index) {
  const result = scoreEntry(entry, expanded, index);
  const queryTerms = [...expanded.terms.keys()];
  let titleHits = 0;
  let maxIdf = 0;
  let matched = 0;
  for (const term of queryTerms) {
    if ((entry.counts.get(term) || 0) > 0) {
      matched += 1;
      maxIdf = Math.max(maxIdf, index.lexicon.idf.get(term) || 0);
    }
    if ((entry.titleCounts.get(term) || 0) > 0) titleHits += 1;
  }
  return [
    1,
    Math.log1p(result.raw),
    result.coverage || 0,
    result.proximity || 0,
    titleHits / Math.max(1, queryTerms.length),
    matched / Math.max(1, queryTerms.length),
    Math.log1p(entry.body.length) / 10,
    maxIdf / 10,
  ];
}

function buildRows(split) {
  const docs = docsFor(split);
  const index = buildRelevanceIndex(docs, { space: null });
  const rows = [];
  for (const query of queriesFor(split)) {
    const expanded = withFeedback(index, expandQuery(query.query, index));
    const scored = index.entries.map((entry) => ({
      entry,
      features: featuresFor(entry, expanded, index),
      gain: query.gains.get(entry.work.id) || 0,
    }));
    rows.push({ query, scored });
  }
  return { index, rows };
}

console.log("building features...");
const tune = buildRows("tune");
const test = buildRows("test");

const WIDTH = tune.rows[0].scored[0].features.length;
let beta = new Array(WIDTH).fill(0);
beta[1] = 1;

const predict = (features) => {
  let total = 0;
  for (let i = 0; i < WIDTH; i += 1) total += beta[i] * features[i];
  return total;
};

function evaluate({ rows }, k = 20) {
  return mean(rows.map(({ scored }) => {
    const ranked = [...scored].sort((a, b) => predict(b.features) - predict(a.features));
    return ndcgAt(ranked, (item) => item.gain, k);
  }));
}

console.log(`start   tune ${evaluate(tune).toFixed(4)}   test ${evaluate(test).toFixed(4)}`);

// Coordinate ascent directly on nDCG@20 over the tuning split.
const GRID = [-2, -1, -0.5, -0.2, 0, 0.2, 0.5, 1, 2, 4];
let best = evaluate(tune);
for (let pass = 0; pass < 5; pass += 1) {
  let improved = false;
  for (let index = 0; index < WIDTH; index += 1) {
    const original = beta[index];
    let bestValue = original;
    for (const candidate of GRID) {
      beta[index] = candidate;
      const value = evaluate(tune);
      if (value > best + 1e-6) { best = value; bestValue = candidate; improved = true; }
    }
    beta[index] = bestValue;
  }
  if (!improved) break;
}

console.log(`tuned   tune ${evaluate(tune).toFixed(4)}   test ${evaluate(test).toFixed(4)}`);
console.log(`\nweights: ${beta.map((b) => b.toFixed(2)).join(", ")}`);
console.log("(bias, log bm25, coverage, proximity, titleHitShare, matchShare, logLen, maxIdf)");
