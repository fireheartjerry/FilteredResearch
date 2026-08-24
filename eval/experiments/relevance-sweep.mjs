// Relevance parameters, swept on the TUNING split only.
//
// The reported relevance numbers come from the test split, so nothing chosen
// here can flatter them.

import { loadCorpus } from "../lib/adapters.mjs";
import { buildRelevanceQueries } from "../lib/labels.mjs";
import { ndcgAt, mean } from "../lib/metrics.mjs";
import { buildRelevanceIndex, expandQuery, scoreEntry } from "../../src/shared/relevance.js";
import { contentTerms, textOf, buildLexicon } from "../../src/shared/text.js";
import { buildSemanticSpace, tokensForSpace } from "../../src/shared/semantic.js";

const corpus = loadCorpus();
const relevanceCorpus = loadCorpus({ forRelevance: true });
const rawById = new Map(corpus.rawCandidates.map((w) => [w.id, w]));
const tuneIds = new Set(corpus.rawCandidates.filter((w) => w.split === "tune").map((w) => w.id));
const docs = relevanceCorpus.candidates.filter((w) => tuneIds.has(w.id));

const queries = buildRelevanceQueries(corpus.rawCandidates, { minimumRelevant: 10, maximumQueries: 60 })
  .map((entry) => {
    const gains = new Map([...entry.gains].filter(([id]) => tuneIds.has(id)));
    return { ...entry, gains, relevantCount: [...gains.values()].filter((g) => g >= 2).length };
  })
  .filter((entry) => entry.relevantCount >= 6);

console.log(`${docs.length} documents, ${queries.length} tuning queries\n`);

// A semantic space over the same documents, so expansion can be switched on.
const terms = docs.map((w) => contentTerms(textOf(w)));
const lexicon = buildLexicon(terms);
const space = buildSemanticSpace(
  (function* () { for (const w of docs) yield tokensForSpace(textOf(w)); })(),
  lexicon,
  { documentCount: docs.length },
);

function evaluate(label, { useSpace }) {
  const index = buildRelevanceIndex(docs, { space: useSpace ? space : null });
  const perQuery = queries.map((entry) => {
    const expanded = expandQuery(entry.query, index);
    const ranked = index.entries
      .map((e) => ({ e, s: scoreEntry(e, expanded, index).raw }))
      .sort((a, b) => b.s - a.s || String(a.e.work.id).localeCompare(String(b.e.work.id)))
      .map((x) => x.e.work);
    const gainOf = (w) => entry.gains.get(w.id) || 0;
    return { ndcg20: ndcgAt(ranked, gainOf, 20), ndcg10: ndcgAt(ranked, gainOf, 10) };
  });
  console.log(`${label.padEnd(30)} nDCG@20 ${mean(perQuery.map((q) => q.ndcg20)).toFixed(4)}   nDCG@10 ${mean(perQuery.map((q) => q.ndcg10)).toFixed(4)}`);
}

evaluate("current (no space)", { useSpace: false });
evaluate("current + semantic expansion", { useSpace: true });

// Pseudo-relevance feedback: rank once, assume the top few documents are about
// the right thing, harvest their strongest terms, and ask again. This is the
// standard answer to a query whose words simply do not appear in the documents
// that answer it -- which is exactly a topic name against papers that use the
// field's own jargon instead.
function withFeedback(index, entry, { depth, terms: termCount, weight }) {
  const first = expandQuery(entry.query, index);
  const initial = index.entries
    .map((e) => ({ e, s: scoreEntry(e, first, index).raw }))
    .sort((a, b) => b.s - a.s)
    .slice(0, depth)
    .filter((x) => x.s > 0);
  if (!initial.length) return first;

  const weightByTerm = new Map();
  for (const { e } of initial) {
    for (const [term, count] of e.counts) {
      const idf = index.lexicon.idf.get(term);
      if (!idf) continue;
      weightByTerm.set(term, (weightByTerm.get(term) || 0) + idf * Math.min(3, count));
    }
  }
  const harvested = [...weightByTerm.entries()]
    .filter(([term]) => !first.terms.has(term))
    .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
    .slice(0, termCount);
  const top = harvested[0]?.[1] || 1;
  const expanded = new Map(first.terms);
  for (const [term, value] of harvested) expanded.set(term, weight * (value / top));
  return { ...first, terms: expanded };
}

function evaluateFeedback(label, options) {
  const index = buildRelevanceIndex(docs, { space: null });
  const perQuery = queries.map((entry) => {
    const expanded = withFeedback(index, entry, options);
    const ranked = index.entries
      .map((e) => ({ e, s: scoreEntry(e, expanded, index).raw }))
      .sort((a, b) => b.s - a.s || String(a.e.work.id).localeCompare(String(b.e.work.id)))
      .map((x) => x.e.work);
    const gainOf = (w) => entry.gains.get(w.id) || 0;
    return { ndcg20: ndcgAt(ranked, gainOf, 20), ndcg10: ndcgAt(ranked, gainOf, 10) };
  });
  console.log(`${label.padEnd(30)} nDCG@20 ${mean(perQuery.map((q) => q.ndcg20)).toFixed(4)}   nDCG@10 ${mean(perQuery.map((q) => q.ndcg10)).toFixed(4)}`);
}

console.log("");
for (const [depth, termCount, weight] of [[5, 40, 0.8], [5, 40, 0.3], [5, 10, 0.8]]) {
  evaluateFeedback(`prf d=${depth} t=${termCount} w=${weight}`, { depth, terms: termCount, weight });
}

// Retrieval parameters. `tune` only.
console.log("");
function evaluateParams(label, params, feedback = null) {
  const index = buildRelevanceIndex(docs, { space: null, ...params });
  const perQuery = queries.map((entry) => {
    const expanded = feedback ? withFeedback(index, entry, feedback) : expandQuery(entry.query, index);
    const ranked = index.entries
      .map((e) => ({ e, s: scoreEntry(e, expanded, index, params).raw }))
      .sort((a, b) => b.s - a.s || String(a.e.work.id).localeCompare(String(b.e.work.id)))
      .map((x) => x.e.work);
    const gainOf = (w) => entry.gains.get(w.id) || 0;
    return { ndcg20: ndcgAt(ranked, gainOf, 20), ndcg10: ndcgAt(ranked, gainOf, 10) };
  });
  console.log(`${label.padEnd(38)} nDCG@20 ${mean(perQuery.map((q) => q.ndcg20)).toFixed(4)}   nDCG@10 ${mean(perQuery.map((q) => q.ndcg10)).toFixed(4)}`);
}
for (const titleWeight of [0, 1.2, 2.6, 4]) {
  for (const coverage of [0, 0.45, 0.8]) {
    evaluateParams(`title=${titleWeight} cov=${coverage}`, { titleWeight, coverageWeight: coverage });
  }
}
for (const k1 of [0.9, 1.4, 2.0]) {
  for (const b of [0.4, 0.72, 0.9]) {
    evaluateParams(`k1=${k1} b=${b}`, { k1, b });
  }
}


// Hybrid: BM25 plus similarity between the query and the document in the corpus's
// own semantic space. A topic name and the papers under it often share no words
// at all, which is precisely the case a vector space is supposed to cover.
import { documentVector, semanticSimilarity, removeCommonComponent } from "../../src/shared/semantic.js";
import { bm25Weights } from "../../src/shared/text.js";

const avgLen = terms.reduce((s2, t) => s2 + t.length, 0) / terms.length;
const docVectors = docs.map((w, i) => documentVector(terms[i], space, lexicon, avgLen));
removeCommonComponent(docVectors, space.dimensions);
const vectorById = new Map(docs.map((w, i) => [w.id, docVectors[i]]));

function evaluateHybrid(label, lambda, feedback) {
  const index = buildRelevanceIndex(docs, { space: null });
  const perQuery = queries.map((entry) => {
    const expanded = feedback ? withFeedback(index, entry, feedback) : expandQuery(entry.query, index);
    const queryVector = documentVector(contentTerms(entry.query), space, lexicon, avgLen);
    const scored = index.entries.map((e) => {
      const lexical = scoreEntry(e, expanded, index).raw;
      const semantic = semanticSimilarity(queryVector, vectorById.get(e.work.id) || { norm: 0 });
      return { e, s: (1 - lambda) * (1 - Math.exp(-lexical / 9)) + lambda * semantic };
    });
    const ranked = scored.sort((a, b) => b.s - a.s || String(a.e.work.id).localeCompare(String(b.e.work.id))).map((x) => x.e.work);
    const gainOf = (w) => entry.gains.get(w.id) || 0;
    return { ndcg20: ndcgAt(ranked, gainOf, 20), ndcg10: ndcgAt(ranked, gainOf, 10) };
  });
  console.log(`${label.padEnd(38)} nDCG@20 ${mean(perQuery.map((q) => q.ndcg20)).toFixed(4)}   nDCG@10 ${mean(perQuery.map((q) => q.ndcg10)).toFixed(4)}`);
}

console.log("");
for (const lambda of [0, 0.15, 0.3, 0.45, 0.6, 0.8, 1]) {
  evaluateHybrid(`hybrid lambda=${lambda}`, lambda, null);
}
console.log("");
for (const lambda of [0.15, 0.3, 0.45]) {
  evaluateHybrid(`hybrid+prf lambda=${lambda}`, lambda, { depth: 5, terms: 40, weight: 0.3 });
}
