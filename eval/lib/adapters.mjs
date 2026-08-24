// Adapters put every ranker behind one interface so the harness can score the
// shipped code and the frozen v1.0.1 baseline through exactly the same path.
//
// The sanitising helpers here are load-bearing for benchmark integrity: they are
// the single place that guarantees a ranker cannot reach the labels.

import { gunzipSync } from "node:zlib";
import { readFileSync, existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
export const FIXTURES = join(HERE, "..", "fixtures");

export function loadFixture(name) {
  const path = join(FIXTURES, `${name}.json.gz`);
  if (!existsSync(path)) {
    throw new Error(`Missing fixture ${path}. Run: node eval/fetch-fixtures.mjs (needs network, once).`);
  }
  return JSON.parse(gunzipSync(readFileSync(path)).toString());
}

// Remove every post-publication field. Anything a ranker could use to cheat has
// to be deleted here, not merely "not read" by the current implementation.
export function sanitize(work) {
  const { outcome, split, ...rest } = work;
  return rest;
}

// For the relevance benchmark the labels come from OpenAlex topic assignment, so
// a relevance ranker must see none of it -- not the names it could match a query
// against, and not the ids it could cluster on to propagate relevance between
// papers that share a label.
export function stripTopicEvidence(work) {
  const clean = sanitize(work);
  return {
    ...clean,
    topicId: null,
    topicName: null,
    subfieldName: null,
    fieldName: null,
    topics: [],
  };
}

export function loadCorpus({ forRelevance = false } = {}) {
  const candidates = loadFixture("candidates");
  const peers = loadFixture("peers");
  const authors = loadFixture("authors");
  const prepare = forRelevance ? stripTopicEvidence : sanitize;
  return {
    rawCandidates: candidates,
    candidates: candidates.map(prepare),
    peers: peers.map(prepare),
    authors,
  };
}

// v1.0.1, imported verbatim from git history (see eval/baseline/README.md and
// the guard in test/eval-baseline.test.js). Never edited to make a comparison
// look better.
export async function baselineAdapter() {
  const scoring = await import("../baseline/scoring.js");
  const ranking = await import("../baseline/ranking.js");
  const filters = await import("../baseline/filters.js");
  return {
    name: "v1.0.1 baseline",
    scoreBatch: (candidates, peers, authors, options) => scoring.scoreBatch(candidates, peers, authors, options),
    // The baseline has no relevance score at all: interest matching is a
    // boolean. Exposing it as 1/0 is the fairest possible reading of it.
    relevanceScore: (work, query) => (filters.matchesInterestQuery(work, query) ? 1 : 0),
    applySelectivity: ranking.applySelectivity,
  };
}

export async function currentAdapter() {
  const scoring = await import("../../src/shared/scoring.js");
  const ranking = await import("../../src/shared/ranking.js");
  const filters = await import("../../src/shared/filters.js");
  return {
    name: "current",
    scoreBatch: (candidates, peers, authors, options) => scoring.scoreBatch(candidates, peers, authors, options),
    relevanceScore: (work, query) => {
      // Once a graded relevance score exists it is used directly; until then the
      // boolean is read the same way the baseline's is.
      if (typeof filters.relevanceScore === "function") return filters.relevanceScore(work, query);
      return filters.matchesInterestQuery(work, query) ? 1 : 0;
    },
    // A ranker may expose a corpus-aware relevance pass; the harness prefers it
    // when present because query-level statistics (BM25 and friends) cannot be
    // computed one document at a time.
    rankByRelevance:
      (await import("../../src/shared/filters.js")).rankByRelevance || null,
    applySelectivity: ranking.applySelectivity,
  };
}
