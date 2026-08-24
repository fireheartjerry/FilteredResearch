import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import { interestMatchEvidence, matchesResearchFilters } from "../src/shared/filters.js";
import { scoreBatch } from "../src/shared/scoring.js";
import { groupDuplicatePapers } from "../src/shared/papers.js";
import { applySelectivity } from "../src/shared/ranking.js";

const matches = (query, text) => Boolean(interestMatchEvidence({ title: "x", abstract: text }, [query]));

test("a short query never matches the inside of an unrelated word", () => {
  // The fast path was a raw substring test, so "RAG" matched storage, average,
  // fragment, paragraph and diaphragm, and "AI" matched chain and plain.
  for (const text of [
    "storage devices for computing",
    "average pooling layers",
    "a fragment of the signal",
    "see paragraph three",
    "the diaphragm response",
  ]) {
    assert.equal(matches("RAG", text), false, `RAG should not match: ${text}`);
  }
  for (const text of ["a chain of custody", "the plain text", "maintaining balance"]) {
    assert.equal(matches("AI", text), false, `AI should not match: ${text}`);
  }
  assert.equal(matches("RAG", "we study RAG pipelines"), true);
  assert.equal(matches("RAG", "a RAG-based retrieval system"), true);
});

test("a known abbreviation finds its expansion and vice versa", () => {
  assert.equal(matches("RAG", "we present a retrieval augmented generation pipeline"), true);
  assert.equal(matches("RAG", "retrieval-augmented generation for question answering"), true);
  assert.equal(matches("retrieval augmented generation", "we evaluate RAG pipelines"), true);
  assert.equal(matches("large language models", "LLM reasoning benchmarks"), true);
  for (const [abbrev, expansion] of [
    ["LLM", "large language models are powerful"],
    ["RLHF", "reinforcement learning from human feedback"],
    ["MoE", "a mixture of experts architecture"],
    ["SAE", "sparse autoencoders for interpretability"],
    ["NLP", "natural language processing tasks"],
    ["CNN", "convolutional neural networks for vision"],
  ]) {
    assert.equal(matches(abbrev, expansion), true, `${abbrev} should find "${expansion}"`);
  }
});

test("a known abbreviation does not match a coincidental expansion", () => {
  // "RAG" means retrieval-augmented generation, not any phrase whose initials
  // happen to line up.
  assert.equal(matches("RAG", "robust adaptive gradient descent methods"), false);
  assert.equal(matches("CNN", "certain nested numerical notation"), false);
  assert.equal(matches("LLM", "the small linear model was used"), false);
  assert.equal(matches("RAG", "a study of medieval crop rotation"), false);
});

test("one malformed stored record cannot take down a whole screen", () => {
  const rows = [
    { id: "ok", title: "Retrieval augmented generation", abstract: "RAG pipelines", language: "en", subfieldId: "1702" },
    {},
    { id: "weird", title: null, abstract: null, language: "en" },
    { id: "nested", title: "x", abstract: "y", language: "en", authorships: null, sources: null, topics: null },
  ];
  const grouped = groupDuplicatePapers(rows);
  assert.ok(grouped.length >= 3);
  let threw = 0;
  for (const work of grouped) {
    try {
      matchesResearchFilters(work, { englishOnly: true });
    } catch {
      threw += 1;
    }
  }
  assert.equal(threw, 0, "filtering must not throw on malformed records");
  assert.equal(matchesResearchFilters(null, { englishOnly: true }), false);
  assert.equal(interestMatchEvidence(null, ["rag"]), null);
});

test("scoring survives degenerate input without NaN or out-of-range scores", () => {
  const cases = [
    [[], []],
    [[{ id: "a" }], []],
    [[{ id: "a", title: "", abstract: "" }], [{ id: "r", title: "", abstract: "" }]],
    [[null, { id: "b", title: "x", abstract: "y", publicationDate: "2026-01-01" }], []],
  ];
  for (const [candidates, references] of cases) {
    const scored = scoreBatch(candidates, references, []);
    for (const work of scored) {
      for (const key of ["noveltyScore", "researcherScore", "discoveryScore"]) {
        const value = work[key];
        assert.ok(Number.isFinite(value) && value >= 0 && value <= 100, `${key}=${value}`);
      }
    }
  }
  assert.deepEqual(scoreBatch(null, null, null), []);
  assert.deepEqual(applySelectivity(null, null).works, []);
});

test("novelty separates consolidation work from genuinely new work", async () => {
  let seed = 5;
  const rnd = () => { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return seed / 4294967296; };
  const SENTENCES = [
    "We propose a retrieval augmented generation pipeline that indexes a document corpus.",
    "Our method improves recall on standard question answering benchmarks.",
    "We evaluate several embedding models and rerankers across datasets.",
    "Experiments show that chunk size affects answer accuracy materially.",
    "We report precision, recall and exact match on established datasets.",
    "A vector store with approximate nearest neighbour search provides passage lookup.",
  ];
  const abstractOf = (n) => Array.from({ length: n }, () => SENTENCES[Math.floor(rnd() * SENTENCES.length)]).join(" ");
  // Real OpenAlex records carry a reference list, and most of the scoring signals
  // read it. A fixture without one exercises only the text fallback, so these
  // papers get bibliographies: the field's own papers cite the field's canon, the
  // survey cites a wide slice of it, and the new idea cites few of them.
  const CANON = Array.from({ length: 40 }, (_, i) => `CANON${i}`);
  const mk = (id, title, abstract, date, references = null) => ({
    id, title, abstract, subfieldId: "1702", domainId: "1",
    publicationDate: date, authorships: [{ authorId: `A${id}`, position: "first" }],
    topics: [{ fieldId: "17", topicId: "T1", score: 0.9 }],
    referencedWorks: references || CANON.slice(0, 10),
  });
  const references = Array.from({ length: 160 }, (_, i) =>
    mk(`R${i}`, `retrieval passage ranking ${i}`, abstractOf(8), "2024-01-01", CANON.slice(i % 20, (i % 20) + 12)));
  const slop = mk("slop", "Retrieval Augmented Generation: A Comprehensive Survey",
    "We present a comprehensive survey of retrieval augmented generation methods. We review indexing, passage retrieval and reranking, and compare embedding models across established benchmark datasets.", "2026-01-01", CANON.slice(0, 34));
  const novel = mk("novel", "Attention Is All You Need",
    "We propose the Transformer, a network architecture based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on machine translation show these models superior in quality while being more parallelizable.", "2026-01-01",
    ["OUTSIDE1", "OUTSIDE2", "OUTSIDE3", "OUTSIDE4", "OUTSIDE5", CANON[3]]);
  const ordinary = Array.from({ length: 24 }, (_, i) =>
    mk(`ord${i}`, `passage reranking index ${i}`, abstractOf(7), "2026-01-01", CANON.slice(i % 18, (i % 18) + 11)));

  const scored = scoreBatch([slop, novel, ...ordinary], references, []);
  const scoreOf = (id) => scored.find((w) => w.id === id).noveltyScore;

  // The whole point of the sliders: these must not land on the same number.
  assert.ok(scoreOf("novel") - scoreOf("slop") >= 8,
    `novel ${Math.round(scoreOf("novel"))} vs survey ${Math.round(scoreOf("slop"))}`);
  // A declared survey carries an explicit consolidation penalty.
  assert.ok(scored.find((w) => w.id === "slop").noveltyEvidence.consolidationPrior > 0);
  assert.equal(scored.find((w) => w.id === "novel").noveltyEvidence.consolidationPrior, 0);
  // And the field as a whole must use the range, not bunch at the top.
  const all = scored.map((w) => w.noveltyScore);
  assert.ok(Math.max(...all) - Math.min(...all) > 20, `range ${Math.min(...all)}-${Math.max(...all)}`);
});

test("no source file carries control characters", async () => {
  // A shell heredoc once turned every \\b in a regex into a literal backspace
  // byte, silently disabling the survey and benchmark detection.
  for (const file of [
    "../src/shared/scoring.js",
    "../src/shared/novelty.js",
    "../src/shared/relevance.js",
    "../src/shared/semantic.js",
    "../src/shared/text.js",
    "../src/shared/filters.js",
    "../src/shared/ranking.js",
    "../src/shared/papers.js",
    "../src/shared/defaults.js",
    "../src/background/service-worker.js",
  ]) {
    const text = await readFile(new URL(file, import.meta.url), "utf8");
    const bad = [...text].filter((ch) => {
      const code = ch.codePointAt(0);
      return code < 9 || (code > 10 && code < 13) || (code > 13 && code < 32);
    });
    assert.equal(bad.length, 0, `${file} contains ${bad.length} control character(s)`);
  }
});
