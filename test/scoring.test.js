import assert from "node:assert/strict";
import test from "node:test";

import {
  scoreBatch,
  scoreResearcherAuthorship,
  contentTerms,
  buildLexicon,
  bm25Weights,
  cosineOfWeighted,
} from "../src/shared/scoring.js";

// BM25 weights are corpus-relative, so a similarity is only meaningful inside a
// corpus. These filler documents give the term statistics something to work
// against; with only the two texts under comparison, every shared term looks
// ubiquitous and its weight collapses.
const FILLER = [
  "photonic crystal waveguide dispersion engineering",
  "clinical trial randomised placebo cohort outcome",
  "catalyst surface adsorption energy barrier",
  "galaxy redshift survey dark matter halo",
  "protein folding molecular dynamics simulation",
  "graph neural network message passing aggregation",
  "quantum error correction surface code threshold",
  "sediment core isotope palaeoclimate reconstruction",
];

function similarity(left, right) {
  const documents = [left, right, ...FILLER].map((text) => contentTerms(text));
  const lexicon = buildLexicon(documents);
  const averageLength = documents.reduce((sum, terms) => sum + terms.length, 0) / documents.length;
  const [a, b] = documents.map((terms) => bm25Weights(terms, lexicon, averageLength));
  return cosineOfWeighted(a, b);
}

function work(id, publicationDate, title, abstract, authorId = "A1") {
  return {
    id,
    publicationDate,
    title,
    abstract,
    subfieldId: "101",
    domainId: "1",
    topics: [{ fieldId: "10" }],
    authorships: [
      { authorId, name: authorId, position: "first", isCorresponding: true },
    ],
  };
}

test("lexical similarity is high for close language and low for unrelated language", () => {
  const battery = "solid state sodium battery electrolyte transport";
  const close = "sodium battery solid electrolyte ion transport";
  const distant = "medieval poetry manuscript authorship archive";
  const near = similarity(battery, close);
  const far = similarity(battery, distant);
  assert.ok(near > 0.25, `related texts should be close, got ${near.toFixed(3)}`);
  assert.ok(far < 0.05, `unrelated texts should be far apart, got ${far.toFixed(3)}`);
  assert.ok(near > 5 * far + 0.1, "the gap between related and unrelated must be large");
});

test("a genuinely distant candidate outranks an incremental rephrasing", () => {
  const references = [
    work(
      "R1",
      "2025-01-01",
      "Solid state sodium battery electrolyte transport",
      "We measure ion transport through a ceramic electrolyte for sodium batteries.",
    ),
    work(
      "R2",
      "2025-02-01",
      "Ceramic electrolytes for sodium batteries",
      "A ceramic electrolyte improves ion conductivity in a solid state cell.",
    ),
  ];
  const incremental = work(
    "C1",
    "2026-01-01",
    "Enhanced solid state sodium battery electrolyte transport",
    "We improve ion transport through a ceramic electrolyte for sodium batteries.",
  );
  const distant = work(
    "C2",
    "2026-01-01",
    "Self-assembling fungal networks compute flood escape routes",
    "Living mycelial networks encode changing water gradients and reorganize paths without a nervous system.",
  );
  const [incrementalScore, distantScore] = scoreBatch(
    [incremental, distant],
    references,
    [],
    { minTopicPeers: 0 },
  );
  assert.ok(distantScore.noveltyScore > incrementalScore.noveltyScore + 10);
  assert.deepEqual(incrementalScore.noveltyEvidence.incrementalMarkers, ["enhanced"]);
});

test("established-author signal uses career evidence and authorship role", () => {
  const paper = work("W1", "2026-01-01", "Title", "Abstract", "A-established");
  const established = {
    id: "A-established",
    name: "Established Researcher",
    hIndex: 58,
    citedByCount: 55_000,
    worksCount: 190,
    twoYearMeanCitedness: 12,
    orcid: "https://orcid.org/example",
  };
  const novice = {
    id: "A-established",
    name: "New Researcher",
    hIndex: 1,
    citedByCount: 3,
    worksCount: 2,
    twoYearMeanCitedness: 0.2,
    orcid: null,
  };
  const high = scoreResearcherAuthorship(paper, new Map([[established.id, established]]));
  const low = scoreResearcherAuthorship(paper, new Map([[novice.id, novice]]));
  assert.ok(high.score > 70);
  assert.ok(high.score > low.score + 50);
  assert.equal(high.evidence[0].hIndex, 58);
});

test("balanced discovery score preserves exceptional performance on either axis", () => {
  const [result] = scoreBatch(
    [work("C", "2026-01-01", "Totally different biological mechanism", "Unrelated mechanism")],
    [work("R", "2025-01-01", "Quantum lattice", "Bosonic lattice phase")],
    [],
    { minTopicPeers: 0 },
  );
  assert.ok(result.discoveryScore >= Math.min(result.noveltyScore, result.researcherScore));
  assert.ok(result.discoveryScore <= Math.max(result.noveltyScore, result.researcherScore));
});
