// npm run eval:adversarial -- construct validity. A novelty score can post a
// good nDCG while measuring the wrong thing entirely, so each probe below
// changes one property of a paper and asserts the score responds the way a
// novelty measure should.

import { writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { loadCorpus, currentAdapter } from "./lib/adapters.mjs";
import { auc, mean, quantile, seededRandom } from "./lib/metrics.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const SAMPLE = 120;

// Nonsense that no corpus has ever seen. Mean-IDF style "distinctiveness"
// treats these as maximally informative; a real novelty measure must not.
const JUNK = [
  "zylophrantic", "quorbulent", "flimtazine", "brendorial", "yaxomeric", "plindrous",
  "trovanquil", "skemulate", "narvithic", "wupendral", "glorbanate", "thrixomere",
  "vandusil", "prelkovian", "morantific", "chelvurate", "dwintaric", "sporlaxene",
  "kravonium", "belquithic", "hastrolene", "junderal", "vexomantic", "pilthorate",
  "grendulous", "xanthiplex", "orbidacene", "tumbrelith", "wrenfastic", "zoblarine",
];

const SYNONYMS = new Map(Object.entries({
  method: "approach", methods: "approaches", propose: "introduce", proposed: "introduced",
  results: "findings", result: "finding", improve: "enhance", improved: "enhanced",
  model: "framework", models: "frameworks", dataset: "data collection", datasets: "data collections",
  novel: "new", significant: "substantial", experiments: "trials", experiment: "trial",
  performance: "effectiveness", accuracy: "correctness", training: "fitting",
  evaluate: "assess", evaluated: "assessed", demonstrate: "show", demonstrates: "shows",
  present: "describe", presents: "describes", show: "indicate", shows: "indicates",
  using: "employing", based: "founded", approach: "technique", technique: "procedure",
  analysis: "examination", study: "investigation", high: "elevated", low: "reduced",
}));

function paraphrase(text) {
  return String(text || "")
    .split(/(\s+)/)
    .map((token) => {
      const bare = token.toLowerCase().replace(/[^a-z]/g, "");
      const swap = SYNONYMS.get(bare);
      if (!swap) return token;
      return token.replace(/[A-Za-z]+/, swap);
    })
    .join("");
}

function scoresById(scored) {
  return new Map(scored.map((work) => [work.id, work.noveltyScore || 0]));
}

async function main() {
  const adapter = await currentAdapter();
  const corpus = loadCorpus();
  const random = seededRandom(99);

  const withText = corpus.candidates.filter((work) => (work.abstract || "").length > 400);
  const sample = [];
  const used = new Set();
  while (sample.length < Math.min(SAMPLE, withText.length)) {
    const pick = withText[Math.floor(random() * withText.length)];
    if (used.has(pick.id)) continue;
    used.add(pick.id);
    sample.push(pick);
  }

  const base = scoresById(adapter.scoreBatch(corpus.candidates, corpus.peers, corpus.authors));
  const probes = {};

  // 1 -- junk tokens. Rare vocabulary that carries no idea.
  //
  // Each paper gets its OWN nonsense, because that is what the real failure looks
  // like: OCR debris, transliteration artefacts and typos are unique to the
  // document they appear in. Handing every paper the same thirty invented words
  // would not be noise at all -- it would be a hundred and twenty authors
  // independently adopting the same new vocabulary, which is what emerging
  // terminology *is*. That case is measured separately below.
  const junked = corpus.candidates.map((work, index) =>
    used.has(work.id)
      ? { ...work, abstract: `${work.abstract} ${JUNK.map((token) => `${token}${index}`).join(" ")}` }
      : work,
  );
  const junkScores = scoresById(adapter.scoreBatch(junked, corpus.peers, corpus.authors));
  const junkDeltas = sample.map((work) => (junkScores.get(work.id) || 0) - (base.get(work.id) || 0));
  probes.junk_tokens = {
    mean_delta: Number(mean(junkDeltas).toFixed(2)),
    p95_delta: Number(quantile([...junkDeltas].sort((a, b) => a - b), 0.95).toFixed(2)),
    threshold: 3,
    pass: mean(junkDeltas) < 3,
  };

  // 1b -- the same invented words adopted across many papers at once. This is
  // supposed to move the score: it is indistinguishable from a field picking up
  // new terminology. What must not happen is an unbounded move, so the bar is
  // that it stays within the range a real signal could occupy.
  const coordinated = corpus.candidates.map((work) =>
    used.has(work.id) ? { ...work, abstract: `${work.abstract} ${JUNK.join(" ")}` } : work,
  );
  const coordinatedScores = scoresById(adapter.scoreBatch(coordinated, corpus.peers, corpus.authors));
  const coordinatedDeltas = sample.map((work) => (coordinatedScores.get(work.id) || 0) - (base.get(work.id) || 0));
  probes.coordinated_vocabulary = {
    mean_delta: Number(mean(coordinatedDeltas).toFixed(2)),
    threshold: 25,
    pass: mean(coordinatedDeltas) < 25,
    note: "Expected to move the score. Bounded, not zero.",
  };

  // 2 -- paraphrase. Same idea, different words.
  const reworded = corpus.candidates.map((work) =>
    used.has(work.id) ? { ...work, title: paraphrase(work.title), abstract: paraphrase(work.abstract) } : work,
  );
  const paraphraseScores = scoresById(adapter.scoreBatch(reworded, corpus.peers, corpus.authors));
  const paraphraseDeltas = sample.map((work) => Math.abs((paraphraseScores.get(work.id) || 0) - (base.get(work.id) || 0)));
  probes.paraphrase_invariance = {
    mean_abs_delta: Number(mean(paraphraseDeltas).toFixed(2)),
    p95_abs_delta: Number(quantile([...paraphraseDeltas].sort((a, b) => a - b), 0.95).toFixed(2)),
    threshold: 8,
    pass: mean(paraphraseDeltas) < 8,
  };

  // 3 -- true duplicates. A verbatim restatement of existing work is the least
  // novel thing that can enter a corpus.
  const peerSample = corpus.peers.filter((peer) => (peer.abstract || "").length > 400).slice(0, 60);
  const clones = peerSample.map((peer, index) => ({
    ...peer,
    id: `CLONE${index}`,
    publicationDate: "2021-08-01",
    isBaseline: false,
  }));
  const withClones = adapter.scoreBatch([...corpus.candidates, ...clones], corpus.peers, corpus.authors);
  const cloneScores = withClones.filter((work) => String(work.id).startsWith("CLONE")).map((work) => work.noveltyScore || 0);
  const allScores = withClones.map((work) => work.noveltyScore || 0).sort((a, b) => a - b);
  const tenthPercentile = quantile(allScores, 0.1);
  const clonesBelow = cloneScores.filter((score) => score <= tenthPercentile).length;
  probes.true_duplicates = {
    clones: cloneScores.length,
    median_clone_score: Number(quantile([...cloneScores].sort((a, b) => a - b), 0.5).toFixed(2)),
    corpus_p10: Number(tenthPercentile.toFixed(2)),
    fraction_in_bottom_decile: Number((clonesBelow / Math.max(1, cloneScores.length)).toFixed(3)),
    threshold: 0.8,
    pass: clonesBelow / Math.max(1, cloneScores.length) >= 0.8,
  };

  // 4 -- consolidation. Reviews should rank low, and they should still rank low
  // when the keyword patterns are switched off, otherwise the algorithm is
  // matching the word "survey" rather than the shape of the work.
  const reviews = corpus.rawCandidates.filter((work) => work.workType === "review");
  const articles = corpus.rawCandidates.filter((work) => work.workType !== "review");
  const reviewScores = reviews.map((work) => base.get(work.id) || 0);
  const articleScores = articles.map((work) => base.get(work.id) || 0);
  const withKeywords = auc(articleScores, reviewScores);

  let withoutKeywords = null;
  try {
    const stripped = scoresById(
      adapter.scoreBatch(corpus.candidates, corpus.peers, corpus.authors, { disableLexicalConsolidationPrior: true }),
    );
    withoutKeywords = auc(
      articles.map((work) => stripped.get(work.id) || 0),
      reviews.map((work) => stripped.get(work.id) || 0),
    );
  } catch {
    withoutKeywords = null;
  }
  const medianGap =
    quantile([...articleScores].sort((a, b) => a - b), 0.5) - quantile([...reviewScores].sort((a, b) => a - b), 0.5);
  probes.consolidation = {
    reviews: reviews.length,
    median_gap_points: Number(medianGap.toFixed(2)),
    auc_with_keywords: Number(withKeywords.toFixed(4)),
    auc_without_keywords: withoutKeywords === null ? null : Number(withoutKeywords.toFixed(4)),
    auc_drop: withoutKeywords === null ? null : Number((withKeywords - withoutKeywords).toFixed(4)),
    pass:
      medianGap >= 20 &&
      withoutKeywords !== null &&
      withKeywords - withoutKeywords < 0.05,
  };

  // 5 -- thin records. A title-only or non-English record knows less, so it must
  // not be rewarded for the vocabulary it happens not to share with the field.
  const thin = corpus.rawCandidates.filter((work) => (work.abstract || "").length < 120);
  const full = corpus.rawCandidates.filter((work) => (work.abstract || "").length >= 600);
  const thinMean = mean(thin.map((work) => base.get(work.id) || 0));
  const fullMean = mean(full.map((work) => base.get(work.id) || 0));
  probes.thin_records = {
    thin_count: thin.length,
    thin_mean: Number(thinMean.toFixed(2)),
    full_mean: Number(fullMean.toFixed(2)),
    excess: Number((thinMean - fullMean).toFixed(2)),
    threshold: 4,
    pass: thinMean - fullMean < 4,
  };

  const passed = Object.values(probes).filter((probe) => probe.pass).length;
  const report = { generatedAt: new Date().toISOString(), sample: sample.length, probes, passed, total: Object.keys(probes).length };
  writeFileSync(join(HERE, "adversarial.json"), `${JSON.stringify(report, null, 2)}\n`);

  for (const [name, probe] of Object.entries(probes)) {
    console.log(`${probe.pass ? "ok  " : "FAIL"} ${name.padEnd(24)} ${JSON.stringify({ ...probe, pass: undefined })}`);
  }
  console.log(`\n${passed}/${report.total} construct-validity probes pass\n`);
  process.exitCode = passed === report.total ? 0 : 1;
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
