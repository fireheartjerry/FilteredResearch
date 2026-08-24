// npm run eval -- the primary benchmark. Fully offline; reads eval/fixtures/.
//
// Reports the shipped algorithm and the frozen v1.0.1 baseline on the identical
// held-out split, so every number in the criteria doc has something to be
// compared against.

import { writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { readFileSync, existsSync } from "node:fs";
import { loadCorpus, baselineAdapter, currentAdapter, FIXTURES } from "./lib/adapters.mjs";
import { buildLabels, gainFromLabel, disruptionClasses, buildRelevanceQueries, QUALITATIVE_QUERIES, LABEL_NOTES } from "./lib/labels.mjs";
import { ndcgAt, spearman, auc, meanReciprocalRank, recallAt, precisionAt, mean } from "./lib/metrics.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const REPORT_PATH = join(HERE, "report.json");

const SPLIT = process.env.FR_EVAL_SPLIT || "test";

function round(value, places = 4) {
  return Number.isFinite(value) ? Number(value.toFixed(places)) : null;
}

async function evaluateRanker(adapter, corpus, labels, queries) {
  const started = Date.now();
  const scored = adapter.scoreBatch(corpus.candidates, corpus.peers, corpus.authors);
  const scoringMs = Date.now() - started;
  const byId = new Map(scored.map((work) => [work.id, work]));

  const inSplit = corpus.rawCandidates.filter((work) => work.split === SPLIT && byId.has(work.id));

  // --- novelty ---
  const noveltyEvaluable = inSplit.filter((work) => labels.get(work.id)?.noveltyLabel !== null);
  const noveltyRanked = [...noveltyEvaluable].sort(
    (left, right) => (byId.get(right.id).noveltyScore || 0) - (byId.get(left.id).noveltyScore || 0),
  );
  const noveltyGain = (work) => gainFromLabel(labels.get(work.id).noveltyLabel);
  const classes = disruptionClasses(inSplit, labels);
  const novelty = {
    evaluated: noveltyEvaluable.length,
    ndcg_at_50: round(ndcgAt(noveltyRanked, noveltyGain, 50)),
    ndcg_at_100: round(ndcgAt(noveltyRanked, noveltyGain, 100)),
    precision_at_20: round(precisionAt(noveltyRanked, (work) => noveltyGain(work) >= 2, 20)),
    spearman: round(spearman(
      noveltyEvaluable.map((work) => byId.get(work.id).noveltyScore || 0),
      noveltyEvaluable.map((work) => labels.get(work.id).noveltyLabel),
    )),
    auc_disruptive_vs_derivative: round(auc(
      classes.positives.map((work) => byId.get(work.id)?.noveltyScore || 0),
      classes.negatives.map((work) => byId.get(work.id)?.noveltyScore || 0),
    )),
    // The headline AUC pools two very different negatives. Reviews are a class the
    // consolidation machinery was explicitly built to catch, so leaving them in
    // makes the number look better than the question its name asks. Both halves
    // are reported; the derivative-only figure is the honest answer to
    // "disruptive versus derivative research".
    auc_vs_reviews_only: round(auc(
      classes.positives.map((work) => byId.get(work.id)?.noveltyScore || 0),
      classes.negatives.filter((work) => work.workType === "review").map((work) => byId.get(work.id)?.noveltyScore || 0),
    )),
    auc_vs_derivative_articles: round(auc(
      classes.positives.map((work) => byId.get(work.id)?.noveltyScore || 0),
      classes.negatives.filter((work) => work.workType !== "review").map((work) => byId.get(work.id)?.noveltyScore || 0),
    )),
    positives: classes.positives.length,
    negatives: classes.negatives.length,
    negatives_reviews: classes.negatives.filter((work) => work.workType === "review").length,
    negatives_articles: classes.negatives.filter((work) => work.workType !== "review").length,
    review_gap_points: round(
      mean(inSplit.filter((work) => work.workType !== "review").map((work) => byId.get(work.id).noveltyScore || 0)) -
      mean(inSplit.filter((work) => work.workType === "review").map((work) => byId.get(work.id).noveltyScore || 0)),
      2,
    ),
    score_spread: {
      min: round(Math.min(...inSplit.map((work) => byId.get(work.id).noveltyScore || 0)), 2),
      max: round(Math.max(...inSplit.map((work) => byId.get(work.id).noveltyScore || 0)), 2),
      distinct_values: new Set(inSplit.map((work) => Math.round(byId.get(work.id).noveltyScore || 0))).size,
    },
  };

  // --- authorship ---
  const authorLabel = new Map();
  const authorById = new Map(corpus.authors.map((author) => [author.id, author]));
  for (const work of inSplit) {
    const standing = Math.max(
      0,
      ...(work.authorships || []).map((authorship) => authorById.get(authorship.authorId)?.citedByCount || 0),
    );
    authorLabel.set(work.id, Math.log1p(standing));
  }
  const fieldMeans = new Map();
  for (const work of inSplit) {
    if (!fieldMeans.has(work.benchField)) fieldMeans.set(work.benchField, []);
    fieldMeans.get(work.benchField).push(byId.get(work.id).researcherScore || 0);
  }
  const perFieldMean = [...fieldMeans.values()].map(mean);
  const authorship = {
    spearman: round(spearman(
      inSplit.map((work) => byId.get(work.id).researcherScore || 0),
      inSplit.map((work) => authorLabel.get(work.id)),
    )),
    // If the same score means different things in medicine and mathematics, the
    // field means drift apart. Expressed as a fraction of the 100-point scale.
    field_bias_gap: round((Math.max(...perFieldMean) - Math.min(...perFieldMean)) / 100),
    per_field_mean: Object.fromEntries([...fieldMeans].map(([field, values]) => [field, round(mean(values), 2)])),
  };

  // --- combined ordering ---
  const combinedGain = (work) => gainFromLabel(labels.get(work.id).combinedLabel);
  const orderBy = (key) => [...inSplit].sort((left, right) => (byId.get(right.id)[key] || 0) - (byId.get(left.id)[key] || 0));
  const combined = {
    ndcg_at_50: round(ndcgAt(orderBy("discoveryScore"), combinedGain, 50)),
    novelty_alone_ndcg_at_50: round(ndcgAt(orderBy("noveltyScore"), combinedGain, 50)),
    authorship_alone_ndcg_at_50: round(ndcgAt(orderBy("researcherScore"), combinedGain, 50)),
  };
  combined.beats_both_parts =
    combined.ndcg_at_50 > combined.novelty_alone_ndcg_at_50 &&
    combined.ndcg_at_50 > combined.authorship_alone_ndcg_at_50;

  // Complementarity measured in both directions. A single label cannot show it:
  // authorship dominates the citation label by construction, so the question is
  // whether the blend carries something each part alone does not, judged on the
  // label that part is worst at.
  const orderCombinedByNovelty = [...noveltyEvaluable].sort(
    (left, right) => (byId.get(right.id).discoveryScore || 0) - (byId.get(left.id).discoveryScore || 0),
  );
  const orderAuthorshipByNovelty = [...noveltyEvaluable].sort(
    (left, right) => (byId.get(right.id).researcherScore || 0) - (byId.get(left.id).researcherScore || 0),
  );
  combined.combined_ndcg_on_novelty_label = round(ndcgAt(orderCombinedByNovelty, noveltyGain, 50));
  combined.authorship_alone_ndcg_on_novelty_label = round(ndcgAt(orderAuthorshipByNovelty, noveltyGain, 50));
  combined.beats_novelty_alone_on_citations = combined.ndcg_at_50 > combined.novelty_alone_ndcg_at_50;
  combined.beats_authorship_alone_on_novelty =
    combined.combined_ndcg_on_novelty_label > combined.authorship_alone_ndcg_on_novelty_label;

  // --- relevance ---
  const relevanceCorpus = corpus.relevanceCandidates.filter((work) => {
    const source = corpus.rawById.get(work.id);
    return source && source.split === SPLIT;
  });
  const perQuery = [];
  for (const entry of queries) {
    const ranked = await rankForQueryAsync(adapter, relevanceCorpus, entry.query, corpus.useModel);
    const gainOf = (work) => entry.gains.get(work.id) || 0;
    const relevant = (work) => gainOf(work) >= 2;
    perQuery.push({
      query: entry.query,
      relevant: entry.relevantCount,
      ndcg_at_20: ndcgAt(ranked, gainOf, 20),
      ndcg_at_10: ndcgAt(ranked, gainOf, 10),
      recall_at_100: recallAt(ranked, relevant, 100, entry.relevantCount),
      precision_at_10: precisionAt(ranked, relevant, 10),
      ranked,
      relevantFn: relevant,
    });
  }
  const relevance = {
    queries: perQuery.length,
    ndcg_at_20: round(mean(perQuery.map((entry) => entry.ndcg_at_20))),
    ndcg_at_10: round(mean(perQuery.map((entry) => entry.ndcg_at_10))),
    mrr: round(mean(perQuery.map((entry) => {
      const position = entry.ranked.findIndex(entry.relevantFn);
      return position >= 0 ? 1 / (position + 1) : 0;
    }))),
    recall_at_100: round(mean(perQuery.map((entry) => entry.recall_at_100))),
    precision_at_10: round(mean(perQuery.map((entry) => entry.precision_at_10))),
    worst_queries: [...perQuery]
      .sort((left, right) => left.ndcg_at_20 - right.ndcg_at_20)
      .slice(0, 5)
      .map((entry) => ({ query: entry.query, ndcg_at_20: round(entry.ndcg_at_20) })),
  };

  const qualitative = await Promise.all(QUALITATIVE_QUERIES.map(async (query) => {
    const needle = query.toLowerCase();
    // How many papers even contain the phrase. Without this a query the corpus
    // cannot answer looks identical to a ranking failure.
    const corpusMatches = relevanceCorpus.filter((work) =>
      `${work.title || ""} ${work.abstract || ""}`.toLowerCase().includes(needle)).length;
    return {
      query,
      corpusMatches,
      answerable: corpusMatches >= 5,
      top: (await rankForQueryAsync(adapter, relevanceCorpus, query, corpus.useModel)).slice(0, 10).map((work) => work.title.slice(0, 110)),
    };
  }));

  return { name: adapter.name, scoringMs, novelty, authorship, combined, relevance, qualitative };
}

// Model-backed ranking when the weights are present, lexical otherwise. Which
// path ran is recorded in the report, so a run without weights reads as a
// different configuration rather than as a worse algorithm.
// A file: URL pathname keeps a leading slash before a Windows drive letter,
// which every fs call then rejects.
const MODEL_ROOT = new URL("../vendor/", import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, "$1");

export const EMBEDDING_OPTIONS = {
  importTransformers: () => import("@xenova/transformers"),
  modelPath: MODEL_ROOT,
  batchSize: 32,
};

async function rankForQueryAsync(adapter, works, query, useModel) {
  if (useModel && typeof adapter.rankByRelevanceWithModel === "function") {
    return adapter.rankByRelevanceWithModel(works, query, { embedding: EMBEDDING_OPTIONS });
  }
  return rankForQuery(adapter, works, query);
}

function rankForQuery(adapter, works, query) {
  if (typeof adapter.rankByRelevance === "function") {
    const ranked = adapter.rankByRelevance(works, query);
    if (Array.isArray(ranked)) return ranked;
  }
  return works
    .map((work) => ({ work, score: Number(adapter.relevanceScore(work, query)) || 0 }))
    .sort((left, right) => right.score - left.score)
    .map((entry) => entry.work);
}

async function main() {
  const corpus = loadCorpus();
  const modelPresent = existsSync(MODEL_ROOT + "Xenova/all-MiniLM-L6-v2/onnx/model_quantized.onnx");
  corpus.useModel = modelPresent && process.env.FR_EVAL_NO_MODEL !== "1";
  const relevanceCorpus = loadCorpus({ forRelevance: true });
  corpus.relevanceCandidates = relevanceCorpus.candidates;
  corpus.rawById = new Map(corpus.rawCandidates.map((work) => [work.id, work]));

  const labels = buildLabels(corpus.rawCandidates);
  // Queries are chosen from the whole corpus so the set does not shrink with the
  // split, then kept only where the evaluated split still holds enough judged
  // documents for the metric to mean anything.
  const inSplitIds = new Set(corpus.rawCandidates.filter((work) => work.split === SPLIT).map((work) => work.id));
  const queries = buildRelevanceQueries(corpus.rawCandidates, { minimumRelevant: 10, maximumQueries: 60 })
    .map((entry) => {
      const gains = new Map([...entry.gains].filter(([id]) => inSplitIds.has(id)));
      return { ...entry, gains, relevantCount: [...gains.values()].filter((gain) => gain >= 2).length };
    })
    .filter((entry) => entry.relevantCount >= 6);

  const current = await evaluateRanker(await currentAdapter(), corpus, labels, queries);

  // The reference point that matters most: a directly-prompted LLM reading the
  // same titles and abstracts, judged against the same labels on the same
  // papers. Cached from eval/llm-baseline.mjs; absent means the comparison is
  // simply not reported rather than silently scored as zero.
  const llmPath = join(FIXTURES, "llm-novelty.json");
  let llm = null;
  if (existsSync(llmPath)) {
    const payload = JSON.parse(readFileSync(llmPath, "utf8"));
    const judgements = payload.scores || payload;
    const provenance = payload.provenance || null;
    const judged = corpus.rawCandidates.filter(
      (work) => work.split === SPLIT && judgements[work.id] !== undefined && labels.get(work.id)?.noveltyLabel !== null,
    );
    if (judged.length >= 50) {
      const gain = (work) => gainFromLabel(labels.get(work.id).noveltyLabel);
      const ranked = [...judged].sort((left, right) => judgements[right.id] - judgements[left.id]);
      const classes = disruptionClasses(judged, labels);
      llm = {
        papers: judged.length,
        ndcg_at_50: round(ndcgAt(ranked, gain, 50)),
        spearman: round(spearman(judged.map((work) => judgements[work.id]), judged.map((work) => labels.get(work.id).noveltyLabel))),
        auc_disruptive_vs_derivative: round(auc(
          classes.positives.map((work) => judgements[work.id]),
          classes.negatives.map((work) => judgements[work.id]),
        )),
        positives: classes.positives.length,
        negatives: classes.negatives.length,
        note: "A general-purpose LLM prompted directly with title and abstract, scored on the same held-out papers. Run once offline; never called at runtime.",
        // A baseline that anti-correlates with the label is a broken instrument,
        // not a bar that was cleared. Saying so here stops "beats the LLM" from
        // ever being read as evidence of quality.
        provenance,
        usableAsBaseline: null,
      };
      llm.usableAsBaseline = llm.spearman > 0.02;
      llm.caveat = llm.usableAsBaseline
        ? "Comparable."
        : "This baseline anti-correlates with the label, so it measures nothing. Beating it is not evidence of quality; it is evidence that prompting a model for novelty from an abstract does not work. Do not use it as a criterion anchor.";
      // The algorithm is scored on exactly the papers the LLM saw, so the two
      // numbers are comparable rather than measured over different sets.
      const scoredNow = new Map((await currentAdapter()).scoreBatch(corpus.candidates, corpus.peers, corpus.authors).map((w) => [w.id, w]));
      const ours = [...judged].sort((left, right) => (scoredNow.get(right.id)?.noveltyScore || 0) - (scoredNow.get(left.id)?.noveltyScore || 0));
      llm.algorithm_on_same_papers = {
        ndcg_at_50: round(ndcgAt(ours, gain, 50)),
        auc_disruptive_vs_derivative: round(auc(
          classes.positives.map((work) => scoredNow.get(work.id)?.noveltyScore || 0),
          classes.negatives.map((work) => scoredNow.get(work.id)?.noveltyScore || 0),
        )),
        spearman: round(spearman(judged.map((work) => scoredNow.get(work.id)?.noveltyScore || 0), judged.map((work) => labels.get(work.id).noveltyLabel))),
      };
    }
  }
  const baseline = await evaluateRanker(await baselineAdapter(), corpus, labels, queries);

  const report = {
    generatedAt: new Date().toISOString(),
    split: SPLIT,
    corpus: {
      candidates: corpus.candidates.length,
      inSplit: corpus.rawCandidates.filter((work) => work.split === SPLIT).length,
      peers: corpus.peers.length,
      authors: corpus.authors.length,
      relevanceQueries: queries.length,
    },
    labelNotes: LABEL_NOTES,
    novelty: current.novelty,
    authorship: current.authorship,
    combined: current.combined,
    relevanceRanker: corpus.useModel ? "lexical + bundled sentence model" : "lexical only (model weights absent)",
    relevance: current.relevance,
    qualitative: current.qualitative,
    scoringMs: current.scoringMs,
    baseline: {
      novelty: baseline.novelty,
      authorship: baseline.authorship,
      combined: baseline.combined,
      relevance: baseline.relevance,
      qualitative: baseline.qualitative,
      scoringMs: baseline.scoringMs,
    },
    llmBaseline: llm,
    deltas: {
      novelty_ndcg_at_50: round((current.novelty.ndcg_at_50 || 0) - (baseline.novelty.ndcg_at_50 || 0)),
      novelty_auc: round((current.novelty.auc_disruptive_vs_derivative || 0) - (baseline.novelty.auc_disruptive_vs_derivative || 0)),
      relevance_ndcg_at_20: round((current.relevance.ndcg_at_20 || 0) - (baseline.relevance.ndcg_at_20 || 0)),
      combined_ndcg_at_50: round((current.combined.ndcg_at_50 || 0) - (baseline.combined.ndcg_at_50 || 0)),
    },
  };

  writeFileSync(REPORT_PATH, `${JSON.stringify(report, null, 2)}\n`);

  const line = (label, now, was) => `  ${label.padEnd(30)} ${String(now).padStart(8)}   (baseline ${was})`;
  console.log(`\nFiltered Research benchmark  —  split: ${SPLIT}\n`);
  console.log(`  corpus: ${report.corpus.inSplit} candidates in split, ${report.corpus.peers} peers, ${report.corpus.relevanceQueries} relevance queries\n`);
  console.log("NOVELTY");
  console.log(line("nDCG@50", report.novelty.ndcg_at_50, report.baseline.novelty.ndcg_at_50));
  console.log(line("AUC (all negatives)", report.novelty.auc_disruptive_vs_derivative, report.baseline.novelty.auc_disruptive_vs_derivative));
  console.log(line("AUC vs derivative articles", report.novelty.auc_vs_derivative_articles, report.baseline.novelty.auc_vs_derivative_articles));
  console.log(line("AUC vs reviews only", report.novelty.auc_vs_reviews_only, report.baseline.novelty.auc_vs_reviews_only));
  console.log(line("Spearman vs label", report.novelty.spearman, report.baseline.novelty.spearman));
  console.log(line("review gap (points)", report.novelty.review_gap_points, report.baseline.novelty.review_gap_points));
  console.log("\nRELEVANCE");
  console.log(line("nDCG@20", report.relevance.ndcg_at_20, report.baseline.relevance.ndcg_at_20));
  console.log(line("MRR", report.relevance.mrr, report.baseline.relevance.mrr));
  console.log(line("recall@100", report.relevance.recall_at_100, report.baseline.relevance.recall_at_100));
  console.log("\nAUTHORSHIP / COMBINED");
  console.log(line("authorship Spearman", report.authorship.spearman, report.baseline.authorship.spearman));
  console.log(line("field bias gap", report.authorship.field_bias_gap, report.baseline.authorship.field_bias_gap));
  console.log(line("combined nDCG@50", report.combined.ndcg_at_50, report.baseline.combined.ndcg_at_50));
  console.log(`\n  wrote ${REPORT_PATH}\n`);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
