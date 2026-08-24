// Ground truth. Every label here is built from information that did not exist
// when the paper was published, so no ranker can reach it: forward citations,
// who cited the paper, and whether those citers also cited its references.
//
// The one exception is OpenAlex `type: review`, which is known at publication
// time -- it is used as a known-negative class, and the adversarial suite
// separately checks that the algorithm is not simply pattern-matching the word
// "review" in the title.

import { cohortPercentiles } from "./metrics.mjs";

export const LABEL_NOTES = {
  disruption:
    "Bornmann & Tekles DI_nok over a fixed 2021-2026 forward window: of the papers citing this one, the share that ignore its own references entirely. Requires >=5 citations and >=5 references.",
  citations: "Forward citations to 2026, percentile-ranked inside the paper's own (field, quarter) cohort.",
  novelty: "0.70 x disruption percentile + 0.30 x citation percentile. Disruption dominates, per the user's instruction, because raw citations measure attention rather than novelty.",
  combined: "Citation percentile alone. A combined ranker is expected to beat either of its parts at predicting which papers got read.",
  relevance:
    "Query = an OpenAlex topic display name. Gain 3 = the paper's own primary topic, 2 = a secondary topic scoring >=0.5, 1 = same subfield, 0 = otherwise. Topic fields are stripped from every record handed to a ranker, so the labels are unreachable from the input.",
};

// The quarter is taken from the paper's own publication date, not from the
// bucket it was fetched in. The oversampled review pass stamps every record it
// adds as 2021Q1 regardless of when it was published, so trusting that field
// put Q2 and Q3 reviews in the wrong citation cohort and gave them percentiles
// computed against the wrong comparison group.
function quarterOf(date) {
  const text = String(date || "");
  if (!/^\d{4}-\d{2}/.test(text)) return "unknown";
  const month = Number(text.slice(5, 7));
  return `${text.slice(0, 4)}Q${Math.floor((month - 1) / 3) + 1}`;
}

function cohortKey(work) {
  return `${work.benchField}|${quarterOf(work.publicationDate)}`;
}

export function buildLabels(candidates) {
  const withDisruption = candidates.filter((work) => work.outcome?.disruption);
  const disruptionPercentile = cohortPercentiles(
    withDisruption,
    cohortKey,
    (work) => work.outcome.disruption.di,
  );
  const citationPercentile = cohortPercentiles(
    candidates,
    cohortKey,
    (work) => Math.log1p(work.outcome?.citedByCount || 0),
  );

  const labels = new Map();
  for (const work of candidates) {
    const citations = citationPercentile.get(work) ?? 0;
    const disruption = disruptionPercentile.has(work) ? disruptionPercentile.get(work) : null;
    labels.set(work.id, {
      id: work.id,
      isReview: work.workType === "review",
      citationPercentile: citations,
      disruptionPercentile: disruption,
      disruptionRaw: work.outcome?.disruption?.di ?? null,
      // Novelty is only defined where a disruption measurement exists. Filling
      // the gap with citations alone would quietly turn this into a popularity
      // benchmark for the two thirds of records that lack references.
      noveltyLabel: disruption === null ? null : 0.7 * disruption + 0.3 * citations,
      combinedLabel: citations,
    });
  }
  return labels;
}

// nDCG needs a bounded gain. Four graded buckets keep 2^g - 1 from letting a
// single top item dominate the whole measure.
export function gainFromLabel(value) {
  if (value === null || value === undefined) return 0;
  if (value >= 0.85) return 3;
  if (value >= 0.65) return 2;
  if (value >= 0.4) return 1;
  return 0;
}

// The separation task: work that displaced its predecessors, against work that
// consolidated them.
export function disruptionClasses(candidates, labels) {
  const positives = [];
  const negatives = [];
  for (const work of candidates) {
    const label = labels.get(work.id);
    if (!label) continue;
    if (label.isReview) {
      negatives.push(work);
      continue;
    }
    if (label.disruptionPercentile === null) continue;
    if (label.disruptionPercentile >= 0.75) positives.push(work);
    else if (label.disruptionPercentile <= 0.25) negatives.push(work);
  }
  return { positives, negatives };
}

// Relevance judgements from each paper's own topic assignment. The ranker never
// sees these fields -- see stripTopicEvidence in adapters.mjs.
export function buildRelevanceQueries(candidates, { minimumRelevant = 12, maximumQueries = 45 } = {}) {
  const byTopic = new Map();
  for (const work of candidates) {
    if (!work.topicId || !work.topicName) continue;
    if (!byTopic.has(work.topicId)) byTopic.set(work.topicId, { topicId: work.topicId, topicName: work.topicName, subfieldId: work.subfieldId, primary: [] });
    byTopic.get(work.topicId).primary.push(work.id);
  }
  const queries = [...byTopic.values()]
    .filter((entry) => entry.primary.length >= minimumRelevant)
    .sort((left, right) => right.primary.length - left.primary.length || left.topicName.localeCompare(right.topicName))
    .slice(0, maximumQueries);

  return queries.map((entry) => {
    const gains = new Map();
    for (const work of candidates) {
      let gain = 0;
      if (work.topicId === entry.topicId) gain = 3;
      else if ((work.topics || []).some((topic) => topic.topicId === entry.topicId && topic.score >= 0.5)) gain = 2;
      else if (work.subfieldId && work.subfieldId === entry.subfieldId) gain = 1;
      if (gain) gains.set(work.id, gain);
    }
    return { query: entry.topicName, topicId: entry.topicId, gains, relevantCount: [...gains.values()].filter((gain) => gain >= 2).length };
  });
}

// Six queries a real user would type, kept separate from the topic-derived set
// because they test phrasing the labels cannot: abbreviations, multi-word
// concepts, and terms whose expansion differs from their acronym.
export const QUALITATIVE_QUERIES = Object.freeze([
  "RAG",
  "LLM agents",
  "diffusion",
  "protein folding",
  "causal inference",
  "sparse autoencoder",
]);
