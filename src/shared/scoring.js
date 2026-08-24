// Scoring: novelty, authorship, and the order the feed is presented in.
//
// The previous model asked one question -- how far is this paper's wording from
// other papers' wording -- and answered it with TF-IDF cosine against the first
// 320 older records in insertion order. Measured against papers whose
// disruptiveness is now known, that score's rank correlation with reality was
// -0.05: no relationship at all.
//
// This version asks five questions (see novelty.js), ranks each inside the
// paper's own field, and fuses only the ones it actually has evidence for. The
// weights are fixed constants fitted on a tuning split that is disjoint from
// every number ever reported; see eval/tune-weights.mjs and eval/README.md.

import { INCREMENTAL_MARKERS } from "./defaults.js";
import { contentTerms, textOf, tokenize, buildLexicon, bm25Weights, cosineOfWeighted } from "./text.js";
import {
  buildNoveltyModel,
  buildCohortCache,
  cohortFor,
  groupPeers,
  measureCandidate,
  rankIn,
  NOVELTY_MODEL_VERSION,
} from "./novelty.js";

export const SCORING_VERSION = `${NOVELTY_MODEL_VERSION}+authorship-fieldnorm-v1`;

// Fitted on the tuning split (see eval/tune-weights.mjs). Each entry is a
// direction as well as a magnitude: `invert` means less of this signal is more
// novel.
// Emitted verbatim by `node eval/tune-weights.mjs`, which fits by coordinate
// ascent on the tuning split and then shrinks toward a uniform prior over the
// signals that correlate positively on their own. The shrink is code, not a
// remark in a comment, so re-running the tuner reproduces these numbers exactly.
//
// Text crowding is absent on purpose: measured against real outcomes its rank
// correlation with disruption is negative, so the shrink rule excludes it. It
// still does the job no reference signal can -- catching a near-verbatim
// restatement -- through the ceiling below rather than as a term in the sum.
//
// `invert` means less of this signal is more novel.
export const NOVELTY_WEIGHTS = Object.freeze([
  { key: "referenceCount", weight: 0.232, invert: true, label: "not a broad survey" },
  { key: "emergentDensity", weight: 0.196, invert: false, label: "introduces new terminology" },
  { key: "canonShare", weight: 0.142, invert: true, label: "not built on the standard canon" },
  { key: "pairSupport", weight: 0.108, invert: true, label: "sources rarely cited together" },
  { key: "unfamiliarReferenceFraction", weight: 0.088, invert: false, label: "draws on outside literature" },
  { key: "unseenPairFraction", weight: 0.075, invert: false, label: "joins sources nobody has joined" },
  { key: "strongestCoupling", weight: 0.062, invert: true, label: "no shared bibliography with existing work" },
  { key: "coupledPeerFraction", weight: 0.053, invert: true, label: "few papers share its sources" },
  { key: "genericness", weight: 0.036, invert: true, label: "not a generic restatement of the field" },
  { key: "topicConcentration", weight: 0.009, invert: false, label: "focused on one topic" },
]);

// A near-verbatim restatement of existing work is the least novel thing that can
// enter a corpus, whatever its bibliography looks like. The reference signals
// above cannot see that -- a copied paper can carry an unusual reference list --
// so text similarity acts as a ceiling on the score rather than as another term
// added into it. Kept out of the additive fusion deliberately: measured against
// real outcomes, text distance carries no disruption signal at all, so paying for
// it across the whole distribution would cost accuracy everywhere to catch a case
// that is rare but unacceptable.
const DUPLICATE_FLOOR = 0.75;
const DUPLICATE_CEILING = 0.95;

function duplicateCeiling(similarity) {
  if (!(similarity > DUPLICATE_FLOOR)) return 100;
  if (similarity >= DUPLICATE_CEILING) return 4;
  return 100 - ((similarity - DUPLICATE_FLOOR) / (DUPLICATE_CEILING - DUPLICATE_FLOOR)) * 96;
}

// Carries more weight than it otherwise would because roughly a third of
// OpenAlex records arrive with no reference list at all, and for those the
// structural detector below has nothing to read. A title that says "A
// Comprehensive Survey" is the only evidence available on such a record, and
// ignoring it would leave surveys unranked exactly where the algorithm is
// weakest.
const LEXICAL_CONSOLIDATION_WEIGHT = 6;
const STRUCTURAL_CONSOLIDATION_WEIGHT = 34;

// Consolidation detected from shape alone -- no keyword list, no document-type
// field. Logistic coefficients fitted on the tuning split against OpenAlex's own
// `type: review` flag (eval/experiments/consolidation-fit.mjs), operating on
// features rank-normalised inside the batch. Held-out separation is AUC 0.727,
// against 0.738 on the data it was fitted to, so it is genuinely reading the
// structure of consolidation work rather than memorising this corpus.
//
// The document-type field is deliberately NOT used as an input. It is what the
// benchmark defines the negative class with, so feeding it back in would make
// the measurement congratulate itself.
const CONSOLIDATION_SHAPE = Object.freeze({
  intercept: -2.869,
  terms: [
    ["referenceCount", 1.206],
    ["topicConcentration", 1.668],
    ["centrality", -1.177],
    ["canonShare", 1.001],
    ["coupledPeerFraction", -0.707],
    ["teamSize", -0.678],
    ["crowding", 0.506],
    ["unfamiliarReferenceFraction", -0.486],
    ["genericness", 0.177],
    ["emergentDensity", 0.112],
  ],
});

export function clamp(value, minimum = 0, maximum = 100) {
  // Scores pass through here, and a NaN would silently corrupt sorting and
  // display rather than failing loudly, so it resolves to the floor instead.
  const low = Number.isFinite(Number(minimum)) ? Number(minimum) : 0;
  const high = Number.isFinite(Number(maximum)) ? Number(maximum) : 100;
  const number = Number(value);
  if (!Number.isFinite(number)) return low;
  return Math.min(high, Math.max(low, number));
}

// ------------------------------------------------------------- lexical prior

// Work that announces itself as consolidating existing results. This is a weak
// prior only: the structural signals (reference count, centrality, terminology)
// are what actually separate reviews, and eval:adversarial verifies that
// switching this list off barely moves the separation.
const SURVEY_PATTERNS = Object.freeze([
  /\ba (?:systematic |comprehensive |brief |short )?(?:survey|review|overview)\b/,
  /\bsurvey of\b/, /\bsystematic review\b/, /\bliterature review\b/,
  /\ba comparative (?:study|analysis|evaluation)\b/, /\ban empirical (?:study|analysis|evaluation)\b/,
  /\bevaluation of\b/, /\bcase study\b/, /\breplication study\b/, /\bposition paper\b/, /\btutorial\b/,
]);

function consolidationShapeOf(measure, shapeDistributions) {
  let total = CONSOLIDATION_SHAPE.intercept;
  let seen = 0;
  for (const [key, coefficient] of CONSOLIDATION_SHAPE.terms) {
    const value = signalValue(measure, key);
    const distribution = shapeDistributions.get(key);
    if (!Number.isFinite(value) || !distribution) continue;
    seen += 1;
    total += coefficient * rankIn(distribution, value);
  }
  // Below about half the features there is not enough shape to judge, and a
  // partial sum would read as confident when it is not.
  if (seen < 5) return null;
  return 1 / (1 + Math.exp(-total));
}

function consolidationPrior(work) {
  const title = String(work.title || "").toLowerCase();
  const head = `${title} ${String(work.abstract || "").slice(0, 400).toLowerCase()}`;
  let hits = 0;
  for (const pattern of SURVEY_PATTERNS) {
    if (pattern.test(title)) hits += 1;
    else if (pattern.test(head)) hits += 0.5;
  }
  const markers = INCREMENTAL_MARKERS.filter((marker) => head.includes(marker));
  return { hits, markers, strength: Math.min(1, hits / 1.5 + markers.length * 0.25) };
}

// ---------------------------------------------------------------- authorship

function authorCareerScore(author) {
  const hIndex = Math.min(1, Math.log1p(author.hIndex || 0) / Math.log1p(80));
  const citations = Math.min(1, Math.log1p(author.citedByCount || 0) / Math.log1p(100_000));
  const works = Math.min(1, Math.log1p(author.worksCount || 0) / Math.log1p(300));
  const recent = Math.min(1, Math.log1p(Math.max(0, author.twoYearMeanCitedness || 0)) / Math.log1p(30));
  const identity = author.orcid ? 1 : 0.35;
  return 100 * (0.45 * hIndex + 0.25 * citations + 0.15 * recent + 0.1 * works + 0.05 * identity);
}

// An h-index of 40 is unremarkable in medicine and exceptional in mathematics.
// Comparing a paper's authors with the authors publishing in the same field is
// the only way one slider position can mean the same thing across a taxonomy.
function buildFieldAuthorNorms(candidates, authorMap) {
  const byField = new Map();
  for (const work of candidates) {
    const field = work.fieldId || work.domainId || "all";
    if (!byField.has(field)) byField.set(field, []);
    const bucket = byField.get(field);
    for (const authorship of work.authorships || []) {
      const author = authorMap.get(authorship.authorId);
      if (author) bucket.push(authorCareerScore(author));
    }
  }
  const norms = new Map();
  const pooled = [];
  for (const [field, values] of byField) {
    pooled.push(...values);
    if (values.length >= 30) norms.set(field, values.sort((left, right) => left - right));
  }
  norms.set("__pooled__", pooled.sort((left, right) => left - right));
  return norms;
}

export function scoreResearcherAuthorship(work, authorMap, fieldNorms = null) {
  if (!work || typeof work !== "object") return { score: 0, confidence: 0, evidence: [] };
  const evidence = [];
  for (const authorship of work.authorships || []) {
    const author = authorMap.get(authorship.authorId);
    if (!author) continue;
    const careerScore = authorCareerScore(author);
    const roleWeight = authorship.isCorresponding || ["first", "last"].includes(authorship.position) ? 1 : 0.86;
    evidence.push({
      id: author.id,
      name: author.name,
      score: clamp(careerScore * roleWeight),
      careerScore: clamp(careerScore),
      role: authorship.isCorresponding ? "corresponding" : authorship.position,
      hIndex: author.hIndex,
      citedByCount: author.citedByCount,
      worksCount: author.worksCount,
      twoYearMeanCitedness: author.twoYearMeanCitedness,
      orcid: author.orcid,
      institution: author.lastInstitution,
    });
  }
  if (!evidence.length) return { score: 0, confidence: 0, evidence: [] };
  evidence.sort((left, right) => right.score - left.score);
  const totalAuthors = Math.max(1, work.authorships?.length || 0);
  const confidence = clamp(evidence.length / totalAuthors, 0, 1);

  const best = evidence[0].score;
  const middle = evidence[Math.floor(evidence.length / 2)].score;
  const raw = 0.82 * best + 0.18 * middle;

  if (!fieldNorms) return { score: clamp(raw), confidence, evidence: evidence.slice(0, 5), fieldNormalised: false };
  const field = work.fieldId || work.domainId || "all";
  const distribution = fieldNorms.get(field) || fieldNorms.get("__pooled__") || [];
  const percentile = distribution.length >= 30 ? rankIn(distribution, raw) : null;
  // Blended rather than replaced: a pure percentile would erase the difference
  // between a field where everyone is prolific and one where nobody is.
  const score = percentile === null ? raw : 100 * (0.75 * percentile + 0.25 * (raw / 100));
  return {
    score: clamp(score),
    confidence,
    evidence: evidence.slice(0, 5),
    fieldNormalised: percentile !== null,
    fieldPercentile: percentile,
    rawCareerScore: clamp(raw),
  };
}

// -------------------------------------------------------------------- fusion

// How early-career the least established author on the paper is.
//
// Measured and NOT fused, which is worth recording because the signal looks so
// promising alone: on the tuning split it separates disruptive from derivative
// research at AUC 0.64 -- better than most of the reference-graph signals -- and
// it points the way the team-composition literature says it should. Added to the
// fusion at any weight the search would take, it made the whole model worse
// (tuned objective 0.586 -> 0.545, nDCG@50 0.367 -> 0.346), because what it knows
// is already carried by signals that know it more precisely. It is kept as
// reported evidence only.
function teamJuniority(work, authorMap) {
  const known = (work.authorships || [])
    .map((authorship) => authorMap.get(authorship.authorId))
    .filter(Boolean)
    .map((author) => Number(author.hIndex) || 0);
  if (!known.length) return null;
  return 1 - Math.min(1, Math.log1p(Math.min(...known)) / Math.log1p(60));
}

function signalValue(measure, key) {
  if (key === "teamJuniority") return measure.teamJuniority;
  if (key === "crowding") return measure.crowding;
  if (key === "centrality") return measure.centrality;
  if (key === "emergentDensity") return measure.emergentDensity;
  if (key === "teamSize") return measure.teamSize > 0 ? measure.teamSize : null;
  if (key === "genericness") return measure.genericness;
  if (key === "topicConcentration") return measure.topicConcentration;
  if (key === "strongestCoupling" || key === "coupledPeerFraction") return measure.coupling?.[key] ?? null;
  if (!measure.combination) return null;
  if (key === "referenceCount") return measure.combination.referenceCount;
  return measure.combination[key] ?? null;
}

function monthOf(date) {
  const text = String(date || "");
  return /^\d{4}-\d{2}/.test(text) ? text.slice(0, 7) : "unknown";
}

// Combine the signals that have evidence, ignoring the ones that do not.
// Imputing a neutral value for a missing signal would drag every record without
// references toward the middle and hide the fact that less was known about it;
// renormalising over what is present keeps the scale honest instead.
function fuse(measure, perCohort) {
  const contributions = [];
  let total = 0;
  let available = 0;
  for (const signal of NOVELTY_WEIGHTS) {
    const value = signalValue(measure, signal.key);
    const distribution = perCohort?.get(signal.key);
    if (!Number.isFinite(value) || !distribution) continue;
    const rank = rankIn(distribution, value);
    const oriented = signal.invert ? 1 - rank : rank;
    total += signal.weight * oriented;
    available += signal.weight;
    contributions.push({ signal: signal.key, label: signal.label, rank: Number(oriented.toFixed(3)), weight: signal.weight });
  }
  return { fusedValue: available > 0 ? total / available : null, contributions, availableWeight: available };
}

// When a cohort is too small to describe a distribution -- a handful of papers,
// or a field with almost no history -- there is nothing to rank against, so the
// score falls back to a fixed curve over how close the nearest existing work is.
// This is deliberately coarse: it is the honest resolution available when the
// field cannot be characterised, and the confidence term reflects that.
function absoluteNovelty(crowding) {
  const points = [[0.05, 0.94], [0.15, 0.84], [0.3, 0.68], [0.45, 0.52], [0.6, 0.34], [0.75, 0.18], [1, 0.05]];
  for (let index = 0; index < points.length; index += 1) {
    const [threshold, value] = points[index];
    if (crowding <= threshold) {
      const [previousThreshold, previousValue] = index ? points[index - 1] : [0, 1];
      const span = threshold - previousThreshold || 1;
      return previousValue + ((crowding - previousThreshold) / span) * (value - previousValue);
    }
  }
  return 0.05;
}

// --------------------------------------------------------------- entry point

export function scoreBatch(candidates, references, authors, options = {}) {
  const settings = {
    disableLexicalConsolidationPrior: false,
    peerAnchorSample: 120,
    structuralConsolidationWeight: STRUCTURAL_CONSOLIDATION_WEIGHT,
    ...options,
  };
  const cleanCandidates = (Array.isArray(candidates) ? candidates : []).filter((work) => work && typeof work === "object");
  if (!cleanCandidates.length) return [];
  const cleanPeers = (Array.isArray(references) ? references : []).filter((work) => work && typeof work === "object");
  const authorList = authors instanceof Map ? [...authors.values()] : Array.isArray(authors) ? authors : [];
  const authorMap = authors instanceof Map ? authors : new Map(authorList.map((author) => [author.id, author]));

  const model = buildNoveltyModel(cleanCandidates, cleanPeers, settings);
  const groups = groupPeers(cleanPeers);

  // A cache is built per (peer group, candidate month) rather than per candidate:
  // the expensive parts -- the posting list and the field centroid -- depend only
  // on which peers are older than the candidate, and that changes by month, not
  // by paper.
  // Candidates are grouped by the cache they need and processed a group at a
  // time, so exactly one cohort cache is alive at once. Retaining every cache for
  // the whole run -- posting lists, peer vectors and a reference index per
  // (cohort, month) -- was the single largest allocation in the pass.
  const plan = new Map();
  for (const work of cleanCandidates) {
    const cohort = cohortFor(work, model, groups);
    const key = `${cohort.key}|${monthOf(work.publicationDate)}`;
    if (!plan.has(key)) plan.set(key, { cohort, work, members: [] });
    plan.get(key).members.push(work);
  }

  const measured = [];
  const anchorsByCacheKey = new Map();
  for (const [key, group] of plan) {
    const sample = group.members[0];
    const boundary = /^\d{4}-\d{2}/.test(String(sample.publicationDate || "")) ? `${monthOf(sample.publicationDate)}-01` : null;
    const cache = buildCohortCache(group.cohort.peers, model, boundary);
    for (const work of group.members) {
      const measure = measureCandidate(work, model, group.cohort, cache);
      measure.teamJuniority = teamJuniority(work, authorMap);
      measured.push({ work, cohort: group.cohort, cacheKey: key, measure });
    }
    // The field's own papers, measured exactly as the candidates were, so the
    // distributions below describe the field rather than only this batch.
    const anchors = [];
    const stride = Math.max(1, Math.floor(cache.olderPeers.length / settings.peerAnchorSample));
    for (let index = 0; index < cache.olderPeers.length && anchors.length < settings.peerAnchorSample; index += stride) {
      const anchor = measureCandidate(cache.olderPeers[index], model, group.cohort, cache);
      anchor.teamJuniority = teamJuniority(cache.olderPeers[index], authorMap);
      anchors.push(anchor);
    }
    anchorsByCacheKey.set(key, anchors);
  }

  // Each signal becomes a rank inside its own cohort, so a raw value's units
  // never matter and no signal can dominate by having a wider range.
  //
  // The reference distribution is the field itself, not just this batch. Ranking
  // a batch only against itself would mean a month in which nothing interesting
  // was published still produces a paper scoring 99, and a batch of three papers
  // could not be scored at all. So a sample of peers is measured the same way
  // the candidates are, and seeds the distribution.
  const cohortKeys = [...new Set(measured.map((item) => item.cohort.key))];
  const distributions = new Map();
  const anchorMeasures = new Map();
  const shapeDistributions = new Map();
  for (const key of cohortKeys) {
    const mine = measured.filter((item) => item.cohort.key === key);
    // Anchors come from one cache, not from every cache in the cohort. Each cache
    // has its own cut-off date, so a peer measured against January's history and
    // the same peer measured against March's are not the same measurement;
    // pooling them mixed measurement conditions into the reference distribution
    // and cost 0.04 nDCG and 0.08 Spearman when it was tried. The largest group
    // is chosen so the reference set is the most representative one available,
    // and ties break on the key so the choice never depends on iteration order.
    const anchorKey = [...new Set(mine.map((item) => item.cacheKey))].sort((left, right) => {
      const bySize = mine.filter((item) => item.cacheKey === right).length - mine.filter((item) => item.cacheKey === left).length;
      return bySize || left.localeCompare(right);
    })[0];
    const anchors = anchorsByCacheKey.get(anchorKey) || [];

    const perSignal = new Map();
    for (const { key: signal } of NOVELTY_WEIGHTS) {
      const fromCandidates = mine.map((item) => signalValue(item.measure, signal)).filter(Number.isFinite);
      // Terminology that is new to the field is zero for every historical paper
      // by construction, so seeding that one signal with peers would bury the
      // whole candidate range under a tie at zero.
      const fromPeers = signal === "emergentDensity" ? [] : anchors.map((measure) => signalValue(measure, signal)).filter(Number.isFinite);
      const values = [...fromCandidates, ...fromPeers];
      if (values.length >= 8) perSignal.set(signal, values.sort((left, right) => left - right));
    }
    distributions.set(key, perSignal);
    anchorMeasures.set(key, anchors);

    // The consolidation features are rank-normalised the same way, against the
    // same pooled candidate-and-peer population, so the fitted coefficients see
    // the input distribution they were fitted on.
    const perShape = new Map();
    for (const [feature] of CONSOLIDATION_SHAPE.terms) {
      const values = [...mine.map((item) => signalValue(item.measure, feature)), ...anchors.map((measure) => signalValue(measure, feature))]
        .filter(Number.isFinite);
      if (values.length >= 8) perShape.set(feature, values.sort((left, right) => left - right));
    }
    shapeDistributions.set(key, perShape);
  }

  const fused = measured.map((item) => ({ ...item, ...fuse(item.measure, distributions.get(item.cohort.key)) }));

  // The fused scale is anchored the same way: a candidate's percentile is taken
  // against the field's own fused distribution, so a quiet month cannot promote
  // its least ordinary paper to the top of the range.
  const fusedByCohort = new Map();
  for (const key of cohortKeys) {
    const fromCandidates = fused
      .filter((item) => item.cohort.key === key && item.fusedValue !== null)
      .map((item) => item.fusedValue);
    const fromPeers = (anchorMeasures.get(key) || [])
      .map((measure) => fuse(measure, distributions.get(key)).fusedValue)
      .filter((value) => value !== null);
    fusedByCohort.set(key, [...fromCandidates, ...fromPeers].sort((left, right) => left - right));
  }

  // Distribution of the consolidation shape itself, pooled over candidates and
  // the peer anchors, so the penalty is positional rather than absolute.
  const shapeRanks = new Map();
  for (const key of cohortKeys) {
    const shapes = [
      ...fused.filter((item) => item.cohort.key === key).map((item) => consolidationShapeOf(item.measure, shapeDistributions.get(key) || new Map())),
      ...(anchorMeasures.get(key) || []).map((measure) => consolidationShapeOf(measure, shapeDistributions.get(key) || new Map())),
    ].filter((value) => value !== null);
    shapeRanks.set(key, shapes.sort((left, right) => left - right));
  }

  const fieldNorms = buildFieldAuthorNorms(cleanCandidates, authorMap);

  const priced = fused.map((item) => {
    const { work, measure, contributions, fusedValue, availableWeight } = item;
    const distribution = fusedByCohort.get(item.cohort.key) || [];

    // Calibration. A rank cannot bunch or saturate, which is what the previous
    // scale did; the small absolute term stops a corpus of uniformly derivative
    // work from still producing a paper that scores 99.
    const percentile = distribution.length >= 8 && fusedValue !== null ? rankIn(distribution, fusedValue) : null;
    const calibrated = percentile !== null
      ? 0.8 * percentile + 0.2 * fusedValue
      : fusedValue !== null
        ? fusedValue
        : absoluteNovelty(measure.crowding);

    const prior = consolidationPrior(work);
    const shape = consolidationShapeOf(measure, shapeDistributions.get(item.cohort.key) || new Map());
    // The logistic output is bunched -- most papers land within a narrow band of
    // probability -- so the penalty is driven by where a paper sits among its
    // peers on that scale rather than by the probability itself. Without this the
    // whole 30-point term collapsed into a 3-point spread.
    const shapeRank = shape === null ? null : rankIn(shapeRanks.get(item.cohort.key) || [], shape);
    const structuralPenalty = shapeRank === null ? 0 : settings.structuralConsolidationWeight * shapeRank ** 1.3;
    const lexicalPenalty = settings.disableLexicalConsolidationPrior ? 0 : LEXICAL_CONSOLIDATION_WEIGHT * prior.strength;
    const penalty = structuralPenalty + lexicalPenalty;

    // How much of this we can believe. A title-only record with no references and
    // twelve peers has not earned a confident score in either direction, so it is
    // pulled toward the middle rather than allowed to top the feed.
    const evidenceBreadth = Math.min(1, availableWeight / 0.75);
    const confidence = clamp(
      measure.textCompleteness * (0.55 + 0.45 * measure.peerConfidence) * (0.7 + 0.3 * evidenceBreadth),
      0,
      1,
    );
    const shrink = 0.35 + 0.65 * confidence;
    const rawScore = 50 + shrink * (100 * calibrated - 50) - penalty;
    return { item, measure, contributions, fusedValue, availableWeight, percentile, calibrated, prior, shape, shapeRank, structuralPenalty, lexicalPenalty, penalty, confidence, rawScore };
  });

  // The scale is re-anchored after the penalties and the confidence shrink, not
  // before. Applied to an already-calibrated number, those two steps pulled the
  // whole population toward the middle and left the top and bottom deciles nearly
  // empty -- the exact bunching this design exists to avoid. Ranking the finished
  // raw score inside its own cohort restores a usable spread, and keeping a slice
  // of the raw value stops a corpus of uniformly derivative work from still
  // producing a paper that scores 99.
  const rawByCohort = new Map();
  for (const key of cohortKeys) {
    rawByCohort.set(
      key,
      priced.filter((entry) => entry.item.cohort.key === key).map((entry) => entry.rawScore).sort((left, right) => left - right),
    );
  }

  return priced.map((entry) => {
    const { item, measure, contributions, fusedValue, availableWeight, percentile, calibrated, prior, shape, shapeRank, structuralPenalty, lexicalPenalty, penalty, confidence, rawScore } = entry;
    const { work } = item;
    const cohortRaw = rawByCohort.get(item.cohort.key) || [];
    const finalPercentile = cohortRaw.length >= 8 ? rankIn(cohortRaw, rawScore) : null;
    const ceiling = duplicateCeiling(Math.max(measure.nearestSemantic, measure.nearestLexical));
    const spread = finalPercentile === null
      ? rawScore
      : 100 * (0.85 * finalPercentile + 0.15 * clamp(rawScore) / 100);
    const noveltyScore = clamp(Math.min(spread, ceiling));
    const shrink = 0.35 + 0.65 * confidence;

    const researcher = scoreResearcherAuthorship(work, authorMap, fieldNorms);

    // Ordering among papers that already cleared both admission gates, so this
    // is a viewing order and not an admission rule. Novelty leads because that is
    // what the product is for; the max term keeps work that is exceptional on
    // one axis from being averaged into the middle.
    //
    // Measured against forward citations, no blend of these two beats authorship
    // on its own -- author standing predicts who gets cited, and mixing in
    // novelty only adds noise to that particular question. Ranking the feed by
    // predicted citations would make it a prestige list, which is the opposite of
    // what it is for, so the blend stays novelty-led deliberately. See
    // eval/experiments/discovery-blend.mjs for the sweep behind that choice.
    const discoveryScore = clamp(
      0.55 * noveltyScore + 0.25 * researcher.score + 0.2 * Math.max(noveltyScore, researcher.score),
    );

    const signed = contributions
      .map((entry) => ({
        ...entry,
        // What this signal actually did to the final number, in points, signed.
        points: Number((shrink * 100 * (entry.weight / Math.max(1e-9, availableWeight)) * (entry.rank - 0.5)).toFixed(2)),
      }))
      .sort((left, right) => Math.abs(right.points) - Math.abs(left.points));

    return {
      ...work,
      noveltyScore,
      noveltyConfidence: confidence,
      researcherScore: researcher.score,
      researcherConfidence: researcher.confidence,
      discoveryScore,
      nearestWorkId: measure.nearestPeer?.id || null,
      nearestTitle: measure.nearestPeer?.title || null,
      nearestSimilarity: measure.nearestSemantic,
      noveltyEvidence: {
        peerCount: measure.peerCount,
        comparedAgainst: item.cohort.key,
        crowding: measure.crowding,
        nearestSemanticSimilarity: measure.nearestSemantic,
        nearestLexicalSimilarity: measure.nearestLexical,
        centrality: measure.centrality,
        emergentTermCount: measure.emergentTermCount,
        emergentTerms: measure.emergentTerms,
        emergentDensity: measure.emergentDensity,
        referenceCount: measure.combination?.referenceCount ?? 0,
        referencePairsExamined: measure.combination?.pairs ?? 0,
        unseenPairFraction: measure.combination?.unseenPairFraction ?? null,
        pairSupport: measure.combination?.pairSupport ?? null,
        canonShare: measure.combination?.canonShare ?? null,
        unfamiliarReferenceFraction: measure.combination?.unfamiliarReferenceFraction ?? null,
        strongestCoupling: measure.coupling?.strongestCoupling ?? null,
        coupledPeerFraction: measure.coupling?.coupledPeerFraction ?? null,
        teamSize: measure.teamSize,
        teamJuniority: measure.teamJuniority,
        genericness: Number((measure.genericness ?? 0).toFixed(4)),
        topicConcentration: measure.topicConcentration,
        fusedValue,
        cohortPercentile: percentile,
        finalPercentile,
        rawScore: Number(rawScore.toFixed(2)),
        calibratedValue: calibrated,
        // Whether the field could be characterised at all. False means the score
        // came from the fixed fallback curve, not from a position in the field.
        calibrated: percentile !== null,
        confidence,
        textCompleteness: measure.textCompleteness,
        consolidationPrior: prior.strength,
        consolidationShape: shape === null ? null : Number(shape.toFixed(4)),
        consolidationShapeRank: shapeRank === null ? null : Number(shapeRank.toFixed(4)),
        consolidationPenaltyPoints: Number((-penalty).toFixed(2)),
        structuralConsolidationPoints: Number((-structuralPenalty).toFixed(2)),
        lexicalConsolidationPoints: Number((-lexicalPenalty).toFixed(2)),
        duplicateCeiling: ceiling < 100 ? Number(ceiling.toFixed(2)) : null,
        lexicalConsolidationPriorDisabled: Boolean(settings.disableLexicalConsolidationPrior),
        incrementalMarkers: prior.markers,
        signalContributions: signed,
        evidenceCoverage: Number(availableWeight.toFixed(3)),
      },
      researcherEvidence: researcher.evidence,
      researcherFieldNormalised: researcher.fieldNormalised,
      scoringVersion: SCORING_VERSION,
      scoredAt: new Date().toISOString(),
    };
  });
}

// Corpus-free lexical primitives, re-exported so callers and tests have one
// place to reach them.
export { contentTerms, textOf, tokenize, buildLexicon, bm25Weights, cosineOfWeighted };
