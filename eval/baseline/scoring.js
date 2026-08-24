import { INCREMENTAL_MARKERS } from "./defaults.js";

export const SCORING_VERSION = "peer-calibrated-novelty-v3";

const STOPWORDS = new Set(
  `a an and are as at be been being by can could did do does for from had has have how if in into is it its may might more most no not of on or our should so such than that the their then there these they this those through to under using via was we were what when where which while who will with would`.split(
    " ",
  ),
);

export function clamp(value, minimum = 0, maximum = 100) {
  // Scores pass through here, and a NaN would silently corrupt sorting and
  // display rather than failing loudly, so it resolves to the floor instead.
  const low = Number.isFinite(Number(minimum)) ? Number(minimum) : 0;
  const high = Number.isFinite(Number(maximum)) ? Number(maximum) : 100;
  const number = Number(value);
  if (!Number.isFinite(number)) return low;
  return Math.min(high, Math.max(low, number));
}

export function tokenize(text, maximum = 700) {
  const tokens = String(text || "")
    .toLowerCase()
    .match(/[a-z][a-z0-9-]{2,}/g);
  if (!tokens) return [];
  return tokens.filter((token) => !STOPWORDS.has(token)).slice(0, maximum);
}

function counts(tokens) {
  const result = new Map();
  for (const token of tokens) result.set(token, (result.get(token) || 0) + 1);
  return result;
}

export function buildVector(text, inverseDocumentFrequency = new Map()) {
  const idf = inverseDocumentFrequency instanceof Map ? inverseDocumentFrequency : new Map();
  const termCounts = counts(tokenize(text));
  const vector = new Map();
  let squaredNorm = 0;
  for (const [term, frequency] of termCounts) {
    const weight = (1 + Math.log(frequency)) * (idf.get(term) || 1);
    vector.set(term, weight);
    squaredNorm += weight * weight;
  }
  return { vector, norm: Math.sqrt(squaredNorm) };
}

export function cosineSimilarity(left, right) {
  if (!left?.norm || !right?.norm || !(left.vector instanceof Map) || !(right.vector instanceof Map)) return 0;
  const [smaller, larger] =
    left.vector.size <= right.vector.size
      ? [left.vector, right.vector]
      : [right.vector, left.vector];
  let dot = 0;
  for (const [term, weight] of smaller) dot += weight * (larger.get(term) || 0);
  return clamp(dot / (left.norm * right.norm), 0, 1);
}

// Raw cosine distance is not a meaningful scale on its own: in a field with a
// wide vocabulary almost every pair of papers sits at 0.05-0.20 similarity, so
// "distance" reads 80-95% for derivative and groundbreaking work alike and the
// resulting scores bunch into a narrow band. Novelty is therefore measured
// against how crowded the field itself is, not against an absolute distance.

function median(values) {
  if (!values.length) return 0;
  const sorted = [...values].sort((left, right) => left - right);
  const middle = sorted.length >> 1;
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

// Term -> flat [peerIndex, weight, peerIndex, weight, ...]. Comparing every
// candidate against every peer meant roughly two million cosine computations on
// a full rescore, almost all of them between papers sharing no vocabulary at
// all. Indexing the peers once lets a candidate touch only the peers that
// actually share a term with it.
function buildPeerPostings(peerVectors) {
  const postings = new Map();
  for (let index = 0; index < peerVectors.length; index += 1) {
    const peer = peerVectors[index];
    if (!peer?.vector) continue;
    for (const [term, weight] of peer.vector) {
      let list = postings.get(term);
      if (!list) postings.set(term, (list = []));
      list.push(index, weight);
    }
  }
  return postings;
}

// How close a paper sits to the work that already exists: its nearest peer plus
// the density of the neighbourhood, so one coincidental match cannot alone make
// a paper look derivative.
function crowdingOf(vector, peerVectors, postings = null, excludeIndex = -1) {
  if (!peerVectors.length) return { crowding: 0, nearestIndex: -1, nearestSimilarity: 0 };
  const index = postings || buildPeerPostings(peerVectors);
  const dots = new Float64Array(peerVectors.length);
  if (vector?.vector && vector.norm) {
    for (const [term, weight] of vector.vector) {
      const list = index.get(term);
      if (!list) continue;
      for (let position = 0; position < list.length; position += 2) {
        dots[list[position]] += weight * list[position + 1];
      }
    }
  }

  const similarities = [];
  let nearestSimilarity = 0;
  let nearestIndex = -1;
  for (let peer = 0; peer < peerVectors.length; peer += 1) {
    if (peer === excludeIndex) continue;
    const peerNorm = peerVectors[peer]?.norm || 0;
    const similarity = peerNorm && vector?.norm
      ? clamp(dots[peer] / (vector.norm * peerNorm), 0, 1)
      : 0;
    similarities.push(similarity);
    if (similarity > nearestSimilarity) {
      nearestSimilarity = similarity;
      nearestIndex = peer;
    }
  }
  if (!similarities.length) return { crowding: 0, nearestIndex: -1, nearestSimilarity: 0 };
  similarities.sort((left, right) => right - left);
  const topFive = similarities.slice(0, 5);
  const neighbourhood = topFive.reduce((sum, value) => sum + value, 0) / topFive.length;
  return { crowding: 0.65 * nearestSimilarity + 0.35 * neighbourhood, nearestIndex, nearestSimilarity };
}

// The field's own crowding distribution, measured by treating sampled peers as
// candidates against the rest. Computed once per peer group and reused.
const PEER_STATS_SAMPLE = 90;
function peerFieldStats(peers, peerVectors, idf, postings = null) {
  const step = Math.max(1, Math.floor(peerVectors.length / PEER_STATS_SAMPLE));
  const peerPostings = postings || buildPeerPostings(peerVectors);
  const crowdings = [];
  const distinctions = [];
  for (let index = 0; index < peerVectors.length && crowdings.length < PEER_STATS_SAMPLE; index += step) {
    // Excluding by position avoids rebuilding a 300-element array per sample.
    crowdings.push(crowdingOf(peerVectors[index], peerVectors, peerPostings, index).crowding);
    const peer = peers[index];
    if (peer) distinctions.push(distinctivenessOf(`${peer.title || ""} ${peer.abstract || ""}`, idf));
  }
  const crowding = distributionOf(crowdings);
  if (!crowding) return null;
  // Median absolute deviation, scaled to a standard-deviation equivalent, so a
  // few unusual peers cannot flatten the scale.
  return { crowding, distinctiveness: distributionOf(distinctions) };
}

// A second, independent signal. Crowding asks "has this been written before";
// distinctiveness asks "does this introduce vocabulary the field does not
// already have". A survey or benchmark paper reuses the field's common terms,
// while work that introduces a genuinely new idea carries rare ones. Mean IDF
// over a paper's own terms captures that cheaply and is orthogonal to cosine
// distance, which is what makes incremental work separable from novel work.
function distinctivenessOf(text, idf) {
  const terms = tokenize(text, 400);
  if (!terms.length) return 0;
  const unique = [...new Set(terms)];
  let total = 0;
  for (const term of unique) total += idf.get(term) || 0;
  return total / unique.length;
}

function distributionOf(values) {
  if (values.length < 4) return null;
  return { sorted: [...values].sort((left, right) => left - right), center: median(values) };
}

// Where a value sits within the field, expressed as a rank rather than a
// z-score. Dividing by a spread is unsafe here: on a homogeneous corpus the
// median absolute deviation collapses toward zero, which sent standings to 20
// or even 170 and pinned every interesting paper to exactly 100 once passed
// through a logistic. A rank cannot collapse, so the top of the scale keeps
// resolving instead of saturating.
function standingIn(distribution, value, higherIsMoreNovel) {
  if (!distribution) return null;
  const direction = higherIsMoreNovel ? 1 : -1;
  // Order by "less novel first" for whichever direction applies, otherwise an
  // inverted signal scans an ascending array and always reports rank zero.
  const sorted = direction === 1
    ? distribution.sorted
    : [...distribution.sorted].reverse();
  const target = direction * value;
  let below = 0;
  while (below < sorted.length && direction * sorted[below] < target) below += 1;
  // The field occupies the lower 90% of the scale, leaving room above it for
  // work that is more unusual than anything already published.
  if (below < sorted.length) return (0.9 * below) / sorted.length;
  const top = direction * sorted[sorted.length - 1];
  const mid = direction * sorted[Math.floor(sorted.length / 2)];
  const spread = Math.max(1e-9, Math.abs(top - mid));
  return 0.9 + 0.1 * (1 - Math.exp(-Math.max(0, target - top) / spread));
}

// Work that announces itself as consolidating existing results. These are the
// papers that should sit at the bottom of a novelty ranking.
const SURVEY_PATTERNS = Object.freeze([
  /\ba (?:systematic |comprehensive |brief |short )?(?:survey|review|overview)\b/,
  /\bsurvey of\b/, /\bsystematic review\b/, /\bliterature review\b/,
  /\ba comparative (?:study|analysis|evaluation)\b/, /\ban empirical (?:study|analysis|evaluation)\b/,
  /\bbenchmark(?:ing|s)?\b/, /\bevaluation of\b/, /\bcase study\b/,
  /\breplication study\b/, /\bposition paper\b/, /\btutorial\b/,
]);

function consolidationPenalty(title, abstract) {
  const head = `${title} ${abstract.slice(0, 400)}`.toLowerCase();
  const titleOnly = String(title || "").toLowerCase();
  let penalty = 0;
  for (const pattern of SURVEY_PATTERNS) {
    if (pattern.test(titleOnly)) penalty += 9;
    else if (pattern.test(head)) penalty += 4;
  }
  return Math.min(26, penalty);
}

function logistic(value) {
  return 1 / (1 + Math.exp(-value));
}

// Without enough peers to describe the field, fall back to a fixed curve over
// the nearest-peer similarity rather than inventing a distribution.
function absoluteNovelty(nearestSimilarity) {
  const points = [[0.02, 96], [0.08, 88], [0.15, 74], [0.25, 58], [0.35, 42], [0.5, 24], [0.7, 8], [1, 2]];
  for (let index = 0; index < points.length; index += 1) {
    const [similarity, score] = points[index];
    if (nearestSimilarity <= similarity) {
      const [previousSimilarity, previousScore] = index ? points[index - 1] : [0, 100];
      const span = similarity - previousSimilarity || 1;
      const progress = (nearestSimilarity - previousSimilarity) / span;
      return previousScore + progress * (score - previousScore);
    }
  }
  return 2;
}

function idfForWorks(works) {
  const documentFrequency = new Map();
  for (const work of works) {
    const unique = new Set(tokenize(`${work.title || ""} ${work.abstract || ""}`));
    for (const term of unique) {
      documentFrequency.set(term, (documentFrequency.get(term) || 0) + 1);
    }
  }
  const total = Math.max(1, works.length);
  return new Map(
    [...documentFrequency].map(([term, frequency]) => [
      term,
      Math.log((total + 1) / (frequency + 1)) + 1,
    ]),
  );
}

function titlePhrases(title) {
  const tokens = tokenize(title, 40);
  const phrases = [];
  for (const size of [2, 3]) {
    for (let index = 0; index <= tokens.length - size; index += 1) {
      phrases.push(tokens.slice(index, index + size).join(" "));
    }
  }
  return phrases;
}

function phraseRarity(work, peers) {
  const candidate = titlePhrases(work.title);
  if (!candidate.length) return 0.5;
  const known = new Set(peers.flatMap((peer) => titlePhrases(peer.title)));
  const unseenFraction = candidate.filter((phrase) => !known.has(phrase)).length / candidate.length;
  return clamp((unseenFraction - 0.35) / 0.65, 0, 1);
}

function bridgeRarity(work, peers) {
  const fields = [...new Set((work.topics || []).map((topic) => topic.fieldId).filter(Boolean))];
  if (fields.length < 2) return 0.35;
  const pairs = [];
  for (let left = 0; left < fields.length; left += 1) {
    for (let right = left + 1; right < fields.length; right += 1) {
      pairs.push([fields[left], fields[right]].sort().join("|"));
    }
  }
  const peerPairs = new Set();
  for (const peer of peers) {
    const peerFields = [...new Set((peer.topics || []).map((topic) => topic.fieldId).filter(Boolean))];
    for (let left = 0; left < peerFields.length; left += 1) {
      for (let right = left + 1; right < peerFields.length; right += 1) {
        peerPairs.add([peerFields[left], peerFields[right]].sort().join("|"));
      }
    }
  }
  return pairs.length ? pairs.filter((pair) => !peerPairs.has(pair)).length / pairs.length : 0.35;
}

function buildPeerIndex(references) {
  const bySubfield = new Map();
  const byDomain = new Map();
  for (const reference of references) {
    if (reference.subfieldId) {
      const peers = bySubfield.get(reference.subfieldId) || [];
      peers.push(reference);
      bySubfield.set(reference.subfieldId, peers);
    }
    if (reference.domainId) {
      const peers = byDomain.get(reference.domainId) || [];
      peers.push(reference);
      byDomain.set(reference.domainId, peers);
    }
  }
  return { all: references, bySubfield, byDomain };
}

function choosePeers(work, peerIndex, minimumTopicPeers, maximumPeers) {
  const subfield = peerIndex.bySubfield.get(work.subfieldId) || [];
  const domain = peerIndex.byDomain.get(work.domainId) || [];
  const [source, groupKey] =
    subfield.length >= minimumTopicPeers
      ? [subfield, `subfield:${work.subfieldId}`]
      : domain.length >= minimumTopicPeers
        ? [domain, `domain:${work.domainId}`]
        : [peerIndex.all, "all"];
  const older = source.filter(
    (reference) => reference.id !== work.id && reference.publicationDate < work.publicationDate,
  );
  return { peers: older.slice(0, maximumPeers), groupKey };
}

function authorCareerScore(author) {
  const hIndex = Math.min(1, Math.log1p(author.hIndex || 0) / Math.log1p(80));
  const citations = Math.min(
    1,
    Math.log1p(author.citedByCount || 0) / Math.log1p(100_000),
  );
  const works = Math.min(1, Math.log1p(author.worksCount || 0) / Math.log1p(300));
  const recent = Math.min(
    1,
    Math.log1p(Math.max(0, author.twoYearMeanCitedness || 0)) / Math.log1p(30),
  );
  const identity = author.orcid ? 1 : 0.35;
  return 100 * (0.45 * hIndex + 0.25 * citations + 0.15 * recent + 0.1 * works + 0.05 * identity);
}

export function scoreResearcherAuthorship(work, authorMap) {
  if (!work || typeof work !== "object") return { score: 0, evidence: [] };
  const evidence = [];
  for (const authorship of work.authorships || []) {
    const author = authorMap.get(authorship.authorId);
    if (!author) continue;
    const careerScore = authorCareerScore(author);
    const roleWeight =
      authorship.isCorresponding || ["first", "last"].includes(authorship.position) ? 1 : 0.86;
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
  evidence.sort((left, right) => right.score - left.score);
  const totalAuthors = Math.max(1, work.authorships?.length || 0);
  const confidence = evidence.length / totalAuthors;
  if (!evidence.length) return { score: 0, confidence: 0, evidence: [] };
  const best = evidence[0].score;
  const median = evidence[Math.floor(evidence.length / 2)].score;
  return {
    score: clamp(0.82 * best + 0.18 * median),
    confidence: clamp(confidence, 0, 1),
    evidence: evidence.slice(0, 5),
  };
}

export function scoreBatch(candidates, references, authors, options = {}) {
  candidates = (Array.isArray(candidates) ? candidates : []).filter((w) => w && typeof w === "object");
  if (!candidates.length) return [];
  references = (Array.isArray(references) ? references : []).filter((w) => w && typeof w === "object");
  if (!Array.isArray(authors) && !(authors instanceof Map)) authors = [];
  const settings = {
    minTopicPeers: 20,
    maxPeerComparisons: 320,
    noveltySpread: 1.15,
    incrementalMarkers: INCREMENTAL_MARKERS,
    ...options,
  };
  const allForIdf = [...references, ...candidates].filter((work) => work.abstract || work.title);
  const idf = idfForWorks(allForIdf);
  const vectors = new Map(
    allForIdf.map((work) => [work.id, buildVector(`${work.title || ""} ${work.abstract || ""}`, idf)]),
  );
  const peerIndex = buildPeerIndex(references);
  const authorMap = authors instanceof Map ? authors : new Map(authors.map((author) => [author.id, author]));

  // First pass measures every candidate. The field distribution is then built
  // from the peers *and* the candidates, so papers are ranked against each other
  // as well as against past work. Ranking against peers alone left every
  // candidate beyond the top of the peer range and therefore tied.
  const statsByGroup = new Map();
  const postingsByGroup = new Map();
  const measured = candidates.map((work) => {
    const { peers, groupKey } = choosePeers(
      work,
      peerIndex,
      settings.minTopicPeers,
      settings.maxPeerComparisons,
    );
    const candidateVector = vectors.get(work.id) || buildVector(`${work.title || ""} ${work.abstract || ""}`, idf);
    const peerVectors = peers.map((peer) => vectors.get(peer.id)).filter(Boolean);
    // The posting list is per peer group, so it is built once and reused by
    // every candidate scored against that group.
    let postings = postingsByGroup.get(groupKey);
    if (!postings) {
      postings = buildPeerPostings(peerVectors);
      postingsByGroup.set(groupKey, postings);
    }
    const measure = crowdingOf(candidateVector, peerVectors, postings);
    if (!statsByGroup.has(groupKey)) statsByGroup.set(groupKey, peerFieldStats(peers, peerVectors, idf, postings));
    return {
      work,
      peers,
      groupKey,
      crowding: measure.crowding,
      nearestIndex: measure.nearestIndex,
      nearestSimilarity: measure.nearestSimilarity,
      distinctiveness: distinctivenessOf(`${work.title} ${work.abstract || ""}`, idf),
    };
  });

  for (const [groupKey, stats] of statsByGroup) {
    if (!stats) continue;
    const mine = measured.filter((item) => item.groupKey === groupKey);
    if (!mine.length) continue;
    stats.crowding = distributionOf([...stats.crowding.sorted, ...mine.map((item) => item.crowding)]);
    if (stats.distinctiveness) {
      stats.distinctiveness = distributionOf([
        ...stats.distinctiveness.sorted,
        ...mine.map((item) => item.distinctiveness),
      ]);
    }
  }

  return measured.map(({ work, peers, groupKey, crowding, nearestIndex, nearestSimilarity, distinctiveness }) => {
    const nearest = nearestIndex >= 0 ? peers[nearestIndex] : null;
    const semanticDistance = 1 - nearestSimilarity;
    const stats = statsByGroup.get(groupKey);

    const phrases = phraseRarity(work, peers);
    const bridge = bridgeRarity(work, peers);
    const haystack = `${work.title || ""} ${String(work.abstract || "").slice(0, 800)}`.toLowerCase();
    const markers = settings.incrementalMarkers.filter((marker) => haystack.includes(marker));
    const consolidation = consolidationPenalty(work.title, work.abstract || "");
    const penalty = Math.min(34, markers.length * 5 + consolidation);

    // Two independent questions, each scored against the field's own spread:
    // how crowded the neighbourhood is, and how much vocabulary the paper adds
    // that the field does not already use. Positioning against the field rather
    // than an absolute distance is what lets the score use the whole 1-100
    // range instead of bunching near the top.
    // Less crowded is more novel, so the crowding rank is inverted.
    const crowdingStanding = standingIn(stats?.crowding, crowding, false);
    const distinctStanding = standingIn(stats?.distinctiveness, distinctiveness, true);
    const relativeStanding = crowdingStanding === null
      ? null
      : distinctStanding === null
        ? crowdingStanding
        : 0.62 * crowdingStanding + 0.38 * distinctStanding;
    const base = relativeStanding === null
      ? absoluteNovelty(nearestSimilarity)
      : 100 * relativeStanding;
    const rawNovelty = clamp(base + 7 * (phrases - 0.5) + 3 * (bridge - 0.35) - penalty);

    const abstractTokens = tokenize(work.abstract || "", 200).length;
    const textCompleteness = abstractTokens >= 50 ? 1 : abstractTokens ? 0.65 : 0.35;
    const peerConfidence = Math.min(1, Math.log1p(peers.length) / Math.log1p(60));
    const noveltyConfidence = peerConfidence * textCompleteness;
    // Shrink only weakly, and only toward the middle of the calibrated scale;
    // the old full shrink to 50 is what put a floor under every score.
    const shrink = 0.25 + 0.75 * noveltyConfidence;
    const noveltyScore = clamp(50 + shrink * (rawNovelty - 50));

    const researcher = scoreResearcherAuthorship(work, authorMap);
    const discoveryScore = clamp(
      0.72 * Math.max(noveltyScore, researcher.score) +
        0.28 * Math.min(noveltyScore, researcher.score),
    );
    return {
      ...work,
      noveltyScore,
      noveltyConfidence,
      researcherScore: researcher.score,
      researcherConfidence: researcher.confidence,
      discoveryScore,
      nearestWorkId: nearest?.id || null,
      nearestTitle: nearest?.title || null,
      nearestSimilarity,
      noveltyEvidence: {
        peerCount: peers.length,
        semanticDistance,
        crowding,
        fieldCrowding: stats ? stats.crowding.center : null,
        distinctiveness,
        fieldDistinctiveness: stats?.distinctiveness ? stats.distinctiveness.center : null,
        crowdingStanding,
        distinctStanding,
        relativeStanding,
        consolidation,
        calibrated: Boolean(stats),
        phraseRarity: phrases,
        bridgeRarity: bridge,
        incrementalMarkers: markers,
        penalty,
        textCompleteness,
      },
      researcherEvidence: researcher.evidence,
      scoringVersion: SCORING_VERSION,
      scoredAt: new Date().toISOString(),
    };
  });
}
