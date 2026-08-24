// Novelty signals.
//
// The question "is this new?" is not one question, and the old implementation
// lost because it only asked one version of it -- how far the wording sits from
// other wording -- which a thesaurus can fake and a typo can win. Five
// independent views are computed here instead, each answering something a
// lexical distance cannot:
//
//   crowding      does work like this already exist, in meaning rather than in
//                 vocabulary
//   combination   does it join literatures that have not been joined before
//                 (the reference graph, not the text)
//   terminology   does it introduce words the field genuinely did not have,
//                 where "genuinely" means other people use them too
//   centrality    does it sit at the middle of its field, which is where
//                 surveys and consolidations live
//   reach         does it draw on work outside what this field normally cites
//
// Each is turned into a rank inside the paper's own field cohort before being
// combined, so no single signal's units can dominate, and so a fused score means
// the same thing in materials science as in machine learning.

import { contentTerms, lexiconFrom, bm25Weights, cosineOfWeighted, textOf, tokenize } from "./text.js";
import { buildSemanticSpace, documentVector, semanticSimilarity, tokensForSpace, removeCommonComponent } from "./semantic.js";

export const NOVELTY_MODEL_VERSION = "multi-signal-novelty-v1";

const EMERGENT_TERM_CAP = 4;

const DEFAULTS = Object.freeze({
  maxPeerComparisons: 400,
  nearestPeersByLexicon: 250,
  minimumCohortPeers: 25,
  neighbourhoodSize: 5,
  maxReferencePairs: 400,
});

// ---------------------------------------------------------------- small stats

function median(sorted) {
  if (!sorted.length) return 0;
  const middle = sorted.length >> 1;
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

function quantileOf(sorted, fraction) {
  if (!sorted.length) return 0;
  const position = (sorted.length - 1) * Math.min(1, Math.max(0, fraction));
  const low = Math.floor(position);
  const high = Math.ceil(position);
  return low === high ? sorted[low] : sorted[low] + (position - low) * (sorted[high] - sorted[low]);
}

// Fraction of a sorted distribution lying below a value, counting ties at half.
// A rank cannot blow up the way a z-score does when a corpus is homogeneous and
// its spread collapses toward zero -- which is the failure that pinned every
// interesting paper in the previous implementation to the same number.
export function rankIn(sorted, value) {
  if (!sorted.length) return 0.5;
  let low = 0;
  let high = sorted.length;
  while (low < high) {
    const middle = (low + high) >> 1;
    if (sorted[middle] < value) low = middle + 1;
    else high = middle;
  }
  const below = low;
  let equal = 0;
  while (below + equal < sorted.length && sorted[below + equal] === value) equal += 1;
  return (below + equal / 2) / sorted.length;
}

function sortedCopy(values) {
  return [...values].sort((left, right) => left - right);
}

// ------------------------------------------------------------ reference graph

// How often each prior work is cited by the field, and how often each pair of
// prior works is cited together. This is the whole basis of the combination
// signal: a pair of references that the field has never put in the same
// bibliography is a join nobody has made.
// A pair key has to be cheap: a corpus of a few thousand bibliographies produces
// millions of pairs, and holding each one as a concatenated "W123|W456" string
// cost hundreds of megabytes on its own. Two 32-bit hashes folded together give
// a numeric key instead, which a Map stores as a tagged value rather than as an
// allocation.
function hashId(value) {
  let hash = 2166136261;
  const text = String(value);
  for (let index = 0; index < text.length; index += 1) {
    hash ^= text.charCodeAt(index);
    hash = Math.imul(hash, 16777619) >>> 0;
  }
  return hash >>> 0;
}

function pairKey(left, right) {
  const a = hashId(left);
  const b = hashId(right);
  const low = a < b ? a : b;
  const high = a < b ? b : a;
  // Kept inside the safe-integer range so the value stays a number, not a double
  // with a fractional part that would break Map identity.
  return low * 4294967296 + high;
}

function buildCoCitationIndex(peers, { maxPairsPerPeer = 220 } = {}) {
  const citedCount = new Map();
  const pairCount = new Map();
  let bibliographies = 0;
  for (const peer of peers) {
    const references = peer?.referencedWorks;
    if (!Array.isArray(references) || references.length < 2) continue;
    bibliographies += 1;
    const unique = [...new Set(references)];
    for (const id of unique) citedCount.set(id, (citedCount.get(id) || 0) + 1);
    // Long bibliographies would contribute O(n^2) pairs and dominate the index,
    // so they are truncated rather than allowed to define the field's norms.
    const capped = unique.length * (unique.length - 1) / 2 > maxPairsPerPeer
      ? unique.slice(0, Math.floor((1 + Math.sqrt(1 + 8 * maxPairsPerPeer)) / 2))
      : unique;
    for (let left = 0; left < capped.length; left += 1) {
      for (let right = left + 1; right < capped.length; right += 1) {
        const key = pairKey(capped[left], capped[right]);
        pairCount.set(key, (pairCount.get(key) || 0) + 1);
      }
    }
  }
  // "Canon" means the top slice of what this field cites, defined from the data
  // rather than by a fixed number, so it adapts to a corpus of any size.
  const counts = [...citedCount.values()].sort((left, right) => right - left);
  const canonThreshold = counts.length ? Math.max(2, counts[Math.floor(counts.length * 0.01)] || 2) : Infinity;
  return { citedCount, pairCount, bibliographies, canonThreshold };
}

// Uzzi-style atypicality, computed against the field's own co-citation history.
// Two numbers matter and they are not the same number: how conventional the
// typical pairing is, and whether there is a tail of pairings nobody has made.
// Work that is only atypical is usually noise; work that is conventional with an
// atypical tail is the shape that new ideas actually have.
// `selfIndexed` means this paper's own bibliography is one of the ones the index
// was built from -- true whenever a peer is being measured to seed the reference
// distributions. Its own contribution is then subtracted, because otherwise every
// such paper is compared against a field that has already read it: its reference
// pairs all have support, none of its pairings are unseen, and it couples
// perfectly with itself. Left uncorrected, the peers seeding the distributions
// sat at the conventional extreme (strongestCoupling exactly 1.0 for 100% of
// them) and compressed the candidate range they were supposed to calibrate.
function referenceCombination(work, index, { selfIndexed = false } = {}) {
  const references = [...new Set(Array.isArray(work.referencedWorks) ? work.referencedWorks : [])];
  const total = index.bibliographies - (selfIndexed ? 1 : 0);
  if (references.length < 3 || total < 1) return null;
  const own = selfIndexed ? 1 : 0;
  const pointwise = [];
  let unseenPairs = 0;
  let pairs = 0;
  outer: for (let left = 0; left < references.length; left += 1) {
    for (let right = left + 1; right < references.length; right += 1) {
      if (pairs >= DEFAULTS.maxReferencePairs) break outer;
      pairs += 1;
      const a = references[left];
      const b = references[right];
      const together = Math.max(0, (index.pairCount.get(pairKey(a, b)) || 0) - own);
      const countA = Math.max(0, (index.citedCount.get(a) || 0) - own);
      const countB = Math.max(0, (index.citedCount.get(b) || 0) - own);
      if (together === 0) unseenPairs += 1;
      // Expected co-citations if the two were cited independently. A pair that
      // occurs far less often than chance predicts is an unusual join; a pair
      // that occurs far more often is the field's standard pairing.
      const expected = (countA * countB) / total;
      const observed = together;
      pointwise.push(Math.log((observed + 0.5) / (expected + 0.5)));
    }
  }
  if (!pairs) return null;

  const sorted = sortedCopy(pointwise);
  const known = references.filter((id) => (index.citedCount.get(id) || 0) - own > 0).length;
  // How heavily this bibliography leans on the works the field cites most. A
  // paper built entirely from the canon is extending it; one that mostly cites
  // work the field rarely touches is doing something else.
  let canonHits = 0;
  for (const id of references) if ((index.citedCount.get(id) || 0) - own >= index.canonThreshold) canonHits += 1;

  return {
    pairs,
    unseenPairFraction: unseenPairs / pairs,
    // Mean co-citation support. The earlier median-PMI and tail-PMI statistics
    // were both dead on real data: with a sampled peer corpus, almost every pair
    // has zero observed co-citations, so both collapsed to a constant and
    // contributed exactly nothing (Spearman 0.0000, AUC 0.5000). This keeps the
    // same idea on a scale that actually varies.
    pairSupport: median(sorted) === 0 ? mean(pointwise) : median(sorted),
    // References the field has never touched: work reaching outside its own
    // literature, which is where cross-disciplinary imports show up.
    unfamiliarReferenceFraction: 1 - known / references.length,
    canonShare: canonHits / references.length,
    referenceCount: references.length,
  };
}

function mean(values) {
  return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : 0;
}

// ------------------------------------------------------------------- the model

export function buildNoveltyModel(candidates, peers, options = {}) {
  const settings = { ...DEFAULTS, ...options };
  const allWorks = [...peers, ...candidates];
  const peerSet = new Set(peers);

  // Pass one counts vocabulary. Term arrays are built, counted and dropped one
  // document at a time so the whole corpus is never resident as token lists.
  const documentFrequency = new Map();
  const historicalDocumentFrequency = new Map();
  const currentDocumentFrequency = new Map();
  // Vocabulary history is pooled across the whole index rather than kept per
  // field. Per-field history is the more intuitive design and was implemented and
  // measured: it did not improve the corpus-stability it was meant to fix (6.65
  // points either way) and it cost real accuracy, dropping nDCG@50 from 0.367 to
  // 0.330 and AUC from 0.654 to 0.619, because a single field's sample is simply
  // a worse estimate of which words are genuinely new. Kept pooled on the
  // evidence.
  let totalLength = 0;
  for (const work of allWorks) {
    const terms = contentTerms(textOf(work));
    totalLength += terms.length;
    const unique = new Set(terms);
    const isPeer = peerSet.has(work);
    const target = isPeer ? historicalDocumentFrequency : currentDocumentFrequency;
    for (const term of unique) {
      documentFrequency.set(term, (documentFrequency.get(term) || 0) + 1);
      target.set(term, (target.get(term) || 0) + 1);
    }
  }
  const lexicon = lexiconFrom(documentFrequency, allWorks.length);
  const averageLength = totalLength / Math.max(1, allWorks.length);

  // The semantic space needs token order, so it reads a bounded sample of the
  // corpus rather than all of it.
  // Sampling strides through the corpus, so the corpus needs a canonical order:
  // striding through arrival order makes the space -- and therefore every score
  // built on it -- depend on the sequence records happened to be fetched in.
  const orderedWorks = [...allWorks].sort((left, right) => String(left.id).localeCompare(String(right.id)));
  const space = buildSemanticSpace(
    (function* streamTokens() {
      for (const work of orderedWorks) yield tokensForSpace(textOf(work));
    })(),
    lexicon,
    { documentCount: orderedWorks.length },
  );

  // Pass two builds the vectors and the terminology measure, again one document
  // at a time.
  const lexicalByWork = new Map();
  const semanticByWork = new Map();
  const emergentByWork = new Map();
  const semanticVectors = [];
  // Canonical order here too: the common-component subtraction sums every vector,
  // and floating-point addition is not associative, so arrival order would leave
  // a small but real dependence behind even after the sampling fix.
  for (const work of orderedWorks) {
    const terms = contentTerms(textOf(work));
    lexicalByWork.set(work, bm25Weights(terms, lexicon, averageLength));
    const vector = documentVector(terms, space, lexicon, averageLength);
    semanticByWork.set(work, vector);
    semanticVectors.push(vector);
    emergentByWork.set(work, emergenceOf(terms, lexicon, historicalDocumentFrequency, currentDocumentFrequency));
  }
  removeCommonComponent(semanticVectors, space.dimensions);

  const coCitation = buildCoCitationIndex(peers);

  return {
    settings,
    lexicon,
    space,
    averageLength,
    lexicalByWork,
    semanticByWork,
    emergentByWork,
    historicalDocumentFrequency,
    currentDocumentFrequency,
    peers,
    candidates,
  };
}

// Vocabulary the field did not have before this batch. Corroboration decides how
// much a new word is worth rather than whether it counts at all: a term several
// papers independently reached for is emerging terminology, while one nobody
// else uses is more likely a typo, an OCR artefact or an injected token, so it
// is admitted at a fraction of the weight instead of being trusted outright.
function emergenceOf(terms, lexicon, historical, current) {
  let weight = 0;
  let count = 0;
  const examples = [];
  for (const term of new Set(terms)) {
    // Single words only. A genuinely new technical term is a word --
    // "transformer", "CRISPR", "diffusion" -- while a previously unseen word
    // *pair* is mostly an accident of sentence order, and counting those let
    // ordinary papers accumulate "new terminology" by rephrasing familiar things.
    if (term.includes(" ")) continue;
    if ((historical.get(term) || 0) !== 0) continue;
    const idf = lexicon.idf.get(term);
    if (!idf) continue;
    const seen = current.get(term) || 0;
    // Two independent users, always. On a large corpus the lexicon's own
    // frequency floor already enforces this, but a fresh install indexes only a
    // few dozen papers and drops that floor to 1 -- which is exactly when a
    // single document's typos and OCR debris would otherwise read as the field
    // inventing vocabulary.
    if (seen < 2) continue;
    const corroboration = seen >= 3 ? 1 : 0.6;
    count += 1;
    weight += corroboration * Math.min(idf, EMERGENT_TERM_CAP);
    if (examples.length < 6) examples.push(term);
  }
  const unique = new Set(terms).size;
  return { count, density: unique ? Math.min(1, weight / unique) : 0, examples };
}


// A candidate's peer group: same subfield if the field has enough history there,
// otherwise domain, otherwise everything. Only work published before it counts,
// because novelty is a claim about what already existed.
export function cohortFor(work, model, groups) {
  const bySubfield = groups.bySubfield.get(work.subfieldId);
  const byDomain = groups.byDomain.get(work.domainId);
  if (bySubfield && bySubfield.length >= model.settings.minimumCohortPeers) return { peers: bySubfield, key: `subfield:${work.subfieldId}` };
  if (byDomain && byDomain.length >= model.settings.minimumCohortPeers) return { peers: byDomain, key: `domain:${work.domainId}` };
  return { peers: groups.all, key: "all" };
}

export function groupPeers(peers) {
  const bySubfield = new Map();
  const byDomain = new Map();
  for (const peer of peers) {
    if (peer.subfieldId) {
      if (!bySubfield.has(peer.subfieldId)) bySubfield.set(peer.subfieldId, []);
      bySubfield.get(peer.subfieldId).push(peer);
    }
    if (peer.domainId) {
      if (!byDomain.has(peer.domainId)) byDomain.set(peer.domainId, []);
      byDomain.get(peer.domainId).push(peer);
    }
  }
  return { bySubfield, byDomain, all: peers };
}

// Term -> [peerIndex, weight, ...]. Used only to shortlist which peers are worth
// a full semantic comparison; comparing against every peer in a large field
// would dominate the runtime for no gain, since almost all of them share nothing
// with the candidate.
export function buildPostings(peerLexical) {
  const postings = new Map();
  for (let index = 0; index < peerLexical.length; index += 1) {
    const entry = peerLexical[index];
    if (!entry?.ids) continue;
    for (let position = 0; position < entry.ids.length; position += 1) {
      const id = entry.ids[position];
      let list = postings.get(id);
      if (!list) postings.set(id, (list = []));
      list.push(index, entry.weights[position]);
    }
  }
  return postings;
}

function shortlist(candidateLexical, cohortPeers, peerLexical, postings, limit) {
  if (cohortPeers.length <= limit) return cohortPeers.map((_, index) => index);
  const dots = new Float64Array(cohortPeers.length);
  for (let entry = 0; entry < candidateLexical.ids.length; entry += 1) {
    const list = postings.get(candidateLexical.ids[entry]);
    if (!list) continue;
    const weight = candidateLexical.weights[entry];
    for (let position = 0; position < list.length; position += 2) dots[list[position]] += weight * list[position + 1];
  }
  const scored = [];
  for (let index = 0; index < cohortPeers.length; index += 1) scored.push([dots[index], index]);
  // Ties break on position so an equal-scoring pair cannot swap between runs.
  scored.sort((left, right) => right[0] - left[0] || left[1] - right[1]);
  const nearest = scored.slice(0, Math.min(limit, scored.length)).map((entry) => entry[1]);
  // A stratified stride over the rest keeps an unbiased picture of how dense the
  // field is overall; a pure nearest-neighbour set would make every field look
  // equally crowded.
  const spare = Math.max(0, limit - nearest.length);
  if (spare > 0) {
    const chosen = new Set(nearest);
    const stride = Math.max(1, Math.floor(cohortPeers.length / spare));
    for (let index = 0; index < cohortPeers.length && nearest.length < limit; index += stride) {
      if (!chosen.has(index)) {
        chosen.add(index);
        nearest.push(index);
      }
    }
  }
  return nearest;
}

export function measureCandidate(work, model, cohort, cache) {
  const candidateLexical = model.lexicalByWork.get(work);
  const candidateSemantic = model.semanticByWork.get(work);

  const olderPeers = cache.olderPeers;
  const measure = {
    peerCount: olderPeers.length,
    cohortKey: cohort.key,
  };

  // --- crowding, in meaning and in wording -----------------------------------
  let nearestSemantic = 0;
  let nearestLexical = 0;
  let nearestIndex = -1;
  const semanticSimilarities = [];
  if (olderPeers.length) {
    const picked = shortlist(
      candidateLexical,
      olderPeers,
      cache.peerLexical,
      cache.postings,
      model.settings.maxPeerComparisons,
    );
    for (const index of picked) {
      const peer = olderPeers[index];
      if (peer.id === work.id) continue;
      const semantic = semanticSimilarity(candidateSemantic, cache.peerSemantic[index]);
      const lexical = cosineOfWeighted(candidateLexical, cache.peerLexical[index]);
      // Either kind of closeness is evidence that this work already exists, so
      // the stronger one counts. Semantic similarity catches a paraphrase that
      // shares no vocabulary; lexical similarity catches the case the semantic
      // space is worst at -- a small or narrow corpus, where there is not enough
      // text for the co-occurrence statistics to mean anything yet.
      const similarity = Math.max(semantic, lexical);
      semanticSimilarities.push(similarity);
      if (similarity > nearestSemantic) {
        nearestSemantic = similarity;
        nearestIndex = index;
      }
      if (lexical > nearestLexical) nearestLexical = lexical;
    }
  }
  semanticSimilarities.sort((left, right) => right - left);
  const neighbourhood = semanticSimilarities.slice(0, model.settings.neighbourhoodSize);
  const neighbourhoodMean = neighbourhood.length
    ? neighbourhood.reduce((sum, value) => sum + value, 0) / neighbourhood.length
    : 0;
  measure.nearestSemantic = nearestSemantic;
  measure.nearestLexical = nearestLexical;
  measure.nearestPeer = nearestIndex >= 0 ? olderPeers[nearestIndex] : null;
  // Weighted toward the single nearest paper: one paper that already did this is
  // what makes a submission derivative, regardless of how empty the rest of the
  // neighbourhood is.
  // How ordinary the paper is: the share of it that survives having the corpus
  // average subtracted. Nothing left means the paper *is* the average.
  measure.genericness = 1 - (candidateSemantic.residualShare ?? 1);

  // Weighted toward the single nearest paper: one paper that already did this is
  // what makes a submission derivative, however empty the rest of the
  // neighbourhood is.
  //
  // Genericness stays a separate signal rather than being folded in here. Every
  // document loses a similar share of itself to the corpus average -- around half
  // -- so combining the two with a maximum simply floored every paper at the same
  // number and erased the difference the neighbour measure had found.
  measure.crowding = 0.6 * nearestSemantic + 0.4 * neighbourhoodMean;

  // --- centrality: how close to the middle of the field ----------------------
  measure.centrality = semanticSimilarity(candidateSemantic, cache.centroid);

  // --- terminology the field did not have ------------------------------------
  const emergence = model.emergentByWork.get(work) || { count: 0, density: 0, examples: [] };
  measure.emergentTermCount = emergence.count;
  measure.emergentDensity = emergence.density;
  measure.emergentTerms = emergence.examples;

  // How tightly the work sits on one topic. OpenAlex scores a paper against
  // several topics; a research article concentrates on one, while a review
  // spreads across the area it is summarising. This is the structural shape of
  // consolidation, visible without reading a single word of the title.
  const topicScores = (work.topics || []).map((topic) => Number(topic.score) || 0).filter((value) => value > 0);
  if (topicScores.length >= 2) {
    const sum = topicScores.reduce((total, value) => total + value, 0) || 1;
    let entropy = 0;
    for (const value of topicScores) {
      const share = value / sum;
      entropy -= share * Math.log(share);
    }
    measure.topicConcentration = 1 - entropy / Math.log(topicScores.length);
  } else {
    measure.topicConcentration = topicScores.length === 1 ? 1 : null;
  }

  // --- reference combination --------------------------------------------------
  // Scoped to the paper's own cohort, not to the whole index. "How often does
  // this field cite these two works together" is a question about a field; asking
  // it of a corpus spanning medicine, physics, materials science and computer
  // science answers a different and much less useful question, and made a paper's
  // score move when unrelated fields were added to the index.
  measure.combination = referenceCombination(work, cache.coCitation, { selfIndexed: cache.peerIds.has(work.id) });

  // --- bibliographic coupling -------------------------------------------------
  // Sharing a bibliography with existing work is the clearest sign of building
  // on the same foundations. It is also much denser than co-citation
  // statistics, because it only needs one peer to have cited one of the same
  // works rather than a specific pair of them.
  measure.coupling = null;
  const ownReferences = Array.isArray(work.referencedWorks) ? [...new Set(work.referencedWorks)] : [];
  if (ownReferences.length >= 3 && cache.referenceIndex.size) {
    const shared = new Map();
    for (const id of ownReferences) {
      const peersCiting = cache.referenceIndex.get(id);
      if (!peersCiting) continue;
      for (const peerIndex of peersCiting) shared.set(peerIndex, (shared.get(peerIndex) || 0) + 1);
    }
    // A paper is not coupled to itself, however perfectly its bibliography matches.
    const selfPosition = cache.positionById.get(work.id);
    if (selfPosition !== undefined) shared.delete(selfPosition);
    let strongest = 0;
    let total = 0;
    for (const [peerIndex, overlap] of shared) {
      const denominator = Math.max(1, Math.min(ownReferences.length, cache.peerReferenceCount[peerIndex] || 1));
      const value = overlap / denominator;
      total += value;
      if (value > strongest) strongest = value;
    }
    measure.coupling = {
      strongestCoupling: strongest,
      coupledPeerFraction: shared.size / Math.max(1, olderPeers.length),
      meanCoupling: shared.size ? total / shared.size : 0,
    };
  }

  // --- team size --------------------------------------------------------------
  // Wu, Wang and Evans (2019) measured this directly across 65 million papers:
  // large teams develop existing ideas, small teams disrupt them. It costs
  // nothing to compute and it is orthogonal to everything above.
  measure.teamSize = Array.isArray(work.authorships) ? work.authorships.length : 0;

  // --- how much of this we can actually believe -------------------------------
  const abstractTokens = tokenize(work.abstract || "", 400).length;
  measure.textCompleteness = abstractTokens >= 60 ? 1 : abstractTokens >= 20 ? 0.7 : abstractTokens ? 0.45 : 0.2;
  measure.semanticCoverage = candidateSemantic.covered;
  measure.peerConfidence = Math.min(1, Math.log1p(olderPeers.length) / Math.log1p(80));
  measure.referenceConfidence = measure.combination
    ? Math.min(1, Math.log1p(measure.combination.referenceCount) / Math.log1p(30))
    : 0;

  return measure;
}

export function buildCohortCache(cohortPeers, model, referenceDate) {
  // Sorted by id, not left in arrival order. Anchor selection strides through
  // this array and shortlist's stratified picks step through it, so its order is
  // load-bearing: without a canonical one, the same corpus delivered in a
  // different sequence produced a different score for 99% of papers, by as much
  // as half the scale.
  const olderPeers = (referenceDate
    ? cohortPeers.filter((peer) => !peer.publicationDate || peer.publicationDate < referenceDate)
    : [...cohortPeers]
  ).sort((left, right) => String(left.id).localeCompare(String(right.id)));
  const peerLexical = olderPeers.map((peer) => model.lexicalByWork.get(peer));
  const peerSemantic = olderPeers.map((peer) => model.semanticByWork.get(peer));
  const postings = buildPostings(peerLexical);

  // reference id -> which peers cite it, so bibliographic coupling costs one
  // pass over a candidate's own reference list rather than a scan of every peer.
  const referenceIndex = new Map();
  const peerIds = new Set();
  const positionById = new Map();
  const peerReferenceCount = new Int32Array(olderPeers.length);
  for (let index = 0; index < olderPeers.length; index += 1) {
    peerIds.add(olderPeers[index].id);
    positionById.set(olderPeers[index].id, index);
    const references = olderPeers[index]?.referencedWorks;
    if (!Array.isArray(references)) continue;
    peerReferenceCount[index] = references.length;
    for (const id of new Set(references)) {
      let list = referenceIndex.get(id);
      if (!list) referenceIndex.set(id, (list = []));
      list.push(index);
    }
  }

  const dimensions = model.space.dimensions;
  const centroidVector = new Float32Array(dimensions);
  let counted = 0;
  for (const entry of peerSemantic) {
    if (!entry?.norm) continue;
    counted += 1;
    for (let slot = 0; slot < dimensions; slot += 1) centroidVector[slot] += entry.vector[slot];
  }
  let squared = 0;
  for (let slot = 0; slot < dimensions; slot += 1) squared += centroidVector[slot] * centroidVector[slot];
  const norm = Math.sqrt(squared);
  if (norm > 0) for (let slot = 0; slot < dimensions; slot += 1) centroidVector[slot] /= norm;

  return {
    olderPeers,
    peerLexical,
    peerSemantic,
    postings,
    referenceIndex,
    peerIds,
    positionById,
    peerReferenceCount,
    coCitation: buildCoCitationIndex(olderPeers),
    centroid: { vector: centroidVector, norm: counted ? 1 : 0 },
  };
}
