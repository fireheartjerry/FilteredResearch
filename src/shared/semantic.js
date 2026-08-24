// A semantic space computed from the corpus itself, in the extension, in about
// a second.
//
// Why not plain TF-IDF cosine: two papers describing the same idea in different
// words share almost no vocabulary, so a lexical measure calls them distant and
// reports the second one as novel. Why not a downloaded embedding model: it
// would be tens of megabytes of remote weights in an MV3 extension that promises
// to ship no remote code, and it would still not know this month's terminology.
//
// Random indexing gets most of the way there for almost nothing. Every term is
// assigned a fixed sparse random signature; a term's meaning vector is the sum
// of the signatures of the words it appears next to. Terms used in the same
// contexts end up pointing the same way, which is the property that makes
// paraphrase look like paraphrase. Terms that appear once -- typos, OCR debris,
// injected nonsense -- never enter the vocabulary at all, so they contribute
// nothing rather than reading as maximally informative.

import { tokenize, bm25Weights } from "./text.js";

export const SEMANTIC_DIMENSIONS = 128;
const SIGNATURE_NONZEROS = 8;
const CONTEXT_WINDOW = 4;
const MAX_SPACE_DOCUMENTS = 4000;

// FNV-1a. A term's signature must be identical on every machine and every run,
// so it is derived from the term's own characters rather than from a stream of
// random numbers whose order would depend on corpus iteration.
function hashTerm(term, salt) {
  let hash = 2166136261 ^ salt;
  for (let index = 0; index < term.length; index += 1) {
    hash ^= term.charCodeAt(index);
    hash = Math.imul(hash, 16777619) >>> 0;
  }
  return hash >>> 0;
}

// Sparse ternary signature: a handful of +1 and -1 positions. Near-orthogonal to
// every other term's signature, which is the whole trick -- unrelated terms do
// not interfere, related terms accumulate.
function signatureOf(term) {
  const positions = new Int16Array(SIGNATURE_NONZEROS);
  const signs = new Int8Array(SIGNATURE_NONZEROS);
  for (let index = 0; index < SIGNATURE_NONZEROS; index += 1) {
    const hash = hashTerm(term, index * 2654435761);
    positions[index] = hash % SEMANTIC_DIMENSIONS;
    signs[index] = (hash >>> 16) & 1 ? 1 : -1;
  }
  return { positions, signs };
}

// `documentTokens` is an iterable, not an array, so the caller can stream token
// lists in one at a time instead of materialising one per document up front.
export function buildSemanticSpace(documentTokens, lexicon, { dimensions = SEMANTIC_DIMENSIONS, documentCount = 0 } = {}) {
  const vocabulary = new Map();
  // Lexicon id -> position in this space, so a document vector can be built from
  // the integer ids its BM25 weights already carry instead of re-hashing strings.
  const indexByLexiconId = new Int32Array(lexicon.idfById.length).fill(-1);
  for (const [term, lexiconId] of lexicon.ids) {
    // Bigrams get a meaning vector by composition later; only unigrams take part
    // in the co-occurrence pass.
    if (term.includes(" ")) continue;
    indexByLexiconId[lexiconId] = vocabulary.size;
    vocabulary.set(term, vocabulary.size);
  }
  const size = vocabulary.size;
  const vectors = new Float32Array(size * dimensions);
  if (!size) return { vocabulary, indexByLexiconId, vectors, dimensions, empty: true };

  const signatures = new Array(size);
  for (const [term, index] of vocabulary) signatures[index] = signatureOf(term);

  // A bounded sample is enough to describe how words keep company, but it has to
  // be spread across the corpus. Taking the first N instead of every Nth read
  // only the historical half of a peers-then-candidates stream and produced a
  // measurably worse space.
  const step = documentCount > MAX_SPACE_DOCUMENTS ? Math.ceil(documentCount / MAX_SPACE_DOCUMENTS) : 1;
  let documentIndex = -1;
  for (const tokens of documentTokens) {
    documentIndex += 1;
    if (documentIndex % step !== 0) continue;
    const mapped = [];
    for (const token of tokens) {
      const index = vocabulary.get(token);
      mapped.push(index === undefined ? -1 : index);
    }
    for (let position = 0; position < mapped.length; position += 1) {
      const target = mapped[position];
      if (target < 0) continue;
      const base = target * dimensions;
      const from = Math.max(0, position - CONTEXT_WINDOW);
      const to = Math.min(mapped.length - 1, position + CONTEXT_WINDOW);
      for (let neighbour = from; neighbour <= to; neighbour += 1) {
        if (neighbour === position) continue;
        const context = mapped[neighbour];
        if (context < 0) continue;
        // Closer words say more about meaning than words at the edge of the
        // window, so their signature counts for more.
        const weight = 1 / Math.abs(neighbour - position);
        const { positions, signs } = signatures[context];
        for (let slot = 0; slot < SIGNATURE_NONZEROS; slot += 1) {
          vectors[base + positions[slot]] += signs[slot] * weight;
        }
      }
    }
  }

  // Unit-length term vectors, so a common term does not dominate a document
  // vector purely by appearing everywhere.
  for (let index = 0; index < size; index += 1) {
    const base = index * dimensions;
    let squared = 0;
    for (let slot = 0; slot < dimensions; slot += 1) squared += vectors[base + slot] * vectors[base + slot];
    const norm = Math.sqrt(squared);
    if (norm > 0) for (let slot = 0; slot < dimensions; slot += 1) vectors[base + slot] /= norm;
  }

  return { vocabulary, indexByLexiconId, vectors, dimensions, empty: false };
}

// A document is the IDF-weighted centroid of the meanings of its terms. Terms
// outside the vocabulary contribute nothing, which is exactly the desired
// behaviour for nonsense.
export function documentVector(terms, space, lexicon, averageLength) {
  const out = new Float32Array(space.dimensions);
  if (space.empty) return { vector: out, norm: 0, covered: 0 };
  const { ids, weights } = bm25Weights(terms, lexicon, averageLength);
  let covered = 0;
  for (let entry = 0; entry < ids.length; entry += 1) {
    const index = space.indexByLexiconId[ids[entry]];
    if (index < 0) continue;
    covered += 1;
    const weight = weights[entry];
    const base = index * space.dimensions;
    for (let slot = 0; slot < space.dimensions; slot += 1) out[slot] += weight * space.vectors[base + slot];
  }
  let squared = 0;
  for (let slot = 0; slot < space.dimensions; slot += 1) squared += out[slot] * out[slot];
  const norm = Math.sqrt(squared);
  if (norm > 0) for (let slot = 0; slot < space.dimensions; slot += 1) out[slot] /= norm;
  return { vector: out, norm, covered };
}

export function semanticSimilarity(left, right) {
  if (!left?.norm || !right?.norm) return 0;
  let dot = 0;
  for (let slot = 0; slot < left.vector.length; slot += 1) dot += left.vector[slot] * right.vector[slot];
  return dot > 0 ? Math.min(1, dot) : 0;
}

// Every document in a corpus shares a large common direction -- the vocabulary
// of academic writing itself -- and it swamps the part that distinguishes one
// paper from another. Uncentred, two unrelated papers in this corpus score 0.44
// similarity and same-topic pairs separate from different-topic pairs at AUC
// 0.53, barely above a coin flip. Projecting out the corpus mean lifts that to
// 0.63 and drops unrelated pairs to ~0.00, which is what makes "how crowded is
// this neighbourhood" mean anything at all.
export function removeCommonComponent(documentVectors, dimensions) {
  const mean = new Float32Array(dimensions);
  let counted = 0;
  for (const entry of documentVectors) {
    if (!entry?.norm) continue;
    counted += 1;
    for (let slot = 0; slot < dimensions; slot += 1) mean[slot] += entry.vector[slot];
  }
  if (!counted) return;
  let squared = 0;
  for (let slot = 0; slot < dimensions; slot += 1) squared += mean[slot] * mean[slot];
  const norm = Math.sqrt(squared);
  if (!norm) return;
  for (let slot = 0; slot < dimensions; slot += 1) mean[slot] /= norm;

  for (const entry of documentVectors) {
    if (!entry?.norm) continue;
    let projection = 0;
    for (let slot = 0; slot < dimensions; slot += 1) projection += entry.vector[slot] * mean[slot];
    for (let slot = 0; slot < dimensions; slot += 1) entry.vector[slot] -= projection * mean[slot];
    let residual = 0;
    for (let slot = 0; slot < dimensions; slot += 1) residual += entry.vector[slot] * entry.vector[slot];
    const length = Math.sqrt(residual);
    // How much of the paper is left once the corpus average is taken away. A
    // survey is close to the average by construction, so almost nothing survives
    // -- and a vector of almost nothing is nearly orthogonal to everything, which
    // made the most generic papers in the corpus read as the most unlike it. That
    // inversion is why this is kept and used rather than discarded: low residual
    // means generic, and generic is the opposite of novel.
    // documentVector already returned a unit vector (its `norm` field is the
    // pre-normalisation magnitude), so the surviving length IS the share.
    entry.residualShare = length;
    entry.norm = length;
    if (length > 0) for (let slot = 0; slot < dimensions; slot += 1) entry.vector[slot] /= length;
  }
}

// Nearest terms in meaning, used to expand a short query ("RAG" -> the words
// that keep company with retrieval-augmented generation in this corpus).
export function nearestTerms(term, space, count = 5, minimumSimilarity = 0.35) {
  const index = space.vocabulary.get(term);
  if (index === undefined || space.empty) return [];
  const dimensions = space.dimensions;
  const base = index * dimensions;
  const found = [];
  for (const [other, otherIndex] of space.vocabulary) {
    if (otherIndex === index) continue;
    const otherBase = otherIndex * dimensions;
    let dot = 0;
    for (let slot = 0; slot < dimensions; slot += 1) dot += space.vectors[base + slot] * space.vectors[otherBase + slot];
    if (dot >= minimumSimilarity) found.push([other, dot]);
  }
  found.sort((left, right) => right[1] - left[1]);
  return found.slice(0, count);
}

export function tokensForSpace(text) {
  return tokenize(text, 400).filter((token) => !token.includes("-"));
}
