// Text statistics shared by novelty and relevance.
//
// Everything here is deterministic and corpus-local: no model is downloaded, no
// request is made, and the same corpus always produces the same numbers.

const STOPWORDS = new Set(
  `a about above after again against all also am an and any are as at be because been before being below between both but by can cannot could did do does doing down during each few for from further had has have having he her here hers herself him himself his how i if in into is it its itself just me more most my myself no nor not now of off on once only or other others our ours ourselves out over own same she should so some such than that the their theirs them themselves then there these they this those through to too under until up very was we were what when where which while who whom why will with would you your yours yourself yourselves
   we our us paper papers work works study studies present presents presented propose proposed proposes show shows shown using used use uses based approach approaches method methods results result new novel provide provides provided
   thus hence therefore however moreover furthermore additionally respectively via within without towards toward among across upon whereas although though since given due able may might must shall`
    .split(/\s+/)
    .filter(Boolean),
);

// Kept out of the stoplist above on purpose: these read as filler in isolation
// but carry the field's meaning in a bigram ("transfer learning", "phase
// transition"), so they are only dropped as standalone unigrams.
const WEAK_UNIGRAMS = new Set(["model", "models", "data", "system", "systems", "analysis", "performance", "framework", "algorithm", "algorithms"]);

// NFKD splits an accented letter into base + combining mark; this strips the
// marks so "Schrodinger" and "Schrödinger" index as the same term.
const COMBINING_MARKS = /[̀-ͯ]/g;

export function normalizeText(value) {
  return String(value ?? "")
    .normalize("NFKD")
    .toLowerCase()
    .replace(COMBINING_MARKS, "")
    .replace(/[^a-z0-9\-\s]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

// Tokens are 2+ characters and must contain a letter, so version numbers and
// stray digits do not become vocabulary. Hyphenated compounds are kept whole and
// also split, because "retrieval-augmented" and "retrieval" should both index.
export function tokenize(value, limit = 900) {
  const normalized = normalizeText(value);
  if (!normalized) return [];
  const out = [];
  for (const raw of normalized.split(" ")) {
    if (out.length >= limit) break;
    if (!raw || raw.length < 2 || !/[a-z]/.test(raw)) continue;
    if (STOPWORDS.has(raw)) continue;
    out.push(raw);
    if (raw.includes("-")) {
      for (const part of raw.split("-")) {
        if (part.length >= 3 && !STOPWORDS.has(part) && out.length < limit) out.push(part);
      }
    }
  }
  return out;
}

export function bigramsOf(tokens) {
  const out = [];
  for (let index = 0; index + 1 < tokens.length; index += 1) {
    if (tokens[index].includes("-") || tokens[index + 1].includes("-")) continue;
    out.push(`${tokens[index]} ${tokens[index + 1]}`);
  }
  return out;
}

// Content terms for a work: unigrams minus the weak ones, plus bigrams. Bigrams
// are what let "language model" mean something different from "language" and
// "model" scattered across an abstract.
export function contentTerms(text, { limit = 900, includeBigrams = true } = {}) {
  const tokens = tokenize(text, limit);
  const unigrams = tokens.filter((token) => !WEAK_UNIGRAMS.has(token));
  return includeBigrams ? [...unigrams, ...bigramsOf(tokens)] : unigrams;
}

export function termCounts(terms) {
  const counts = new Map();
  for (const term of terms) counts.set(term, (counts.get(term) || 0) + 1);
  return counts;
}

export function textOf(work) {
  return `${work?.title || ""} . ${work?.abstract || ""}`;
}

// Document frequencies plus the IDF derived from them. Terms seen in fewer than
// `minimumDocumentFrequency` documents are dropped outright, which is what makes
// the whole pipeline immune to typos, OCR debris and injected nonsense: a term
// nobody else uses is not evidence of a new idea, it is noise.
export function buildLexicon(documents, options = {}) {
  const documentFrequency = new Map();
  for (const terms of documents) {
    for (const term of new Set(terms)) documentFrequency.set(term, (documentFrequency.get(term) || 0) + 1);
  }
  return lexiconFrom(documentFrequency, documents.length, { retainFrequencies: true, ...options });
}

// Same lexicon, built from frequencies that were accumulated in a streaming pass.
// Holding one term array per document is what a corpus of ten thousand papers
// cannot afford: the arrays alone ran to hundreds of megabytes, all of it alive
// at once purely to be counted.
export function lexiconFrom(documentFrequency, documentCount, options = {}) {
  const total = Math.max(1, documentCount);
  // Both cutoffs adapt to corpus size. A fixed "seen in at least 3 documents"
  // rule empties the vocabulary of a small corpus, and a fixed "in at most half
  // the documents" rule empties the vocabulary of a narrow one: a subfield where
  // every paper legitimately says "electrolyte" would keep no terms at all and
  // every similarity would read as zero. The BM25 IDF below already drives
  // ubiquitous terms to near-zero weight, so the upper cutoff only has to stop
  // them costing time -- it does not have to carry the down-weighting.
  const minimumDocumentFrequency = options.minimumDocumentFrequency ?? (total < 60 ? 1 : total < 400 ? 2 : 3);
  const maximumDocumentShare = options.maximumDocumentShare ?? (total < 200 ? 1 : 0.85);
  const ceiling = Math.max(2, Math.floor(total * maximumDocumentShare));
  const idf = new Map();
  // Every kept term also gets an integer id. Document vectors are then stored as
  // parallel typed arrays of ids and weights rather than as Maps keyed by
  // strings: a bigram like "sodium battery" is built with a template literal, so
  // it is a fresh allocation in every document that contains it, and a corpus of
  // ten thousand papers was retaining several million of them.
  const ids = new Map();
  const idfById = [];
  for (const [term, frequency] of documentFrequency) {
    if (frequency < minimumDocumentFrequency || frequency > ceiling) continue;
    const value = Math.log(1 + (total - frequency + 0.5) / (frequency + 0.5));
    idf.set(term, value);
    ids.set(term, idfById.length);
    idfById.push(value);
  }
  // The raw frequency table is a quarter of a million string-keyed entries and
  // nothing in the scoring path reads it once the IDF has been derived, so it is
  // dropped unless a caller asks to keep it. Holding it cost roughly thirty
  // megabytes for the whole length of a pass.
  return {
    documentFrequency: options.retainFrequencies ? documentFrequency : null,
    idf,
    ids,
    idfById,
    documentCount: total,
  };
}

// BM25 term weights for one document, restricted to the lexicon and returned as
// sorted parallel typed arrays: `ids` ascending, `weights` alongside.
export function bm25Weights(terms, lexicon, averageLength, { k1 = 1.4, b = 0.7 } = {}) {
  const counts = new Map();
  for (const term of terms) {
    const id = lexicon.ids.get(term);
    if (id === undefined) continue;
    counts.set(id, (counts.get(id) || 0) + 1);
  }
  const length = Math.max(1, terms.length);
  const ids = new Int32Array(counts.size);
  const weights = new Float32Array(counts.size);
  let position = 0;
  for (const id of counts.keys()) ids[position++] = id;
  ids.sort();
  let squaredNorm = 0;
  for (let index = 0; index < ids.length; index += 1) {
    const frequency = counts.get(ids[index]);
    const saturated = (frequency * (k1 + 1)) / (frequency + k1 * (1 - b + (b * length) / Math.max(1, averageLength)));
    const weight = lexicon.idfById[ids[index]] * saturated;
    weights[index] = weight;
    squaredNorm += weight * weight;
  }
  return { ids, weights, norm: Math.sqrt(squaredNorm) || 0 };
}

// Merge of two ascending id arrays -- no hashing, no allocation.
export function cosineOfWeighted(left, right) {
  if (!left?.norm || !right?.norm) return 0;
  let dot = 0;
  let a = 0;
  let b = 0;
  while (a < left.ids.length && b < right.ids.length) {
    const leftId = left.ids[a];
    const rightId = right.ids[b];
    if (leftId === rightId) {
      dot += left.weights[a] * right.weights[b];
      a += 1;
      b += 1;
    } else if (leftId < rightId) a += 1;
    else b += 1;
  }
  const value = dot / (left.norm * right.norm);
  return value > 0 ? Math.min(1, value) : 0;
}
