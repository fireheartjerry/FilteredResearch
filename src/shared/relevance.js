// Graded relevance ranking.
//
// `SPEC.md` and `README.md` both say interest phrases rank results. Until now
// they did not: `matchesResearchFilters` returned a boolean, so every paper that
// mentioned a phrase was equally relevant and the feed's order carried no
// information about what the reader actually asked for. Searching "RAG" over a
// real index returned Tagalog relative-clause processing above every
// retrieval-augmented generation paper in the corpus.
//
// This is BM25 over title and abstract with three additions that matter for
// research text: title terms count for more, adjacent query words are rewarded
// for staying adjacent, and a short query is expanded -- through a glossary for
// the abbreviations researchers actually type, and through the corpus's own
// semantic space for everything else.

import { contentTerms, tokenize, buildLexicon, termCounts, textOf, normalizeText } from "./text.js";
import { nearestTerms } from "./semantic.js";
import { embedWorks, embedQuery, cosine } from "./embedding.js";

// How much of the ordering the embedding decides when one is available. Swept on
// the tuning split, where lexical alone scores nDCG@20 0.433 and the model alone
// 0.638, with the blend rising monotonically between them. The model therefore
// decides the order outright and the lexical score survives only as the
// tie-break, which is what still catches an acronym or a rare technical term the
// encoder has never seen.
//
// Variants that were measured and are not used: encoding title and abstract
// separately and taking the better match (0.640 against 0.639, three times the
// inference), and three retrieval-specific encoders of the same size, all of
// which lost to this general-purpose one -- bge-small 0.606, gte-small 0.616,
// e5-small 0.527.
const EMBEDDING_SHARE = 1;

// Retrieval parameters, swept on the tuning split
// (eval/experiments/relevance-sweep.mjs).
const TITLE_WEIGHT = 1.2;
const K1 = 0.9;
const B = 0.9;
const EXPANSION_WEIGHT = 0.45;
const PROXIMITY_BONUS = 0.35;

// Pseudo-relevance feedback. A reader's phrase and the papers that answer it
// often share no words: "Quantum and electron transport phenomena" is not what a
// paper about ballistic conduction in nanowires actually says. So the query is
// asked once, the strongest terms of the best few answers are harvested, and the
// query is asked again carrying them. Measured on the tuning split this is the
// single largest gain available to the ranker (nDCG@20 0.387 -> 0.414); query
// expansion through the semantic space and a hybrid vector score were both tried
// and both made it worse.
const FEEDBACK_DEPTH = 5;
const FEEDBACK_TERMS = 40;
const FEEDBACK_WEIGHT = 0.3;

// Fitted by coordinate ascent on nDCG@20 over the tuning split. The negative
// coefficients are not saying that a title match hurts -- they are decorrelating
// features that BM25 has already counted once.
const RERANK = Object.freeze({
  bias: 0,
  bm25: 1,
  coverage: 1,
  proximity: 1,
  titleShare: -1,
  matchShare: -0.5,
  length: -0.2,
  rarity: 4,
});

// Abbreviations researchers actually type. A known acronym resolves to its
// established meaning rather than to any phrase whose initials happen to line
// up, so "RAG" finds retrieval-augmented generation and not "robust adaptive
// gradient".
export const ACRONYM_GLOSSARY = Object.freeze({
  rag: ["retrieval augmented generation", "retrieval augmented generative"],
  llm: ["large language model", "large language models"],
  llms: ["large language models"],
  nlp: ["natural language processing"],
  cv: ["computer vision"],
  rl: ["reinforcement learning"],
  rlhf: ["reinforcement learning from human feedback"],
  gan: ["generative adversarial network", "generative adversarial networks"],
  cnn: ["convolutional neural network", "convolutional neural networks"],
  rnn: ["recurrent neural network", "recurrent neural networks"],
  gnn: ["graph neural network", "graph neural networks"],
  vlm: ["vision language model", "vision language models"],
  moe: ["mixture of experts"],
  sae: ["sparse autoencoder", "sparse autoencoders"],
  peft: ["parameter efficient fine tuning"],
  lora: ["low rank adaptation"],
  sft: ["supervised fine tuning"],
  dpo: ["direct preference optimization"],
  mcts: ["monte carlo tree search"],
  ssl: ["self supervised learning"],
  ood: ["out of distribution"],
  qa: ["question answering"],
  asr: ["automatic speech recognition"],
  tts: ["text to speech"],
  ocr: ["optical character recognition"],
  slam: ["simultaneous localization and mapping"],
  mpc: ["model predictive control"],
  pde: ["partial differential equation", "partial differential equations"],
  dft: ["density functional theory"],
  mri: ["magnetic resonance imaging"],
  ai: ["artificial intelligence"],
  agi: ["artificial general intelligence"],
  hci: ["human computer interaction"],
  iot: ["internet of things"],
  api: ["application programming interface"],
});

const ACRONYM_FOR_PHRASE = new Map();
for (const [acronym, expansions] of Object.entries(ACRONYM_GLOSSARY)) {
  for (const phrase of expansions) ACRONYM_FOR_PHRASE.set(phrase, acronym);
}

function fieldTerms(work) {
  const title = String(work?.title || "");
  const abstract = String(work?.abstract || "");
  return {
    title: contentTerms(title, { limit: 60 }),
    body: contentTerms(`${title} . ${abstract}`, { limit: 900 }),
    bodyTokens: tokenize(`${title} . ${abstract}`, 900),
  };
}

export function buildRelevanceIndex(works, { space = null, ...tuning } = {}) {
  const entries = works.map((work) => {
    const parts = fieldTerms(work);
    return { work, ...parts, counts: termCounts(parts.body), titleCounts: termCounts(parts.title) };
  });
  const lexicon = buildLexicon(entries.map((entry) => entry.body), { minimumDocumentFrequency: 1, maximumDocumentShare: 0.85 });
  const averageLength = entries.reduce((sum, entry) => sum + entry.body.length, 0) / Math.max(1, entries.length);
  const positions = new Map();
  for (const entry of entries) {
    const map = new Map();
    for (let index = 0; index < entry.bodyTokens.length; index += 1) {
      const token = entry.bodyTokens[index];
      if (!map.has(token)) map.set(token, []);
      const list = map.get(token);
      if (list.length < 24) list.push(index);
    }
    entry.positions = map;
  }
  return { entries, lexicon, averageLength, space, tuning, byId: new Map(entries.map((entry) => [entry.work.id, entry])) };
}

// A query becomes a set of weighted terms. Exact wording carries full weight;
// glossary expansions carry full weight too, because an abbreviation and its
// expansion are the same subject; corpus-derived neighbours carry less, because
// they are a guess.
export function expandQuery(query, index) {
  const normalized = normalizeText(query);
  const queryTokens = tokenize(normalized, 24);
  if (!queryTokens.length) return { terms: new Map(), phrases: [], expansions: [] };

  const terms = new Map();
  const add = (term, weight) => {
    if (!term) return;
    terms.set(term, Math.max(terms.get(term) || 0, weight));
  };
  for (const token of queryTokens) add(token, 1);
  for (let position = 0; position + 1 < queryTokens.length; position += 1) {
    add(`${queryTokens[position]} ${queryTokens[position + 1]}`, 1.3);
  }

  const expansions = [];
  // Acronym -> expansion.
  if (queryTokens.length === 1) {
    for (const phrase of ACRONYM_GLOSSARY[queryTokens[0]] || []) {
      const parts = tokenize(phrase, 8);
      expansions.push(phrase);
      for (const part of parts) add(part, 1);
      for (let position = 0; position + 1 < parts.length; position += 1) add(`${parts[position]} ${parts[position + 1]}`, 1.3);
    }
  }
  // Expansion -> acronym, so the full phrase also finds papers that only ever
  // print the abbreviation.
  const joined = queryTokens.join(" ");
  const acronym = ACRONYM_FOR_PHRASE.get(joined);
  if (acronym) {
    expansions.push(acronym);
    add(acronym, 1);
  }

  // Corpus-derived neighbours. These are what let a query find papers that use a
  // different word for the same thing without a hand-written thesaurus.
  if (index?.space && !index.space.empty) {
    for (const token of queryTokens) {
      for (const [neighbour, similarity] of nearestTerms(token, index.space, 4, 0.45)) {
        if (terms.has(neighbour)) continue;
        expansions.push(neighbour);
        add(neighbour, EXPANSION_WEIGHT * similarity);
      }
    }
  }

  return { terms, phrases: queryTokens, expansions: [...new Set(expansions)].slice(0, 12) };
}

function bm25For(entry, term, lexicon, averageLength, tuning = {}) {
  const frequency = entry.counts.get(term) || 0;
  if (!frequency) return 0;
  const idf = lexicon.idf.get(term);
  if (!idf) return 0;
  const titleBoost = 1 + (tuning.titleWeight ?? TITLE_WEIGHT) * ((entry.titleCounts.get(term) || 0) > 0 ? 1 : 0);
  const length = Math.max(1, entry.body.length);
  const k1 = tuning.k1 ?? K1;
  const b = tuning.b ?? B;
  const saturated = (frequency * (k1 + 1)) / (frequency + k1 * (1 - b + (b * length) / Math.max(1, averageLength)));
  return idf * saturated * titleBoost;
}

// Query words that stay together in the document are far more likely to mean
// what the reader meant. Without this, "language model" matches a paper that
// says "language" in one sentence and "model" in another.
function proximityScore(entry, phrases) {
  if (phrases.length < 2) return 0;
  let best = 0;
  for (let index = 0; index + 1 < phrases.length; index += 1) {
    const left = entry.positions.get(phrases[index]);
    const right = entry.positions.get(phrases[index + 1]);
    if (!left || !right) continue;
    let closest = Infinity;
    for (const a of left) for (const b of right) closest = Math.min(closest, Math.abs(a - b));
    if (closest < Infinity) best += 1 / (1 + closest);
  }
  return best / (phrases.length - 1);
}

export function scoreEntry(entry, expanded, index, overrides = {}) {
  const tuning = { ...(index.tuning || {}), ...overrides };
  let raw = 0;
  let matched = 0;
  let titleHits = 0;
  let strongestMatchedIdf = 0;
  const contributions = [];
  for (const [term, weight] of expanded.terms) {
    const value = bm25For(entry, term, index.lexicon, index.averageLength, tuning) * weight;
    if ((entry.titleCounts.get(term) || 0) > 0) titleHits += 1;
    if (value > 0) {
      raw += value;
      matched += 1;
      strongestMatchedIdf = Math.max(strongestMatchedIdf, index.lexicon.idf.get(term) || 0);
      if (contributions.length < 6) contributions.push({ term, points: Number(value.toFixed(2)) });
    }
  }
  if (raw <= 0) return { raw: 0, score: 0, matched: 0, contributions: [], rank: 0 };
  const proximity = proximityScore(entry, expanded.phrases);
  raw *= 1 + (tuning.proximityBonus ?? PROXIMITY_BONUS) * proximity;
  // Coverage matters as much as intensity: a paper that hits every query term
  // once is more relevant than one that hits a single term repeatedly.
  const coverage = matched / Math.max(1, expanded.terms.size);
  const coverageWeight = tuning.coverageWeight ?? 0.45;
  raw *= (1 - coverageWeight) + coverageWeight * Math.sqrt(coverage);
  const queryWidth = Math.max(1, expanded.terms.size);
  return {
    raw,
    matched,
    proximity,
    coverage,
    contributions,
    // The value results are actually ordered by. BM25 alone cannot weigh its own
    // by-products against each other -- whether the match landed in the title,
    // how much of the query was covered, how rare the rarest matched term was --
    // so those are combined by coefficients fitted on the tuning split
    // (eval/experiments/relevance-reranker.mjs) instead of by assertion.
    rank: RERANK.bias
      + RERANK.bm25 * Math.log1p(raw)
      + RERANK.coverage * coverage
      + RERANK.proximity * proximity
      + RERANK.titleShare * (titleHits / queryWidth)
      + RERANK.matchShare * (matched / queryWidth)
      + RERANK.length * (Math.log1p(entry.body.length) / 10)
      + RERANK.rarity * (strongestMatchedIdf / 10),
  };
}

// BM25 has no natural ceiling, so the displayed 0-100 comes from a saturating
// curve rather than from dividing by the corpus maximum -- otherwise one
// unusually strong match would compress every other result toward zero.
export function toDisplayScore(raw) {
  if (!(raw > 0)) return 0;
  return Math.round(100 * (1 - Math.exp(-raw / 9)) * 100) / 100;
}

// Harvest the vocabulary the best answers actually use, and add it to the query
// at reduced weight. Terms already in the query are skipped so the original
// wording is never diluted.
export function withFeedback(index, expanded) {
  const initial = index.entries
    .map((entry) => ({ entry, raw: scoreEntry(entry, expanded, index).raw }))
    .filter((item) => item.raw > 0)
    .sort((left, right) => right.raw - left.raw || String(left.entry.work.id).localeCompare(String(right.entry.work.id)))
    .slice(0, FEEDBACK_DEPTH);
  if (!initial.length) return expanded;

  const weightByTerm = new Map();
  for (const { entry } of initial) {
    for (const [term, count] of entry.counts) {
      const idf = index.lexicon.idf.get(term);
      if (!idf || expanded.terms.has(term)) continue;
      weightByTerm.set(term, (weightByTerm.get(term) || 0) + idf * Math.min(3, count));
    }
  }
  const harvested = [...weightByTerm.entries()]
    .sort((left, right) => right[1] - left[1] || left[0].localeCompare(right[0]))
    .slice(0, FEEDBACK_TERMS);
  if (!harvested.length) return expanded;

  const strongest = harvested[0][1] || 1;
  const terms = new Map(expanded.terms);
  for (const [term, value] of harvested) terms.set(term, FEEDBACK_WEIGHT * (value / strongest));
  return { ...expanded, terms, feedbackTerms: harvested.slice(0, 8).map(([term]) => term) };
}

// The same ranking, with a semantic model in the loop. Async because the model
// is, and separate from `rankByRelevance` so that nothing on the scoring path can
// accidentally acquire a dependency on it. Falls back to the lexical order
// whenever the model is unavailable, which is the normal case for an install that
// has not finished unpacking it.
export async function rankByRelevanceWithModel(works, query, options = {}) {
  const index = options.index || buildRelevanceIndex(works, { space: options.space || null });
  const base = expandQuery(query, index);
  if (!base.terms.size) return [...works];
  const expanded = options.feedback === false ? base : withFeedback(index, base);

  const lexical = new Map(index.entries.map((entry) => {
    const result = scoreEntry(entry, expanded, index);
    return [entry.work.id, result.raw > 0 ? result.rank : -50];
  }));

  // The model is given the EXPANDED query, not the raw one. An encoder has no
  // idea that "DFT" means density functional theory -- it is three letters that
  // occur in many contexts -- so embedding the bare acronym returned newsletters
  // and animal studies. Handing it the glossary expansion alongside the original
  // restores the acronym handling that the lexical path always had, and which
  // going fully semantic would otherwise have thrown away.
  const semanticQuery = [query, ...base.expansions.filter((term) => term.includes(" "))].join(". ");
  const vectors = await embedWorks(works, options.embedding || {});
  const queryVector = vectors ? await embedQuery(semanticQuery, options.embedding || {}) : null;
  if (!vectors || !queryVector) {
    return [...works].sort((left, right) =>
      (lexical.get(right.id) ?? -50) - (lexical.get(left.id) ?? -50) ||
      String(left.id).localeCompare(String(right.id)));
  }

  const share = options.embeddingShare ?? EMBEDDING_SHARE;
  const combined = works.map((work, position) => ({
    work,
    // The lexical rank is a log-odds-ish quantity and the cosine is bounded, so
    // the cosine is scaled to put the two on a comparable footing before mixing.
    score: (1 - share) * (lexical.get(work.id) ?? -50) + share * 12 * cosine(queryVector, vectors[position]),
  }));
  combined.sort((left, right) => right.score - left.score || String(left.work.id).localeCompare(String(right.work.id)));
  return combined.map((entry) => entry.work);
}

export function rankByRelevance(works, query, options = {}) {
  const index = options.index || buildRelevanceIndex(works, { space: options.space || null });
  const base = expandQuery(query, index);
  if (!base.terms.size) return [...works];
  const expanded = options.feedback === false ? base : withFeedback(index, base);
  const scored = index.entries.map((entry) => ({ entry, ...scoreEntry(entry, expanded, index) }));
  scored.sort((left, right) =>
    (right.raw > 0 ? right.rank : -Infinity) - (left.raw > 0 ? left.rank : -Infinity) ||
    right.raw - left.raw ||
    String(left.entry.work.id).localeCompare(String(right.entry.work.id)));
  return scored.map((item) => item.entry.work);
}

// Relevance for one paper against one query, with the evidence a card needs.
export function relevanceEvidenceFor(work, query, index) {
  const entry = index.byId.get(work.id) || { work, ...fieldTerms(work), counts: new Map(), titleCounts: new Map(), positions: new Map() };
  if (!entry.counts.size) {
    entry.counts = termCounts(entry.body);
    entry.titleCounts = termCounts(entry.title);
  }
  const expanded = withFeedback(index, expandQuery(query, index));
  const result = scoreEntry(entry, expanded, index);
  return {
    query,
    score: toDisplayScore(result.raw),
    feedbackTerms: expanded.feedbackTerms || [],
    matchedTerms: result.matched,
    coverage: Number((result.coverage || 0).toFixed(3)),
    proximity: Number((result.proximity || 0).toFixed(3)),
    expandedWith: expanded.expansions,
    topTerms: result.contributions,
  };
}

// The best score across every interest phrase the user has saved, which is how
// the feed is actually ordered when several interests are configured.
export function bestRelevance(work, queries, index) {
  let best = null;
  for (const query of queries) {
    const evidence = relevanceEvidenceFor(work, query, index);
    if (!best || evidence.score > best.score) best = evidence;
  }
  return best;
}
