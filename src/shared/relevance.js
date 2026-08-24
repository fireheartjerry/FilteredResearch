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

const TITLE_WEIGHT = 2.6;
const K1 = 1.4;
const B = 0.72;
const EXPANSION_WEIGHT = 0.45;
const PROXIMITY_BONUS = 0.35;

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

export function buildRelevanceIndex(works, { space = null } = {}) {
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
  return { entries, lexicon, averageLength, space, byId: new Map(entries.map((entry) => [entry.work.id, entry])) };
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

function bm25For(entry, term, lexicon, averageLength) {
  const frequency = entry.counts.get(term) || 0;
  if (!frequency) return 0;
  const idf = lexicon.idf.get(term);
  if (!idf) return 0;
  const titleBoost = 1 + TITLE_WEIGHT * ((entry.titleCounts.get(term) || 0) > 0 ? 1 : 0);
  const length = Math.max(1, entry.body.length);
  const saturated = (frequency * (K1 + 1)) / (frequency + K1 * (1 - B + (B * length) / Math.max(1, averageLength)));
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

export function scoreEntry(entry, expanded, index) {
  let raw = 0;
  let matched = 0;
  const contributions = [];
  for (const [term, weight] of expanded.terms) {
    const value = bm25For(entry, term, index.lexicon, index.averageLength) * weight;
    if (value > 0) {
      raw += value;
      matched += 1;
      if (contributions.length < 6) contributions.push({ term, points: Number(value.toFixed(2)) });
    }
  }
  if (raw <= 0) return { raw: 0, score: 0, matched: 0, contributions: [] };
  const proximity = proximityScore(entry, expanded.phrases);
  raw *= 1 + PROXIMITY_BONUS * proximity;
  // Coverage matters as much as intensity: a paper that hits every query term
  // once is more relevant than one that hits a single term repeatedly.
  const coverage = matched / Math.max(1, expanded.terms.size);
  raw *= 0.55 + 0.45 * Math.sqrt(coverage);
  return { raw, matched, proximity, coverage, contributions };
}

// BM25 has no natural ceiling, so the displayed 0-100 comes from a saturating
// curve rather than from dividing by the corpus maximum -- otherwise one
// unusually strong match would compress every other result toward zero.
export function toDisplayScore(raw) {
  if (!(raw > 0)) return 0;
  return Math.round(100 * (1 - Math.exp(-raw / 9)) * 100) / 100;
}

export function rankByRelevance(works, query, options = {}) {
  const index = options.index || buildRelevanceIndex(works, { space: options.space || null });
  const expanded = expandQuery(query, index);
  if (!expanded.terms.size) return [...works];
  const scored = index.entries.map((entry) => ({ entry, ...scoreEntry(entry, expanded, index) }));
  scored.sort((left, right) => right.raw - left.raw || String(left.entry.work.id).localeCompare(String(right.entry.work.id)));
  return scored.map((item) => item.entry.work);
}

// Relevance for one paper against one query, with the evidence a card needs.
export function relevanceEvidenceFor(work, query, index) {
  const entry = index.byId.get(work.id) || { work, ...fieldTerms(work), counts: new Map(), titleCounts: new Map(), positions: new Map() };
  if (!entry.counts.size) {
    entry.counts = termCounts(entry.body);
    entry.titleCounts = termCounts(entry.title);
  }
  const expanded = expandQuery(query, index);
  const result = scoreEntry(entry, expanded, index);
  return {
    query,
    score: toDisplayScore(result.raw),
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
