const LEGACY_CATEGORY_SELECTIONS = Object.freeze({
  ai: { subfieldIds: ["1702"] },
  "computer-science": { fieldIds: ["17"] },
  "math-stats": { fieldIds: ["18", "26"] }, physics: { fieldIds: ["31"] },
  "chem-materials": { fieldIds: ["15", "16", "25"] }, biology: { fieldIds: ["11", "13", "24", "34"] },
  "medicine-health": { fieldIds: ["27", "29", "30", "35", "36"] }, "psych-neuro": { fieldIds: ["28", "32"] },
  "earth-environment": { fieldIds: ["19", "23"] }, "engineering-energy": { fieldIds: ["21", "22"] },
  "social-sciences": { fieldIds: ["33"] }, "economics-business": { fieldIds: ["14", "20"] }, "arts-humanities": { fieldIds: ["12"] },
});

const AI_EVIDENCE = Object.freeze([
  /\bartificial intelligence\b/, /\bmachine learning\b/, /\bdeep learning\b/, /\breinforcement learning\b/,
  /\bneural (?:network|model|representation|architecture)/,
  /\blarge language model|\blanguage model(?:s|ing)?\b|\bllms?\b/,
  /\bgenerative ai\b|\bgenerative model/, /\bcomputer vision\b|\bimage (?:classification|segmentation|generation)/,
  /\bnatural language (?:processing|understanding|generation)/, /\bknowledge graph\b|\bexpert system\b/,
  /\bmulti[ -]?agent\b|\bautonomous agent\b|\bagent(?:ic)? (?:system|reasoning|planning)/,
  /\brobot(?:ics|ic| learning)\b/, /\btransformer(?:s| architecture)?\b|\bdiffusion model\b/,
  /\bgraph neural\b|\bfoundation model\b|\brepresentation learning\b/,
  /\b(?:few|zero)[ -]?shot\b|\bself[ -]?supervised\b|\btransfer learning\b/,
  /\bai[- ](?:based|driven|enabled|assisted|generated)\b/, /\b(?:algorithmic|automated) (?:reasoning|planning|decision making)\b/,
]);

const STOP_WORDS = new Set(["a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "is", "of", "on", "or", "the", "to", "via", "with", "without", "system", "systems", "method", "methods"]);

export function taxonomyId(value) { return String(value || "").match(/(\d+)\/?$/)?.[1] || ""; }
function normalizeIds(value) { return Array.isArray(value) ? [...new Set(value.map(taxonomyId).filter(Boolean))] : []; }
export function normalizeFieldIds(value) { return normalizeIds(value); }
export function normalizeSubfieldIds(value) { return normalizeIds(value); }

export function legacySelection(categoryIds) {
  const fieldIds = [], subfieldIds = [];
  for (const categoryId of Array.isArray(categoryIds) ? categoryIds : []) {
    const selection = LEGACY_CATEGORY_SELECTIONS[categoryId];
    fieldIds.push(...(selection?.fieldIds || [])); subfieldIds.push(...(selection?.subfieldIds || []));
  }
  const normalizedFields = normalizeFieldIds(fieldIds);
  return { fieldIds: normalizedFields, subfieldIds: normalizeSubfieldIds(subfieldIds).filter((id) => !(id === "1702" && normalizedFields.includes("17"))) };
}

export function selectionFromSettings(settings = {}) {
  if (!settings || typeof settings !== "object") settings = {};
  const fieldIds = normalizeFieldIds(settings.selectedFields), subfieldIds = normalizeSubfieldIds(settings.selectedSubfields);
  return fieldIds.length || subfieldIds.length ? { fieldIds, subfieldIds } : legacySelection(settings.selectedCategories);
}
export function fieldIdsForCategories(categoryIds) { return legacySelection(categoryIds).fieldIds; }

export function normalizeSearchText(value) {
  return String(value || "").normalize("NFKD").toLocaleLowerCase("en").replace(/[\u0300-\u036f]/g, "").replace(/[^a-z0-9]+/g, " ").trim();
}

function editDistanceWithin(left, right, maximum) {
  if (Math.abs(left.length - right.length) > maximum) return false;
  let previous = Array.from({ length: right.length + 1 }, (_, index) => index);
  for (let row = 1; row <= left.length; row += 1) {
    const current = [row];
    for (let column = 1; column <= right.length; column += 1) current.push(Math.min(previous[column] + 1, current[column - 1] + 1, previous[column - 1] + (left[row - 1] === right[column - 1] ? 0 : 1)));
    previous = current;
  }
  return previous[right.length] <= maximum;
}
function tokenMatches(queryToken, documentToken) {
  if (queryToken === documentToken) return true;
  // "rag" should match "rag-based" but never the inside of "storage".
  if (documentToken.includes("-") && documentToken.split("-").includes(queryToken)) return true;
  if (queryToken.length < 6 || queryToken[0] !== documentToken[0]) return false;
  return editDistanceWithin(queryToken, documentToken, queryToken.length >= 10 ? 2 : 1);
}

// Whole-token phrase match. A raw substring test made every short query match
// any word containing it: "RAG" matched storage, average, fragment, paragraph
// and diaphragm, and "AI" matched chain and plain.
function containsPhrase(documentTokens, queryTokens) {
  if (!queryTokens.length) return true;
  for (let start = 0; start + queryTokens.length <= documentTokens.length; start += 1) {
    let matched = true;
    for (let offset = 0; offset < queryTokens.length; offset += 1) {
      if (!tokenMatches(queryTokens[offset], documentTokens[start + offset])) {
        matched = false;
        break;
      }
    }
    if (matched) return true;
  }
  return false;
}
// An acronym and its expansion are the same subject. Searching "RAG" must find
// papers that only ever write "retrieval-augmented generation", and searching
// the full phrase must find papers that only use the acronym; otherwise a
// perfectly relevant literature is invisible to the query.
const STOP_INITIALS = new Set(["of", "the", "and", "for", "a", "an", "in", "on", "to", "with"]);

// Abbreviations researchers actually type. A known acronym resolves to its
// established meaning rather than to any phrase whose initials happen to line
// up, so "RAG" finds retrieval-augmented generation and not "robust adaptive
// gradient". Unknown abbreviations still fall back to initial matching.
const ACRONYM_GLOSSARY = Object.freeze({
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

function isAcronymCandidate(token) {
  return /^[a-z]{2,6}$/.test(token);
}

function glossaryExpansions(token) {
  const known = ACRONYM_GLOSSARY[token];
  return known ? known.map((phrase) => phrase.split(" ")) : null;
}

// The acronym a known expansion belongs to, so the full phrase also finds
// papers that only ever print the abbreviation.
function glossaryAcronymFor(queryTokens) {
  const phrase = queryTokens.join(" ");
  for (const [acronym, expansions] of Object.entries(ACRONYM_GLOSSARY)) {
    if (expansions.includes(phrase)) return acronym;
  }
  return "";
}

function expansionMatches(documentTokens, acronym) {
  const letters = acronym.split("");
  for (let start = 0; start + letters.length <= documentTokens.length; start += 1) {
    let offset = 0;
    let index = start;
    while (index < documentTokens.length && offset < letters.length) {
      const token = documentTokens[index];
      // Small joining words inside an expansion are skipped, but only between
      // matched initials, never at the start.
      if (offset > 0 && STOP_INITIALS.has(token)) {
        index += 1;
        continue;
      }
      const parts = token.includes("-") ? token.split("-") : [token];
      let consumed = false;
      for (const part of parts) {
        if (offset < letters.length && part.startsWith(letters[offset])) {
          offset += 1;
          consumed = true;
        }
      }
      if (!consumed) break;
      index += 1;
    }
    if (offset === letters.length) return true;
  }
  return false;
}

function acronymOf(tokens) {
  const initials = tokens.filter((token) => !STOP_INITIALS.has(token)).map((token) => token[0]);
  return initials.length >= 2 ? initials.join("") : "";
}

function matchWindow(text, query) {
  const documentTokens = normalizeSearchText(text).split(" ").filter(Boolean);
  const queryTokens = normalizeSearchText(query).split(" ").filter(Boolean);
  if (!queryTokens.length) return true;
  if (containsPhrase(documentTokens, queryTokens)) return true;
  if (queryTokens.length === 1 && isAcronymCandidate(queryTokens[0])) {
    const known = glossaryExpansions(queryTokens[0]);
    if (known) {
      // A recognised abbreviation means exactly what it is known to mean.
      if (known.some((expansion) => containsPhrase(documentTokens, expansion))) return true;
    } else if (expansionMatches(documentTokens, queryTokens[0])) {
      // Unrecognised abbreviation: fall back to matching its initials.
      return true;
    }
  }
  // A full phrase must also find papers that only ever print the abbreviation.
  if (queryTokens.length > 1) {
    const known = glossaryAcronymFor(queryTokens);
    if (known && documentTokens.includes(known)) return true;
    const derived = acronymOf(queryTokens);
    if (derived && documentTokens.includes(derived)) return true;
  }
  const meaningful = queryTokens.filter((token) => !STOP_WORDS.has(token));
  const required = meaningful.length ? meaningful : queryTokens;
  const width = Math.max(required.length + 5, 8);
  for (let start = 0; start < documentTokens.length; start += 1) {
    const window = documentTokens.slice(start, start + width);
    if (required.every((queryToken) => window.some((token) => tokenMatches(queryToken, token)))) return true;
  }
  return false;
}
function snippet(value, query) {
  const text = String(value || "").replace(/\s+/g, " ").trim();
  if (!text) return "";
  const needle = normalizeSearchText(query).split(" ").find((token) => token.length >= 4);
  const normalized = normalizeSearchText(text); const position = needle ? normalized.indexOf(needle) : 0;
  const start = Math.max(0, position - 70), end = Math.min(text.length, start + 190);
  return `${start ? "…" : ""}${text.slice(start, end).trim()}${end < text.length ? "…" : ""}`;
}

// Records come from storage and from a remote API, so a single malformed one
// must not throw out of a filter and take down an entire feed render.
export function interestMatchEvidence(work, queries = []) {
  if (!work || typeof work !== "object") return null;
  const list = Array.isArray(queries) ? queries : [queries];
  for (const query of list.filter((item) => item != null).map(String).map((item) => item.trim()).filter(Boolean)) {
    if (matchWindow(work.title, query)) return { query, location: "title", snippet: snippet(work.title, query) };
    if (matchWindow(work.abstract, query)) return { query, location: "abstract", snippet: snippet(work.abstract, query) };
    const labels = [work.topicName, work.subfieldName, work.fieldName, ...(work.topics || []).map((topic) => topic.topicName)]
      .filter(Boolean)
      .join(" ");
    if (labels && matchWindow(labels, query)) return { query, location: "topic", snippet: labels.slice(0, 190) };
  }
  return null;
}
export function matchesInterestQuery(work, query) { return Boolean(interestMatchEvidence(work, [query])); }
export function hasCategoryEvidence(work) {
  if (!work || typeof work !== "object") return false;
  if (taxonomyId(work.subfieldId) !== "1702") return true;
  const text = normalizeSearchText(`${work.title || ""} ${work.abstract || ""}`);
  return AI_EVIDENCE.some((pattern) => pattern.test(text));
}
export function matchesResearchFilters(work, settings = {}) {
  if (!work || typeof work !== "object") return false;
  if (!settings || typeof settings !== "object") settings = {};
  if (settings.englishOnly && work.language !== "en") return false;
  const selection = selectionFromSettings(settings);
  const arxivCategories = Array.isArray(work.arxivCategories) ? work.arxivCategories : [];
  const selectedArxivCategories = Array.isArray(settings.selectedArxivCategories) ? settings.selectedArxivCategories : [];
  const selectedArxivGroups = Array.isArray(settings.selectedArxivGroups) ? settings.selectedArxivGroups : [];
  const exactArxivScope = arxivCategories.length > 0 && (selectedArxivCategories.length > 0 || selectedArxivGroups.length > 0);
  if (arxivCategories.length && selectedArxivCategories.length && !arxivCategories.some((code) => selectedArxivCategories.includes(code))) return false;
  if (arxivCategories.length && selectedArxivGroups.length && !arxivCategories.some((code) => selectedArxivGroups.some((group) => code === group || code.startsWith(`${group}.`) || (group === "physics" && !/^(cs|econ|eess|math|q-bio|q-fin|stat)\./.test(code))))) return false;
  if (!exactArxivScope && (selection.fieldIds.length || selection.subfieldIds.length) && !selection.fieldIds.includes(taxonomyId(work.fieldId)) && !selection.subfieldIds.includes(taxonomyId(work.subfieldId))) return false;
  if (!hasCategoryEvidence(work)) return false;
  const queries = Array.isArray(settings.queries) ? settings.queries.map(String).map((query) => query.trim()).filter(Boolean) : [];
  if (!queries.length) return true;
  // With a category chosen, that category is the filter and interests only
  // steer ranking. Strict mode restores the old exclude-everything-else rule.
  const selection2 = selectionFromSettings(settings);
  const hasScope = selection2.fieldIds.length || selection2.subfieldIds.length ||
    (settings.selectedArxivGroups || []).length || (settings.selectedArxivCategories || []).length;
  if (hasScope && !settings.strictInterestFilter) return true;
  return Boolean(interestMatchEvidence(work, queries));
}
export function discoveryScopeSignature(settings = {}) {
  if (!settings || typeof settings !== "object") settings = {};
  const selection = selectionFromSettings(settings);
  return JSON.stringify({ fields: [...selection.fieldIds].sort(), subfields: [...selection.subfieldIds].sort(), englishOnly: Boolean(settings.englishOnly) });
}
export function researchFilterSignature(settings = {}) {
  if (!settings || typeof settings !== "object") settings = {};
  return JSON.stringify({ scope: JSON.parse(discoveryScopeSignature(settings)), arxivGroups: [...(settings.selectedArxivGroups || [])].sort(), arxivCategories: [...(settings.selectedArxivCategories || [])].sort(), queries: (settings.queries || []).map(normalizeSearchText).filter(Boolean).sort() });
}
