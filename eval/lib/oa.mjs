// Rate-limited OpenAlex client for benchmark construction only. This file never
// ships in the extension; it exists to build the cached fixtures that the
// offline evaluation runs against.

const API_ROOT = "https://api.openalex.org";
export const EVAL_CONTACT = "filteredresearch-eval@example.com";

const MIN_REQUEST_GAP_MS = Number(process.env.FR_EVAL_GAP_MS || 220);
let last_request_at = 0;

async function pace() {
  const wait = MIN_REQUEST_GAP_MS - (Date.now() - last_request_at);
  if (wait > 0) await new Promise((resolve) => setTimeout(resolve, wait));
  last_request_at = Date.now();
}

// OpenAlex now meters the free tier: 1,000 requests (USD 0.10) per day, resetting
// at midnight UTC. Blowing through it returns 429 for the rest of the day, so the
// fetcher counts its own spend and stops cleanly rather than thrashing on retries.
export const BUDGET = { max: Number(process.env.FR_EVAL_MAX_REQUESTS || 900), spent: 0, exhausted: false };

export class BudgetExhausted extends Error {
  constructor() {
    super("OpenAlex free-tier budget for this run is spent");
    this.name = "BudgetExhausted";
  }
}

export async function oaGet(path, params = {}, attempt = 0) {
  if (attempt === 0) {
    if (BUDGET.spent >= BUDGET.max) {
      BUDGET.exhausted = true;
      throw new BudgetExhausted();
    }
    BUDGET.spent += 1;
  }
  await pace();
  const url = new URL(`${API_ROOT}/${path}`);
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined && value !== null && value !== "") url.searchParams.set(key, String(value));
  }
  url.searchParams.set("mailto", EVAL_CONTACT);
  let response;
  try {
    response = await fetch(url, { headers: { "User-Agent": `FilteredResearchEval/1.0 (${EVAL_CONTACT})` } });
  } catch (error) {
    if (attempt >= 4) throw error;
    await new Promise((resolve) => setTimeout(resolve, 600 * 2 ** attempt));
    return oaGet(path, params, attempt + 1);
  }
  if (response.status === 429) {
    // The daily allowance is gone. Retrying cannot help until midnight UTC, so
    // the run stops and keeps whatever it has already checkpointed.
    const remaining = response.headers.get("x-ratelimit-remaining");
    if (remaining === "0") {
      BUDGET.exhausted = true;
      throw new BudgetExhausted();
    }
  }
  if (response.status === 429 || response.status >= 500) {
    if (attempt >= 5) throw new Error(`OpenAlex ${response.status} for ${url}`);
    // A 429 means the whole client is going too fast, not just this request, so
    // the pacer is widened for everyone rather than only this retry sleeping.
    await new Promise((resolve) => setTimeout(resolve, Math.min(20_000, 1200 * 2 ** attempt)));
    return oaGet(path, params, attempt + 1);
  }
  if (!response.ok) throw new Error(`OpenAlex ${response.status} for ${url}`);
  return response.json();
}

export function shortId(value) {
  return String(value || "").replace(/\/$/, "").split("/").at(-1);
}

// Run `size` tasks with at most `width` in flight. Keeps the polite pool happy
// while still finishing ~800 requests in a couple of minutes.
export async function mapLimit(items, width, worker) {
  const results = new Array(items.length);
  let cursor = 0;
  const runners = Array.from({ length: Math.min(width, items.length) }, async () => {
    while (cursor < items.length) {
      const index = cursor;
      cursor += 1;
      results[index] = await worker(items[index], index);
    }
  });
  await Promise.all(runners);
  return results;
}

export function reconstructAbstract(index) {
  if (!index || typeof index !== "object") return "";
  let highest = -1;
  for (const positions of Object.values(index)) {
    if (!Array.isArray(positions)) continue;
    for (const position of positions) highest = Math.max(highest, Number(position));
  }
  if (highest < 0 || highest > 100_000) return "";
  const words = new Array(highest + 1).fill("");
  for (const [word, positions] of Object.entries(index)) {
    if (!Array.isArray(positions)) continue;
    for (const position of positions) if (position >= 0 && position < words.length) words[position] = word;
  }
  return words.join(" ").replace(/\s+/g, " ").trim();
}

// The benchmark stores a deliberately narrow projection of each record: enough
// to score it exactly as the extension would, plus the outcome fields the
// extension never sees.
export function projectWork(payload) {
  const primary = payload.primary_topic || {};
  return {
    id: shortId(payload.id),
    title: String(payload.title || "").slice(0, 600),
    abstract: reconstructAbstract(payload.abstract_inverted_index).slice(0, 6000),
    language: payload.language || null,
    publicationDate: payload.publication_date || null,
    workType: payload.type || "article",
    topicId: shortId(primary.id),
    topicName: primary.display_name || null,
    subfieldId: shortId(primary.subfield?.id),
    subfieldName: primary.subfield?.display_name || null,
    fieldId: shortId(primary.field?.id),
    fieldName: primary.field?.display_name || null,
    domainId: shortId(primary.domain?.id),
    topics: (payload.topics || []).slice(0, 8).map((topic) => ({
      topicId: shortId(topic.id),
      topicName: topic.display_name || null,
      score: Number(topic.score || 0),
      subfieldId: shortId(topic.subfield?.id),
      fieldId: shortId(topic.field?.id),
      domainId: shortId(topic.domain?.id),
    })),
    sourceId: shortId(payload.primary_location?.source?.id),
    sourceName: payload.primary_location?.source?.display_name || null,
    referencedWorks: (payload.referenced_works || []).map(shortId),
    authorships: (payload.authorships || []).slice(0, 25).map((authorship) => ({
      authorId: shortId(authorship.author?.id),
      name: authorship.author?.display_name || null,
      position: authorship.author_position || null,
      isCorresponding: Boolean(authorship.is_corresponding),
      institutions: (authorship.institutions || []).map((institution) => institution.display_name).filter(Boolean),
      rawAffiliations: (authorship.raw_affiliation_strings || []).slice(0, 3),
    })),
    // ---- outcome fields: labels only, never fed to any ranker ----
    outcome: {
      citedByCount: Number(payload.cited_by_count || 0),
      countsByYear: (payload.counts_by_year || []).map((entry) => ({
        year: Number(entry.year),
        cited: Number(entry.cited_by_count || 0),
      })),
    },
  };
}

export const WORK_SELECT = [
  "id", "doi", "title", "language", "publication_date", "type", "authorships",
  "abstract_inverted_index", "primary_topic", "topics", "primary_location",
  "referenced_works", "cited_by_count", "counts_by_year",
].join(",");
