const API_ROOT = "https://api.openalex.org";

// Contact address for the polite pool. OpenAlex asks tools to identify
// themselves so it can reach whoever runs one that misbehaves; it is a
// project alias, published deliberately, and carries no user data.
export const CONTACT_EMAIL = "shatteredstudious@gmail.com";

const WORK_FIELDS = [
  "id",
  "doi",
  "title",
  "language",
  "publication_date",
  "type",
  "authorships",
  "abstract_inverted_index",
  "primary_topic",
  "topics",
  "primary_location",
  "locations",
].join(",");

const NAMED_ENTITIES = Object.freeze({
  amp: "&",
  apos: "'",
  gt: ">",
  lt: "<",
  nbsp: " ",
  quot: '"',
});

function decodeHtmlEntitiesOnce(value) {
  return value.replace(/&(#x[0-9a-f]+|#\d+|[a-z]+);/gi, (match, entity) => {
    if (entity[0] !== "#") return NAMED_ENTITIES[entity.toLowerCase()] ?? match;
    const hexadecimal = entity[1]?.toLowerCase() === "x";
    const codePoint = Number.parseInt(entity.slice(hexadecimal ? 2 : 1), hexadecimal ? 16 : 10);
    if (!Number.isInteger(codePoint) || codePoint < 0 || codePoint > 0x10ffff) return "";
    try {
      return String.fromCodePoint(codePoint);
    } catch {
      return "";
    }
  });
}

export function cleanScholarlyText(value, { repairSpacing = true } = {}) {
  let text = String(value || "");
  for (let pass = 0; pass < 2; pass += 1) text = decodeHtmlEntitiesOnce(text);
  text = text
    .normalize("NFKC")
    .replace(/<\/?[a-z][^>]{0,500}>/gi, " ")
    .replace(/[\u200b-\u200d\ufeff]/g, "")
    .replace(/[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f]/g, " ");
  if (repairSpacing) {
    text = text
      .replace(
        /\b([A-Z][a-zÀ-öø-ÿ]{3,}|[a-zÀ-öø-ÿ]{4,})([A-Z][a-zÀ-öø-ÿ]{2,})\b/g,
        (match, left, right) =>
          new Set(["OpenAlex", "GitHub", "YouTube", "iPhone", "eBay", "arXiv", "LaTeX"]).has(
            match,
          )
            ? match
            : `${left} ${right}`,
      )
      .replace(/([,;:!?])(?=[A-Za-zÀ-öø-ÿ])/g, "$1 ");
  }
  return text.replace(/\s+/g, " ").trim();
}

const AUTHOR_FIELDS = [
  "id",
  "display_name",
  "orcid",
  "works_count",
  "cited_by_count",
  "summary_stats",
  "last_known_institutions",
].join(",");

export function shortOpenAlexId(value) {
  return String(value || "")
    .replace(/\/$/, "")
    .split("/")
    .at(-1);
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
    for (const position of positions) {
      if (position >= 0 && position < words.length) words[position] = word;
    }
  }
  return words.filter(Boolean).join(" ");
}

function topicParts(topic = {}) {
  return {
    topicId: shortOpenAlexId(topic.id) || null,
    topicName: cleanScholarlyText(topic.display_name, { repairSpacing: false }) || null,
    subfieldId: shortOpenAlexId(topic.subfield?.id) || null,
    subfieldName:
      cleanScholarlyText(topic.subfield?.display_name, { repairSpacing: false }) || null,
    fieldId: shortOpenAlexId(topic.field?.id) || null,
    fieldName: cleanScholarlyText(topic.field?.display_name, { repairSpacing: false }) || null,
    domainId: shortOpenAlexId(topic.domain?.id) || null,
    domainName: cleanScholarlyText(topic.domain?.display_name, { repairSpacing: false }) || null,
    score: Number(topic.score || 0),
  };
}

function extractArxivId(locations = []) {
  for (const location of locations) {
    const url = location?.landing_page_url || location?.pdf_url || "";
    const match = url.match(/arxiv\.org\/(?:abs|pdf)\/([^?#]+)/i);
    if (match) return match[1].replace(/\.pdf$/i, "").replace(/v\d+$/i, "");
  }
  return null;
}

export function normalizeWork(payload, { baseline = false, maxAuthors = 25 } = {}) {
  const primary = topicParts(payload.primary_topic || {});
  const locations = [payload.primary_location, ...(payload.locations || [])].filter(Boolean);
  const primaryLocation = payload.primary_location || {};
  const authorships = (payload.authorships || [])
    .slice(0, maxAuthors)
    .map((entry) => ({
      authorId: shortOpenAlexId(entry.author?.id),
      name:
        cleanScholarlyText(entry.author?.display_name || entry.raw_author_name, {
          repairSpacing: false,
        }) || "Unknown",
      orcid: entry.author?.orcid || null,
      position: entry.author_position || "middle",
      isCorresponding: Boolean(entry.is_corresponding),
      institutions: (entry.institutions || []).map((item) => cleanScholarlyText(item.display_name, { repairSpacing: false })).filter(Boolean),
      rawAffiliations: (entry.raw_affiliation_strings || []).map((item) => cleanScholarlyText(item, { repairSpacing: false })).filter(Boolean),
    }))
    .filter((entry) => entry.authorId);
  return {
    id: shortOpenAlexId(payload.id),
    title: cleanScholarlyText(payload.title || "Untitled"),
    abstract: cleanScholarlyText(reconstructAbstract(payload.abstract_inverted_index)).slice(
      0,
      12_000,
    ),
    language: String(payload.language || "").toLowerCase() || null,
    publicationDate: payload.publication_date || new Date().toISOString().slice(0, 10),
    doi: payload.doi || null,
    url: primaryLocation.landing_page_url || payload.doi || payload.id,
    sourceName:
      cleanScholarlyText(primaryLocation.source?.display_name, { repairSpacing: false }) || null,
    workType: payload.type || "article",
    arxivId: extractArxivId(locations),
    sources: [...new Map(locations.map((location) => ({
      name: cleanScholarlyText(location.source?.display_name, { repairSpacing: false }) || "Repository",
      url: location.landing_page_url || location.pdf_url || payload.doi || payload.id,
      doi: payload.doi || null,
    })).filter((source) => source.url).map((source) => [`${source.name}|${source.url}`, source])).values()],
    authorships,
    topics: (payload.topics || []).slice(0, 10).map(topicParts),
    ...primary,
    isBaseline: baseline,
    fetchedAt: new Date().toISOString(),
  };
}

export function normalizeAuthor(payload) {
  const stats = payload.summary_stats || {};
  return {
    id: shortOpenAlexId(payload.id),
    name: cleanScholarlyText(payload.display_name, { repairSpacing: false }) || "Unknown",
    orcid: payload.orcid || null,
    worksCount: Number(payload.works_count || 0),
    citedByCount: Number(payload.cited_by_count || 0),
    hIndex: Number(stats.h_index || 0),
    i10Index: Number(stats.i10_index || 0),
    twoYearMeanCitedness: Number(stats["2yr_mean_citedness"] || 0),
    lastInstitution:
      cleanScholarlyText(payload.last_known_institutions?.[0]?.display_name, {
        repairSpacing: false,
      }) || null,
    fetchedAt: new Date().toISOString(),
  };
}

export function cleanWorkForDisplay(work) {
  return {
    ...work,
    title: cleanScholarlyText(work.title || "Untitled"),
    abstract: cleanScholarlyText(work.abstract || ""),
    sourceName:
      cleanScholarlyText(work.sourceName, { repairSpacing: false }) || null,
    topicName: cleanScholarlyText(work.topicName, { repairSpacing: false }) || null,
    subfieldName: cleanScholarlyText(work.subfieldName, { repairSpacing: false }) || null,
    fieldName: cleanScholarlyText(work.fieldName, { repairSpacing: false }) || null,
    nearestTitle: cleanScholarlyText(work.nearestTitle) || null,
    authorships: (work.authorships || []).map((authorship) => ({
      ...authorship,
      name:
        cleanScholarlyText(authorship.name, { repairSpacing: false }) || "Unknown",
    })),
  };
}

export class OpenAlexClient {
  constructor({ apiKey = "", timeoutMs = 25_000, maxEstimatedCostUsd = Infinity, onUsage = null } = {}) {
    this.apiKey = apiKey;
    this.timeoutMs = timeoutMs;
    this.maxEstimatedCostUsd = maxEstimatedCostUsd;
    this.costUsd = 0;
    this.estimatedCostUsd = 0;
    this.onUsage = onUsage;
  }

  async request(endpoint, parameters) {
    const estimatedCost = parameters.search ? 0.001 : 0.0001;
    if (this.estimatedCostUsd + estimatedCost > this.maxEstimatedCostUsd) {
      const error = new Error(
        `The OpenAlex budget guard stopped this scan at $${this.estimatedCostUsd.toFixed(4)}.`,
      );
      error.name = "OpenAlexBudgetError";
      throw error;
    }
    this.estimatedCostUsd += estimatedCost;
    const url = new URL(`${API_ROOT}/${endpoint}`);
    for (const [key, value] of Object.entries(parameters)) {
      if (value !== undefined && value !== null && value !== "") {
        url.searchParams.set(key, String(value));
      }
    }
    // OpenAlex serves identified callers from its polite pool, which is faster
    // and throttled later than the anonymous pool. Users without their own key
    // are the ones who would feel that throttling first, so every request
    // identifies the extension. This is a contact address for the tool, not
    // anything about the user, and no user data is attached to it.
    url.searchParams.set("mailto", CONTACT_EMAIL);
    if (this.apiKey) url.searchParams.set("api_key", this.apiKey);

    for (let attempt = 0; attempt <= 4; attempt += 1) {
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), this.timeoutMs);
      try {
        const response = await fetch(url, {
          headers: { Accept: "application/json" },
          signal: controller.signal,
        });
        if (!response.ok) {
          if ((response.status === 429 || response.status >= 500) && attempt < 4) {
            const retryAfter = Number(response.headers.get("Retry-After"));
            await new Promise((resolve) =>
              setTimeout(resolve, Number.isFinite(retryAfter) ? retryAfter * 1000 : 2 ** attempt * 750),
            );
            continue;
          }
          throw new Error(`OpenAlex returned ${response.status}: ${(await response.text()).slice(0, 300)}`);
        }
        const payload = await response.json();
        const actualCost = Number(payload.meta?.cost_usd || 0);
        this.costUsd += actualCost;
        try {
          await this.onUsage?.({
            costUsd: actualCost || estimatedCost,
            estimated: !actualCost,
            remaining: Number(response.headers.get("X-RateLimit-Remaining")),
            limit: Number(response.headers.get("X-RateLimit-Limit")),
            resetSeconds: Number(response.headers.get("X-RateLimit-Reset")),
          });
        } catch (usageError) {
          console.warn("Could not record OpenAlex usage", usageError);
        }
        return payload;
      } catch (error) {
        if (attempt >= 4 || (error.name !== "AbortError" && !String(error).includes("fetch"))) {
          throw error;
        }
        await new Promise((resolve) => setTimeout(resolve, 2 ** attempt * 750));
      } finally {
        clearTimeout(timer);
      }
    }
    throw new Error("OpenAlex request failed after retries");
  }

  static buildFilter(
    since,
    until,
    { fieldIds = [], subfieldIds = [], englishOnly = false, requireAbstract = false } = {},
  ) {
    const filters = [
      `from_publication_date:${since}`,
      `to_publication_date:${until}`,
      "type:article|preprint",
    ];
    if (requireAbstract) filters.push("has_abstract:true");
    if (englishOnly) filters.push("language:en");
    if (fieldIds.length) filters.push(`primary_topic.field.id:${fieldIds.join("|")}`);
    if (subfieldIds.length) {
      filters.push(`primary_topic.subfield.id:${subfieldIds.join("|")}`);
    }
    return filters.join(",");
  }

  async fetchWorks({
    since,
    until,
    limit,
    query = "",
    seed,
    baseline = false,
    fieldIds = [],
    subfieldIds = [],
    englishOnly = false,
    requireAbstract = true,
  }) {
    if (limit <= 0) return [];
    const works = [];
    let page = 1;
    const pageSize = Math.min(100, limit);
    while (works.length < limit) {
      const parameters = {
        filter: OpenAlexClient.buildFilter(since, until, {
          fieldIds,
          subfieldIds,
          englishOnly,
          requireAbstract,
        }),
        per_page: pageSize,
        page,
        select: WORK_FIELDS,
      };
      if (query) {
        parameters.search = query;
        parameters.sort = "publication_date:desc,relevance_score:desc";
      } else {
        parameters.sample = limit;
        parameters.seed = seed;
      }
      const payload = await this.request("works", parameters);
      const results = payload.results || [];
      works.push(
        ...results.map((item) => normalizeWork(item, { baseline })).filter((work) => work.id),
      );
      if (results.length < pageSize) break;
      page += 1;
    }
    return works.slice(0, limit);
  }

  async fetchWorksCursor({
    since,
    until,
    limit = 50_000,
    query = "",
    baseline = false,
    fieldIds = [],
    subfieldIds = [],
    englishOnly = false,
    requireAbstract = false,
    onProgress = null,
  }) {
    const works = [];
    const seen = new Set();
    let cursor = "*";
    let pages = 0;
    let total = null;
    while (cursor && works.length < limit) {
      const parameters = {
        filter: OpenAlexClient.buildFilter(since, until, {
          fieldIds,
          subfieldIds,
          englishOnly,
          requireAbstract,
        }),
        per_page: 100,
        cursor,
        select: WORK_FIELDS,
      };
      if (query) parameters.search = query;
      const payload = await this.request("works", parameters);
      const results = payload.results || [];
      total ??= Number(payload.meta?.count || 0);
      pages += 1;
      for (const item of results) {
        const work = normalizeWork(item, { baseline });
        if (!work.id || seen.has(work.id)) continue;
        seen.add(work.id);
        works.push(work);
        if (works.length >= limit) break;
      }
      await onProgress?.({ fetched: works.length, total, pages });
      cursor = payload.meta?.next_cursor || null;
      if (!results.length) break;
    }
    return {
      works,
      total: total || works.length,
      pages,
      truncated: works.length < (total || works.length) && works.length >= limit,
    };
  }

  async fetchAuthors(authorIds, { onProgress = null } = {}) {
    const unique = [...new Set(authorIds.filter(Boolean))];
    const authors = [];
    for (let offset = 0; offset < unique.length; offset += 100) {
      const batch = unique.slice(offset, offset + 100);
      const payload = await this.request("authors", {
        filter: `openalex_id:${batch.join("|")}`,
        per_page: 100,
        select: AUTHOR_FIELDS,
      });
      authors.push(...(payload.results || []).map(normalizeAuthor));
      await onProgress?.({
        fetched: Math.min(offset + batch.length, unique.length),
        total: unique.length,
        pages: Math.floor(offset / 100) + 1,
      });
    }
    return authors.filter((author) => author.id);
  }

  async fetchTaxonomy() {
    const fieldsPayload = await this.request("fields", {
      per_page: 100,
      page: 1,
      select: "id,display_name,domain",
    });
    const subfields = [];
    for (let page = 1; page <= 3; page += 1) {
      const payload = await this.request("subfields", {
        per_page: 100,
        page,
        select: "id,display_name,field,domain",
      });
      subfields.push(...(payload.results || []));
      if ((payload.results || []).length < 100) break;
    }
    const fields = (fieldsPayload.results || [])
      .map((field) => ({
        id: shortOpenAlexId(field.id),
        name: cleanScholarlyText(field.display_name, { repairSpacing: false }),
        domainName: cleanScholarlyText(field.domain?.display_name, { repairSpacing: false }),
      }))
      .filter((field) => field.id && field.name);
    const byField = new Map(fields.map((field) => [field.id, { ...field, subfields: [] }]));
    for (const subfield of subfields) {
      const fieldId = shortOpenAlexId(subfield.field?.id);
      const parent = byField.get(fieldId);
      if (!parent) continue;
      parent.subfields.push({
        id: shortOpenAlexId(subfield.id),
        name: cleanScholarlyText(subfield.display_name, { repairSpacing: false }),
      });
    }
    return [...byField.values()]
      .map((field) => ({
        ...field,
        subfields: field.subfields.sort((left, right) => left.name.localeCompare(right.name)),
      }))
      .sort((left, right) => left.name.localeCompare(right.name));
  }

  async findWorkByTitle(title) {
    const payload = await this.request("works", {
      search: title,
      filter: "language:en",
      per_page: 5,
      select: WORK_FIELDS,
    });
    const target = normalizeText(title);
    const match = (payload.results || []).find((item) => normalizeText(item.title) === target);
    return match ? normalizeWork(match) : null;
  }

  async findWorksByTitles(titles = []) {
    const requested = [...new Set(titles.map((title) => cleanScholarlyText(title)).filter(Boolean))];
    const byTitle = new Map();
    for (let offset = 0; offset < requested.length; offset += 16) {
      const batch = requested.slice(offset, offset + 16);
      const search = batch.map((title) => `"${title.replace(/["\\]/g, " ")}"`).join(" OR ");
      const payload = await this.request("works", { search, per_page: 100, select: WORK_FIELDS });
      const wanted = new Set(batch.map(normalizeText));
      for (const item of payload.results || []) {
        const normalized = normalizeText(item.title);
        if (wanted.has(normalized) && !byTitle.has(normalized)) byTitle.set(normalized, normalizeWork(item));
      }
    }
    return requested.map((title) => byTitle.get(normalizeText(title))).filter(Boolean);
  }
}

function normalizeText(value) {
  return String(value || "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, " ")
    .trim();
}

// arXiv identifiers encode the month the preprint was first submitted: modern
// ids are YYMM.NNNNN and legacy ids are archive/YYMMNNN. OpenAlex stores a
// single publication_date per work, which for a preprint later picked up by a
// journal is the journal's date. Ranking on that date presented year-old
// research as brand new, so first release is derived separately.
export function arxivSubmissionMonth(arxivId) {
  const raw = String(arxivId || "").trim();
  const modern = raw.match(/^(\d{2})(\d{2})\.\d{4,5}/);
  const legacy = raw.match(/^[a-z-]+(?:\.[A-Za-z-]{2})?\/(\d{2})(\d{2})\d{3}/i);
  const match = modern || legacy;
  if (!match) return null;
  const month = Number(match[2]);
  if (month < 1 || month > 12) return null;
  const year = 1900 + Number(match[1]) + (Number(match[1]) < 90 ? 100 : 0);
  return `${year}-${String(month).padStart(2, "0")}-01`;
}

// Returns the earliest date the work was demonstrably public. When the preprint
// month precedes the recorded publication month the exact day is unavailable,
// but such a work is old enough that day precision no longer matters; when the
// two fall in the same month the recorded date is kept because it is precise.
export function firstReleaseDate(work = {}) {
  const published = String(work.publicationDate || "").slice(0, 10);
  const preprint = arxivSubmissionMonth(work.arxivId);
  if (!preprint) return published;
  if (!published) return preprint;
  return preprint < `${published.slice(0, 7)}-01` ? preprint : published;
}
