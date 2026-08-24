import {
  WINDOWS,
  autoScanDue,
  effectiveHorizonDays,
  loadSettings,
  normalizeAutoScanHours,
  normalizeSettings,
  windowsWithin,
} from "../shared/defaults.js";
import {
  bulkPut,
  clearDatabase,
  databaseStats,
  deleteBaselineWorks,
  feedStats,
  getAll,
  getById,
  getMetadata,
  getMetadataMany,
  pruneCandidates,
  setMetadata,
} from "../shared/db.js";
import {
  matchesResearchFilters,
  discoveryScopeSignature,
  interestMatchEvidence,
  buildRelevanceIndex,
  bestRelevance,
  researchFilterSignature,
  selectionFromSettings,
} from "../shared/filters.js";
import { rankByRelevanceWithModel } from "../shared/relevance.js";
import { OpenAlexClient, cleanWorkForDisplay, firstReleaseDate } from "../shared/openalex.js";
import { applySelectivity } from "../shared/ranking.js";
import { groupDuplicatePapers } from "../shared/papers.js";
import { annotateProminence, buildProminentResearcherRoster } from "../shared/prominence.js";
import { SCORING_VERSION, scoreBatch } from "../shared/scoring.js";

const FALLBACK_TAXONOMY = Object.freeze([
  {
    id: "17",
    name: "Computer Science",
    domainName: "Physical Sciences",
    subfields: [
      ["1702", "Artificial Intelligence"],
      ["1703", "Computational Theory and Mathematics"],
      ["1704", "Computer Graphics and Computer-Aided Design"],
      ["1705", "Computer Networks and Communications"],
      ["1706", "Computer Science Applications"],
      ["1707", "Computer Vision and Pattern Recognition"],
      ["1708", "Hardware and Architecture"],
      ["1709", "Human-Computer Interaction"],
      ["1710", "Information Systems"],
      ["1711", "Signal Processing"],
      ["1712", "Software"],
    ].map(([id, name]) => ({ id, name })),
  },
]);

const ARXIV_GROUP_FIELDS = Object.freeze({
  cs: ["17"], econ: ["20"], eess: ["21", "22"], math: ["26"], physics: ["31"],
  "q-bio": ["11", "13", "24", "34"], "q-fin": ["14", "20"], stat: ["18"],
});
const ARXIV_CS_SUBFIELDS = Object.freeze({
  "cs.AI":["1702"],"cs.AR":["1708"],"cs.CC":["1703"],"cs.CE":["1706"],"cs.CG":["1704"],"cs.CL":["1702"],"cs.CR":["1710"],"cs.CV":["1707"],"cs.CY":["1709"],"cs.DB":["1710"],"cs.DC":["1705"],"cs.DL":["1710"],"cs.DM":["1703"],"cs.DS":["1703"],"cs.ET":["1708"],"cs.FL":["1703"],"cs.GL":["17"],"cs.GR":["1704"],"cs.GT":["1703"],"cs.HC":["1709"],"cs.IR":["1710"],"cs.IT":["1711"],"cs.LG":["1702"],"cs.LO":["1703"],"cs.MA":["1702"],"cs.MM":["1704"],"cs.MS":["1706"],"cs.NA":["1706"],"cs.NE":["1702"],"cs.NI":["1705"],"cs.OH":["17"],"cs.OS":["1712"],"cs.PF":["1705"],"cs.PL":["1712"],"cs.RO":["1702"],"cs.SC":["1706"],"cs.SD":["1711"],"cs.SE":["1712"],"cs.SI":["1710"],"cs.SY":["1706"],
});

let refreshPromise = null;
let pendingRefreshReason = null;
let usageAccumulator = null;
let usageFlushTimer = null;
let feedCorpusCache = null;

function isoDate(date = new Date()) {
  return date.toISOString().slice(0, 10);
}

function daysAgo(days, from = new Date()) {
  const value = new Date(from);
  value.setUTCDate(value.getUTCDate() - days);
  return isoDate(value);
}

function yearsAgo(years, fromIso) {
  const value = new Date(`${fromIso}T00:00:00Z`);
  value.setUTCFullYear(value.getUTCFullYear() - years);
  return isoDate(value);
}

// Recency everywhere in this worker means first public release, not the date a
// journal happened to re-publish an existing preprint.
function releasedOn(work) {
  return work.firstReleaseDate || firstReleaseDate(work);
}

function deduplicate(works) {
  return [...new Map(works.map((work) => [work.id, work])).values()];
}

// The filtered corpus is shared by the feed and the notification pass, and its
// cache key cannot see writes that happen before `lastRefresh` is stamped, so
// every write to the works store drops it explicitly.
async function storeWorks(works) {
  if (!works.length) return;
  await bulkPut("works", works);
  feedCorpusCache = null;
}

async function recordApiUsage(event) {
  const date = isoDate();
  const current = usageAccumulator || (await getMetadata("apiUsageDaily")) || {};
  const base = current.date === date ? current : { date, requests: 0, costUsd: 0 };
  usageAccumulator = {
    ...base, requests: base.requests + 1, costUsd: base.costUsd + Number(event.costUsd || 0),
    hasEstimatedCalls: Boolean(base.hasEstimatedCalls || event.estimated),
    providerRemaining: Number.isFinite(event.remaining) ? event.remaining : base.providerRemaining,
    providerLimit: Number.isFinite(event.limit) ? event.limit : base.providerLimit,
    resetAt: Number.isFinite(event.resetSeconds) ? new Date(Date.now() + event.resetSeconds * 1000).toISOString() : base.resetAt,
    updatedAt: new Date().toISOString(),
  };
  clearTimeout(usageFlushTimer);
  usageFlushTimer = setTimeout(() => {
    setMetadata("apiUsageDaily", usageAccumulator).catch(console.error);
  }, 500);
}

function openAlexClient(options) { return new OpenAlexClient({ ...options, onUsage: recordApiUsage }); }

function deduplicateAuthors(authors) {
  return [...new Map(authors.map((author) => [author.id, author])).values()];
}

async function getProminenceRoster(authors = null) {
  if (authors) {
    const roster = buildProminentResearcherRoster(authors);
    await setMetadata("prominentResearcherRoster", { version: 2, researchers: roster });
    return roster;
  }
  const cached = await getMetadata("prominentResearcherRoster");
  if (cached?.version === 2 && Array.isArray(cached.researchers)) return cached.researchers;
  return getProminenceRoster(await getAll("authors"));
}

function migratedCoverage(coverage) {
  if (!coverage?.fullCompletedAt || Number(coverage.horizonDays) > 0) return coverage;
  // v0.4 always built a one-year index but did not persist the horizon. Preserve
  // that expensive work instead of interpreting the missing field as zero days.
  return { ...coverage, horizonDays: 365, migratedLegacyHorizon: true };
}

function chooseAuthorIds(works, limit) {
  const priority = [];
  const remainder = [];
  for (const work of works) {
    for (const authorship of work.authorships || []) {
      if (
        authorship.isCorresponding ||
        authorship.position === "first" ||
        authorship.position === "last"
      ) {
        priority.push(authorship.authorId);
      } else {
        remainder.push(authorship.authorId);
      }
    }
  }
  return [...new Set([...priority, ...remainder])]
    .filter((id) => /^A\d+$/.test(String(id || "")))
    .slice(0, limit);
}

async function ensureDefaults() {
  if (chrome.storage.local.setAccessLevel) {
    await chrome.storage.local.setAccessLevel({ accessLevel: "TRUSTED_CONTEXTS" });
  }
  const stored = await chrome.storage.sync.get("settings");
  const normalized = normalizeSettings(stored.settings || {});
  if (JSON.stringify(stored.settings || null) !== JSON.stringify(normalized)) {
    await chrome.storage.sync.set({ settings: normalized });
  }
}

async function getOpenAlexTaxonomy({ force = false } = {}) {
  const cached = await getMetadata("openAlexTaxonomy");
  const fresh = Date.parse(cached?.fetchedAt || 0) >= Date.now() - 30 * 24 * 60 * 60 * 1000;
  if (!force && fresh && cached?.fields?.length) return cached.fields;
  try {
    const settings = await loadSettings();
    const client = openAlexClient({
      apiKey: settings.apiKey,
      maxEstimatedCostUsd: 0.005,
    });
    const fields = await client.fetchTaxonomy();
    if (fields.length) {
      await setMetadata("openAlexTaxonomy", { fields, fetchedAt: new Date().toISOString() });
      return fields;
    }
  } catch (error) {
    console.warn("Taxonomy refresh failed", error);
  }
  return cached?.fields?.length ? cached.fields : FALLBACK_TAXONOMY;
}

function parseArxivTaxonomy(html) {
  const headingPattern = /<h2 class="accordion-head"[^>]*id="accordion-head-grp_([^"]+)"[\s\S]*?<button[^>]*>[\s\S]*?<span[^>]*>[\s\S]*?<\/span>\s*([^<]+?)\s*<\/button>[\s\S]*?<\/h2>/gi;
  const headings = [...html.matchAll(headingPattern)];
  return headings.map((heading, index) => {
    const groupId = heading[1];
    const start = heading.index + heading[0].length;
    const end = headings[index + 1]?.index || html.length;
    const body = html.slice(start, end);
    const categories = [...body.matchAll(/<h4>([^ <]+)\s*<span>\(([^<]+)\)<\/span><\/h4>/gi)].map((match) => {
      const code = match[1].trim();
      const mapped = ARXIV_CS_SUBFIELDS[code] || [];
      const mappedFields = mapped.filter((id) => id.length <= 2);
      const mappedSubfields = mapped.filter((id) => id.length > 2);
      return { id: code, name: match[2].trim(), arxivCode: code,
        openAlexFieldIds: mappedFields.length ? mappedFields : mappedSubfields.length ? [] : ARXIV_GROUP_FIELDS[groupId] || [],
        openAlexSubfieldIds: mappedSubfields };
    });
    return { id: groupId, name: heading[2].trim(), kind: "arxiv", domainName: "arXiv", openAlexFieldIds: ARXIV_GROUP_FIELDS[groupId] || [], subfields: categories };
  }).filter((group) => group.subfields.length);
}

async function getCategoryTaxonomy(options = {}) {
  const [arxiv, openAlex] = await Promise.all([getArxivTaxonomy(options), getOpenAlexTaxonomy(options)]);
  const cleanArxiv = arxiv.map((field) => ({ ...field, kind: "arxiv", subfields: (field.subfields || []).map((subfield) => ({ ...subfield, name: String(subfield.name || subfield.id).replace(/^\S+\s+[—-]\s+/, "") })) }));
  const arxivFieldIds = new Set(Object.values(ARXIV_GROUP_FIELDS).flat());
  const general = openAlex.filter((field) => !arxivFieldIds.has(String(field.id))).map((field) => ({
    id: `oa:${field.id}`, name: field.name, kind: "openalex", domainName: field.domainName,
    openAlexFieldIds: [String(field.id)],
    subfields: (field.subfields || []).map((subfield) => ({ id: `oa:${subfield.id}`, name: subfield.name, openAlexFieldIds: [], openAlexSubfieldIds: [String(subfield.id)] })),
  }));
  return [...cleanArxiv, ...general];
}

async function getArxivTaxonomy({ force = false } = {}) {
  const cached = await getMetadata("arxivTaxonomy");
  const fresh = Date.parse(cached?.fetchedAt || 0) >= Date.now() - 30 * 24 * 60 * 60 * 1000;
  if (!force && fresh && cached?.fields?.length) return cached.fields;
  try {
    const response = await fetch("https://arxiv.org/category_taxonomy", { headers: { Accept: "text/html" } });
    if (!response.ok) throw new Error(`arXiv taxonomy returned ${response.status}`);
    const fields = parseArxivTaxonomy(await response.text());
    if (fields.length === 8 && fields.reduce((sum, field) => sum + field.subfields.length, 0) >= 150) {
      await setMetadata("arxivTaxonomy", { fields, fetchedAt: new Date().toISOString(), source: "https://arxiv.org/category_taxonomy" });
      return fields;
    }
    throw new Error("arXiv taxonomy response was incomplete");
  } catch (error) {
    console.warn("arXiv taxonomy refresh failed", error);
    if (cached?.fields?.length) return cached.fields;
    return [{ id: "cs", name: "Computer Science", kind: "arxiv", domainName: "arXiv", openAlexFieldIds: ["17"], subfields: Object.entries(ARXIV_CS_SUBFIELDS).map(([id, mapped]) => ({ id, name: id, arxivCode: id, openAlexFieldIds: [], openAlexSubfieldIds: mapped.filter((value) => value.length > 2) })) }];
  }
}

function expandedSubfieldIds(selection, taxonomy) {
  const result = new Set(selection.subfieldIds);
  for (const fieldId of selection.fieldIds) {
    const field = taxonomy.find((item) => item.id === fieldId);
    for (const subfield of field?.subfields || []) result.add(subfield.id);
  }
  return [...result];
}

function progressWriter(baseState) {
  let lastWrite = 0;
  return async (progress) => {
    const now = Date.now();
    if (now - lastWrite < 750 && progress.fetched < progress.total) return;
    lastWrite = now;
    await setMetadata("refreshState", {
      ...baseState,
      status: "running",
      ...progress,
      updatedAt: new Date().toISOString(),
    });
  };
}

async function fetchDiscovery(client, settings, taxonomy, since, until, mode, writeProgress) {
  const selection = selectionFromSettings(settings);
  const hasTaxonomySelection = selection.fieldIds.length || selection.subfieldIds.length;
  const laneLimit = mode === "limited" ? settings.broadSample : settings.maxDiscoveryWorks;
  const lanes = [];
  // A chosen category is indexed exhaustively, with no `search` term attached.
  // Attaching one made discovery itself keyword-scoped, so the local index only
  // ever held papers repeating the interest phrase and everything else in the
  // subfield was never retrieved at all.
  if (selection.fieldIds.length) {
    lanes.push({ label: "selected fields", fieldIds: selection.fieldIds, subfieldIds: [], query: "" });
  }
  if (selection.subfieldIds.length) {
    lanes.push({ label: "selected subfields", fieldIds: [], subfieldIds: selection.subfieldIds, query: "" });
  }
  if (!hasTaxonomySelection && settings.queries.length) {
    for (const query of settings.queries) {
      lanes.push({ label: query, fieldIds: [], subfieldIds: [], query });
    }
  }

  if (!lanes.length && !settings.queries.length) {
    const works = await client.fetchWorks({
      since,
      until,
      limit: settings.broadSample,
      seed: Number(isoDate().replaceAll("-", "")),
      englishOnly: settings.englishOnly,
      requireAbstract: false,
    });
    await writeProgress({
      phase: "discovery",
      lane: "cross-disciplinary preview",
      fetched: works.length,
      total: works.length,
      pages: Math.ceil(works.length / 100),
    });
    return {
      works,
      total: works.length,
      retrieved: works.length,
      pages: Math.ceil(works.length / 100),
      truncated: true,
      mode: "preview",
    };
  }

  const allWorks = [];
  let total = 0;
  let pages = 0;
  let truncated = false;
  for (const lane of lanes) {
    const result = await client.fetchWorksCursor({
      since,
      until,
      limit: lane.query ? Math.min(settings.perQuery, laneLimit) : laneLimit,
      query: lane.query || "",
      fieldIds: lane.fieldIds,
      subfieldIds: lane.subfieldIds,
      englishOnly: settings.englishOnly,
      requireAbstract: false,
      onProgress: (progress) =>
        writeProgress({ phase: "discovery", lane: lane.label, ...progress }),
    });
    allWorks.push(...result.works);
    total += result.total;
    pages += result.pages;
    truncated ||= result.truncated;
  }
  const uniqueRecords = deduplicate(allWorks);
  const retrieved = groupDuplicatePapers(uniqueRecords);
  const matched = retrieved.filter((work) => matchesResearchFilters(work, settings));
  return {
    works: matched,
    total,
    records: uniqueRecords.length,
    retrieved: retrieved.length,
    pages,
    truncated,
    mode,
    taxonomySubfields: expandedSubfieldIds(selection, taxonomy).length,
  };
}

async function maybeFetchBaseline(client, settings, taxonomy, candidateSince, writeProgress) {
  const signature = `${discoveryScopeSignature(settings)}:${SCORING_VERSION}`;
  const previousSignature = await getMetadata("baselineSignature");
  if (previousSignature !== signature) await deleteBaselineWorks();

  const selection = selectionFromSettings(settings);
  const targetSubfields = expandedSubfieldIds(selection, taxonomy);
  const existing = (await getAll("works")).filter((work) => work.isBaseline);
  const minimumExpected = targetSubfields.length
    ? targetSubfields.length * Math.min(100, settings.baselinePerSubfield)
    : Math.min(100, settings.baselinePerSubfield);
  if (previousSignature === signature && existing.length >= minimumExpected) return [];

  const until = daysAgo(1, new Date(`${candidateSince}T00:00:00Z`));
  const since = yearsAgo(settings.historyYears, until);
  const baseline = [];
  if (targetSubfields.length) {
    for (let index = 0; index < targetSubfields.length; index += 1) {
      const subfieldId = targetSubfields[index];
      const works = await client.fetchWorks({
        since,
        until,
        limit: settings.baselinePerSubfield,
        seed: 3103 + Number(subfieldId),
        baseline: true,
        subfieldIds: [subfieldId],
        englishOnly: settings.englishOnly,
        requireAbstract: true,
      });
      baseline.push(...works);
      await writeProgress({
        phase: "reference corpus",
        lane: `subfield ${index + 1} of ${targetSubfields.length}`,
        fetched: baseline.length,
        total: targetSubfields.length * settings.baselinePerSubfield,
        pages: Math.ceil(baseline.length / 100),
      });
    }
  } else {
    baseline.push(
      ...(await client.fetchWorks({
        since,
        until,
        limit: settings.baselinePerSubfield,
        seed: 3103,
        baseline: true,
        englishOnly: settings.englishOnly,
        requireAbstract: true,
      })),
    );
  }
  const result = deduplicate(baseline).map((work) => ({ ...work, isBaseline: true }));
  await setMetadata("baselineSignature", signature);
  return result;
}

function selectReferences(works, limit) {
  const baseline = works.filter((work) => work.isBaseline);
  const recent = works
    .filter((work) => !work.isBaseline)
    .sort((left, right) => right.publicationDate.localeCompare(left.publicationDate));
  return [...baseline, ...recent].slice(0, limit);
}

function safePaperUrl(work) {
  try {
    const url = new URL(work.url || work.doi || `https://openalex.org/${work.id}`);
    if (["https:", "http:"].includes(url.protocol)) return url.href;
  } catch {
    // Use OpenAlex as a known-safe fallback.
  }
  return `https://openalex.org/${work.id}`;
}

// Grouping, prominence annotation and filter matching over the whole works
// store is the single most expensive thing this worker does. Every caller now
// shares one cached result instead of repeating the scan, and the cache is
// invalidated by the same key the feed uses.
async function filteredCorpus(settings, { lastRefresh = null } = {}) {
  const stamp = lastRefresh ?? (await getMetadata("lastRefresh"));
  const cacheKey = `${stamp || "none"}:${researchFilterSignature(settings)}`;
  if (feedCorpusCache?.key !== cacheKey) {
    const roster = await getProminenceRoster();
    const stored = (await getAll("works")).filter((work) => work && typeof work === "object");
    // One unreadable record must not abort the whole render, so each is grouped
    // and screened defensively and simply dropped if it cannot be processed.
    let skipped = 0;
    const works = [];
    for (const work of groupDuplicatePapers(stored)) {
      try {
        const annotated = annotateProminence(work, roster);
        if (!annotated.isBaseline && annotated.scoringVersion && matchesResearchFilters(annotated, settings)) {
          works.push(annotated);
        }
      } catch (error) {
        skipped += 1;
        if (skipped <= 3) console.warn("Skipped an unreadable stored paper", work?.id, error);
      }
    }
    if (skipped) console.warn(`Skipped ${skipped} unreadable stored papers while building the feed`);
    feedCorpusCache = { key: cacheKey, works };
  }
  return feedCorpusCache.works;
}

// `days: null` means "everything in the local index". Notifications still want
// the 30-day view, but on-page highlighting must be able to match the older
// papers that dominate search-result pages.
async function qualifiedPapers(settings, { days = 30 } = {}) {
  const corpus = await filteredCorpus(settings);
  const cutoff = days === null ? null : daysAgo(days);
  const relevant = cutoff ? corpus.filter((work) => releasedOn(work) >= cutoff) : corpus;
  return applySelectivity(relevant, settings);
}

async function qualifiedMonth(settings, allWorks = null) {
  if (allWorks) {
    const roster = await getProminenceRoster();
    const relevant = groupDuplicatePapers(allWorks)
      .map((work) => annotateProminence(work, roster))
      .filter(
        (work) =>
          !work.isBaseline &&
          work.scoringVersion &&
          releasedOn(work) >= daysAgo(30) &&
          matchesResearchFilters(work, settings),
      );
    return applySelectivity(relevant, settings);
  }
  return qualifiedPapers(settings, { days: 30 });
}

async function recordNewPaperNotifications(
  scored,
  settings,
  existingCandidateIds,
  previousLastRefresh,
  reason,
) {
  if (reason === "install") return 0;
  const qualified = await qualifiedMonth(settings);
  const qualifiedIds = new Set(qualified.works.map((work) => work.id));
  // The very first pass has no previous run to diff against. Seeding the inbox
  // with the strongest recent results makes the feature reachable instead of
  // leaving a new user with a permanently empty page.
  const seeding = !previousLastRefresh;
  const previousDate = previousLastRefresh ? previousLastRefresh.slice(0, 10) : null;
  const newWorks = seeding
    ? qualified.works
        .slice()
        .sort((left, right) => Number(right.discoveryScore || 0) - Number(left.discoveryScore || 0))
        .slice(0, 10)
    : scored.filter(
        (work) =>
          !existingCandidateIds.has(work.id) &&
          releasedOn(work) >= previousDate &&
          qualifiedIds.has(work.id),
      );
  if (!newWorks.length) return 0;

  const oldInbox = (await getMetadata("notificationInbox")) || [];
  const known = new Set(oldInbox.map((item) => item.workId));
  const createdAt = new Date().toISOString();
  const entries = newWorks
    .filter((work) => !known.has(work.id))
    .map((work) => ({
      id: `${work.id}:${createdAt}`,
      workId: work.id,
      title: cleanWorkForDisplay(work).title,
      url: safePaperUrl(work),
      publicationDate: releasedOn(work) || work.publicationDate,
      topic: work.subfieldName || work.fieldName || "Research",
      noveltyScore: Math.round(work.noveltyScore || 0),
      researcherScore: Math.round(work.researcherScore || 0),
      createdAt,
      unread: true,
    }));
  if (!entries.length) return 0;
  await setMetadata("notificationInbox", [...entries, ...oldInbox].slice(0, 250));

  const notificationsAllowed =
    settings.notificationsEnabled &&
    (await chrome.permissions.contains({ permissions: ["notifications"] }));
  if (!notificationsAllowed || !chrome.notifications) return entries.length;
  for (const entry of entries.slice(0, 3)) {
    await chrome.notifications.create(`paper:${entry.workId}`, {
      type: "basic",
      iconUrl: chrome.runtime.getURL("assets/icon-128.png"),
      title: "New paper cleared both filters",
      message: `${entry.title}\nNovelty ${entry.noveltyScore} · Authorship ${entry.researcherScore}`,
      contextMessage: entry.topic,
      priority: 1,
    });
  }
  if (entries.length > 3) {
    await chrome.notifications.create(`summary:${Date.now()}`, {
      type: "basic",
      iconUrl: chrome.runtime.getURL("assets/icon-128.png"),
      title: `${entries.length} new papers cleared both filters`,
      message: "Open the Filtered Research inbox to review the full batch.",
      priority: 1,
    });
  }
  return entries.length;
}

async function performRefresh(reason = "manual") {
  const settings = await loadSettings();
  const taxonomy = await getOpenAlexTaxonomy();
  const signature = discoveryScopeSignature(settings);
  const coverageByScope = (await getMetadata("coverageByScope")) || {};
  const legacyCoverage = await getMetadata("discoveryCoverage");
  const previousCoverage = migratedCoverage(coverageByScope[signature] || (legacyCoverage?.signature === signature ? legacyCoverage : null));
  const selection = selectionFromSettings(settings);
  const hasFocusedScope = selection.fieldIds.length || selection.subfieldIds.length;
  const fullScan =
    Boolean(settings.apiKey) &&
    hasFocusedScope &&
    (["manual", "rebuild"].includes(reason) || !previousCoverage?.fullCompletedAt);
  const mode = fullScan ? "full" : settings.apiKey && hasFocusedScope ? "incremental" : "limited";
  const startedAt = new Date().toISOString();
  const baseState = { reason, startedAt, mode, signature };
  const writeProgress = progressWriter(baseState);
  await writeProgress({ phase: "starting", fetched: 0, total: null, pages: 0 });

  const client = openAlexClient({
    apiKey: settings.apiKey,
    maxEstimatedCostUsd: fullScan ? settings.fullScanBudgetUsd : settings.incrementalScanBudgetUsd,
  });
  const until = isoDate();
  const since = fullScan ? daysAgo(settings.maxTimeframeDays) : mode === "limited" ? daysAgo(Math.min(30, settings.maxTimeframeDays)) : daysAgo(settings.incrementalLookbackDays);

  try {
    const [previousLastRefresh, previouslyStoredWorks] = await Promise.all([
      getMetadata("lastRefresh"),
      getAll("works"),
    ]);
    const existingCandidateIds = new Set(
      previouslyStoredWorks.filter((work) => !work.isBaseline).map((work) => work.id),
    );
    const discovery = await fetchDiscovery(
      client,
      settings,
      taxonomy,
      since,
      until,
      mode,
      writeProgress,
    );
    const candidates = discovery.works.map((work) => ({ ...work, isBaseline: false }));

    const baseline = await maybeFetchBaseline(client, settings, taxonomy, daysAgo(settings.maxTimeframeDays), writeProgress);
    if (baseline.length) await storeWorks(baseline);

    const existingAuthors = await getAll("authors");
    const freshCutoff = Date.now() - 30 * 24 * 60 * 60 * 1000;
    const freshAuthorIds = new Set(
      existingAuthors
        .filter((author) => Date.parse(author.fetchedAt || 0) >= freshCutoff)
        .map((author) => author.id),
    );
    const requestedAuthorIds = chooseAuthorIds(candidates, settings.maxAuthors);
    const missingAuthorIds = requestedAuthorIds.filter((id) => !freshAuthorIds.has(id));
    const fetchedAuthors = await client.fetchAuthors(missingAuthorIds, {
      onProgress: (progress) => writeProgress({ phase: "authorship", lane: "authors", ...progress }),
    });
    if (fetchedAuthors.length) await bulkPut("authors", fetchedAuthors);

    await writeProgress({
      phase: "scoring",
      lane: "local cosine comparison",
      fetched: 0,
      total: candidates.length,
      pages: 0,
    });
    const allAuthors = deduplicateAuthors([...existingAuthors, ...fetchedAuthors]);
    await getProminenceRoster(allAuthors);
    const existingWorks = await getAll("works");
    const candidateIds = new Set(candidates.map((work) => work.id));
    const references = selectReferences(
      deduplicate([...existingWorks, ...baseline]).filter((work) => !candidateIds.has(work.id)),
      settings.maxReferenceWorks,
    );
    // Novelty is calibrated against the field, so scores from an older scoring
    // version are not comparable with new ones. Anything stale is rescored in
    // the same batch, which also keeps one consistent calibration across the
    // whole feed rather than mixing scales.
    const stale = existingWorks.filter(
      (item) => !item.isBaseline && item.scoringVersion && item.scoringVersion !== SCORING_VERSION && !candidateIds.has(item.id),
    );
    if (stale.length) {
      await writeProgress({
        phase: "rescoring",
        lane: `${stale.length} saved papers on an older scoring version`,
        fetched: 0,
        total: stale.length,
        pages: 0,
      });
    }
    const scored = scoreBatch([...candidates, ...stale], references, allAuthors, {
      maxPeerComparisons: settings.maxPeerComparisons,
    });
    await storeWorks(scored);

    const notificationsGenerated = await recordNewPaperNotifications(
      scored,
      settings,
      existingCandidateIds,
      previousLastRefresh,
      reason,
    );
    await pruneCandidates(daysAgo(400));

    const completedAt = new Date().toISOString();
    if (usageAccumulator) await setMetadata("apiUsageDaily", usageAccumulator);
    const coverage = fullScan
      ? {
          signature, filterSignature: researchFilterSignature(settings),
          horizonDays: settings.maxTimeframeDays,
          mode,
          available: discovery.total,
          retrieved: discovery.retrieved,
          records: discovery.records,
          matched: candidates.length,
          // Measured on records actually downloaded, not on unique papers left
          // after duplicate merging, which made a complete pass look partial.
          coveragePercent: discovery.total
            ? Math.min(100, (100 * (discovery.records ?? discovery.retrieved)) / discovery.total)
            : 100,
          truncated: discovery.truncated,
          fullCompletedAt: completedAt,
          lastIncrementalAt: completedAt,
        }
      : {
          ...(previousCoverage || {}), filterSignature: researchFilterSignature(settings), horizonDays: Math.max(Number(previousCoverage?.horizonDays || 0), settings.maxTimeframeDays),
          signature,
          mode,
          limitedAvailable: discovery.total,
          limitedRetrieved: discovery.retrieved,
          limitedMatched: candidates.length,
          lastIncrementalAt: completedAt,
          needsApiKey: !settings.apiKey && hasFocusedScope,
        };
    const state = {
      status: "ready",
      reason,
      mode,
      startedAt,
      completedAt,
      candidatesFetched: candidates.length,
      indexedRetrieved: discovery.retrieved,
      indexedAvailable: discovery.total,
      baselineFetched: baseline.length,
      authorsFetched: fetchedAuthors.length,
      apiCostUsd: client.costUsd,
      estimatedApiCostUsd: client.estimatedCostUsd,
      keyPresent: Boolean(settings.apiKey),
      notificationsGenerated,
    };
    const history = (await getMetadata("searchHistory")) || [];
    const currentFilterSignature = researchFilterSignature(settings);
    const historyEntry = { id: `${completedAt}:${Math.random().toString(36).slice(2)}`, filterSignature: currentFilterSignature, savedAt: completedAt,
      settings: { queries: settings.queries, selectedFields: settings.selectedFields, selectedSubfields: settings.selectedSubfields, selectedArxivGroups: settings.selectedArxivGroups, selectedArxivCategories: settings.selectedArxivCategories, englishOnly: settings.englishOnly,
        noveltySelectivity: settings.noveltySelectivity, authorshipSelectivity: settings.authorshipSelectivity, defaultWindow: settings.defaultWindow, defaultSort: settings.defaultSort },
      resultCount: (await qualifiedMonth(settings)).works.length, mode };
    await Promise.all([
      setMetadata("lastRefresh", completedAt),
      setMetadata("refreshState", state),
      setMetadata("discoveryCoverage", coverage),
      setMetadata("coverageByScope", { ...coverageByScope, [signature]: coverage }),
      setMetadata("searchHistory", [historyEntry, ...history.filter((item) => item.filterSignature !== currentFilterSignature)].slice(0, 12)),
    ]);
    // Any pass can add unread papers, so the toolbar dot follows every one.
    await refreshUnreadBadge();
    return state;
  } catch (error) {
    const state = {
      status: "error",
      reason,
      mode,
      startedAt,
      completedAt: new Date().toISOString(),
      message: error instanceof Error ? error.message : String(error),
      estimatedApiCostUsd: client.estimatedCostUsd,
    };
    await setMetadata("refreshState", state);
    throw error;
  }
}

function refresh(reason) {
  if (refreshPromise) {
    return refreshPromise;
  }
  refreshPromise = performRefresh(reason).finally(() => {
    refreshPromise = null;
    if (pendingRefreshReason) {
      const nextReason = pendingRefreshReason;
      pendingRefreshReason = null;
      refresh(nextReason).catch(console.error);
    }
  });
  return refreshPromise;
}

const SORTERS = {
  balanced: (left, right) => right.discoveryScore - left.discoveryScore,
  novelty: (left, right) => right.noveltyScore - left.noveltyScore,
  researcher: (left, right) => right.researcherScore - left.researcherScore,
  newest: (left, right) => releasedOn(right).localeCompare(releasedOn(left)),
};

// Shared preamble for both the single-window and all-window paths, so a bundle
// costs exactly one settings load, one batched metadata read and one corpus
// scan no matter how many windows it covers.
async function feedContext(settings) {
  const signature = discoveryScopeSignature(settings);
  const meta = await getMetadataMany([
    "coverageByScope",
    "discoveryCoverage",
    "lastRefresh",
    "settingsChangedAt",
  ]);
  const legacyCoverage = meta.discoveryCoverage;
  const coverage = migratedCoverage(
    (meta.coverageByScope || {})[signature] ||
      (legacyCoverage?.signature === signature ? legacyCoverage : null),
  );
  const indexedHorizonDays = effectiveHorizonDays(settings, coverage);
  return {
    coverage,
    indexedHorizonDays,
    maxTimeframeDays: settings.maxTimeframeDays,
    settingsChangedAt: meta.settingsChangedAt,
    corpus: await filteredCorpus(settings, { lastRefresh: meta.lastRefresh }),
    stats: await feedStats(),
  };
}

// Interests no longer exclude anything, so they steer order instead. This used
// to be a boolean -- every paper mentioning the phrase ranked equally -- which
// is why searching "RAG" put Tagalog relative-clause processing above every
// retrieval-augmented generation paper in the index. Relevance is now graded,
// and one index is built for the whole window rather than per paper, because
// BM25 needs corpus-wide term statistics.
async function interestRanked(works, settings) {
  const queries = (settings.queries || []).filter(Boolean);
  if (!queries.length) return works;
  const index = buildRelevanceIndex(works);
  const annotated = works.map((work) => {
    const relevance = bestRelevance(work, queries, index);
    return {
      ...work,
      interestMatch: interestMatchEvidence(work, queries),
      relevanceScore: relevance?.score || 0,
      relevanceEvidence: relevance,
    };
  });

  // The lexical score above is the fallback and the tie-break. When the bundled
  // sentence model is present it decides the order instead, because a reader's
  // phrase and the paper that answers it routinely share no words: measured on
  // held-out queries, ranking this way lifts nDCG@20 from 0.43 to 0.59.
  const ordered = await rankByRelevanceWithModel(annotated, queries[0], {
    index,
    embedding: EMBEDDING_RUNTIME,
  }).catch(() => null);
  if (!ordered) return annotated;

  const position = new Map(ordered.map((work, index2) => [work.id, index2]));
  return annotated.map((work) => ({ ...work, semanticRank: position.get(work.id) ?? null }));
}

// Where the model lives inside the packaged extension, and how to reach the
// runtime that executes it. Both resolve to bundled files; nothing is fetched.
const EMBEDDING_RUNTIME = {
  importTransformers: () => import("../../vendor/transformers/transformers.js"),
  modelPath: chrome.runtime.getURL("vendor/"),
  wasmPath: chrome.runtime.getURL("vendor/transformers/"),
  batchSize: 16,
};

// How much clearer one paper's relevance has to be before it outranks the
// chosen sort order. Without a band, a one-point relevance difference would
// silently override the novelty ordering the user actually selected.
const RELEVANCE_DECIDES_ABOVE = 8;

async function windowSlice(context, settings, { window, sort, includeAll, offset, limit }) {
  const windowConfig = WINDOWS[window] || WINDOWS.week;
  const effectiveDays = Math.min(windowConfig.days, context.indexedHorizonDays);
  const cutoff = daysAgo(effectiveDays);
  const relevant = context.corpus.filter((work) => releasedOn(work) >= cutoff);
  const selected = applySelectivity(relevant, settings, { includeAll });
  const ranked = await interestRanked(selected.works, settings);
  const base = SORTERS[sort] || SORTERS.balanced;
  ranked.sort((left, right) => {
    // A semantic ordering, when one exists, is the strongest evidence of what the
    // reader asked for, so it leads. Without it the graded lexical score decides,
    // and below the band the user's chosen sort takes over.
    if (left.semanticRank != null && right.semanticRank != null && left.semanticRank !== right.semanticRank) {
      return left.semanticRank - right.semanticRank;
    }
    const delta = (right.relevanceScore || 0) - (left.relevanceScore || 0);
    if (Math.abs(delta) >= RELEVANCE_DECIDES_ABOVE) return delta;
    return base(left, right);
  });
  selected.works = ranked;
  const safeOffset = Math.max(0, Number(offset) || 0);
  const safeLimit = Math.min(250, Math.max(1, Number(limit) || 120));
  const page = selected.works.slice(safeOffset, safeOffset + safeLimit).map((work) => ({
    ...work,
    interestMatch: interestMatchEvidence(work, settings.queries || []),
  }));
  return {
    papers: page.map(cleanWorkForDisplay),
    resultCount: selected.works.length,
    offset: safeOffset,
    hasMore: safeOffset + page.length < selected.works.length,
    screenedCount: relevant.length,
    window,
    sort,
    thresholds: {
      noveltySelectivity: settings.noveltySelectivity,
      authorshipSelectivity: settings.authorshipSelectivity,
      cutoffs: selected.cutoffs,
      topFractions: selected.topFractions,
    },
    coverage: context.coverage,
    indexedHorizonDays: context.indexedHorizonDays,
    maxTimeframeDays: context.maxTimeframeDays,
    requestedBeyondCoverage: windowConfig.days > context.indexedHorizonDays,
    settingsChangedAt: context.settingsChangedAt,
    stats: context.stats,
  };
}

async function getFeed({
  window = "week",
  sort = "balanced",
  includeAll = false,
  offset = 0,
  limit = 120,
} = {}) {
  const settings = await loadSettings();
  const context = await feedContext(settings);
  return await windowSlice(context, settings, { window, sort, includeAll, offset, limit });
}

// Every date view in one response. The sidebar renders tab switches straight
// from this, so changing tabs costs no message round trip and cannot be slowed
// down by a cold service worker.
async function getFeedBundle({ sort = "balanced", limit = 120 } = {}) {
  const settings = await loadSettings();
  const context = await feedContext(settings);
  // Availability follows the depth the user chose, not how far the last pass
  // happened to reach. Building only up to the indexed horizon left a depth the
  // user had just selected showing as unavailable until they refreshed.
  const available = windowsWithin(context.maxTimeframeDays);
  const windows = Object.fromEntries(
    await Promise.all(available.map(async (window) => [
      window,
      await windowSlice(context, settings, { window, sort, includeAll: false, offset: 0, limit }),
    ])),
  );
  return {
    sort,
    windows,
    availableWindows: available,
    indexedHorizonDays: context.indexedHorizonDays,
    maxTimeframeDays: context.maxTimeframeDays,
  };
}

// Hybrid retrieval. The feed always renders from the local index first, then
// this fills the window's coverage hole from OpenAlex in the background. It is
// bounded, scoped to one window, and rate-limited per scope so it can never
// turn a render into a network wait.
const GAP_FILL_LIMIT = 400;
const GAP_FILL_INTERVAL_MS = 6 * 60 * 60 * 1000;

async function fillFeedGap({ window = "week" } = {}) {
  const settings = await loadSettings();
  const selection = selectionFromSettings(settings);
  if (!settings.apiKey || !(selection.fieldIds.length || selection.subfieldIds.length)) {
    return { added: 0, skipped: "needs an API key and a selected category" };
  }
  const signature = discoveryScopeSignature(settings);
  const horizon = effectiveHorizonDays(settings, migratedCoverage(
    ((await getMetadata("coverageByScope")) || {})[signature],
  ));
  const days = Math.min(WINDOWS[window]?.days || 7, horizon);
  const stampKey = `gapFill:${signature}:${days}`;
  const stamps = (await getMetadata("gapFillStamps")) || {};
  if (Date.now() - Number(stamps[stampKey] || 0) < GAP_FILL_INTERVAL_MS) {
    return { added: 0, skipped: "already filled recently" };
  }

  const client = openAlexClient({ apiKey: settings.apiKey, maxEstimatedCostUsd: settings.gapFillBudgetUsd });
  let fetched = [];
  try {
    const result = await client.fetchWorksCursor({
      since: daysAgo(days),
      until: isoDate(),
      limit: GAP_FILL_LIMIT,
      query: "",
      fieldIds: selection.fieldIds,
      subfieldIds: selection.subfieldIds,
      englishOnly: settings.englishOnly,
      requireAbstract: false,
    });
    fetched = result.works || [];
  } catch (error) {
    console.warn("Gap fill stopped", error);
    return { added: 0, skipped: "budget or network limit" };
  }

  const known = new Set((await getAll("works")).map((work) => work.id));
  const candidates = fetched
    .filter((work) => !known.has(work.id))
    .filter((work) => matchesResearchFilters(work, settings))
    .map((work) => ({ ...work, isBaseline: false }));
  await setMetadata("gapFillStamps", { ...stamps, [stampKey]: Date.now() });
  if (!candidates.length) return { added: 0 };

  const authorIds = chooseAuthorIds(candidates, Math.min(settings.maxAuthors, 3_000));
  const fetchedAuthors = await client.fetchAuthors(authorIds).catch(() => []);
  if (fetchedAuthors.length) await bulkPut("authors", fetchedAuthors);
  const allAuthors = deduplicateAuthors([...(await getAll("authors")), ...fetchedAuthors]);
  await getProminenceRoster(allAuthors);
  const references = selectReferences(await getAll("works"), settings.maxReferenceWorks);
  const scored = scoreBatch(candidates, references, allAuthors, {
    maxPeerComparisons: settings.maxPeerComparisons,
  });
  await storeWorks(scored);
  return { added: scored.length, window, days };
}

async function getNotifications() {
  return (await getMetadata("notificationInbox")) || [];
}

async function markNotificationsRead(ids = []) {
  const selected = new Set(ids.map(String));
  const inbox = await getNotifications();
  const updated = inbox.map((item) => ({
    ...item,
    unread: selected.size ? item.unread && !selected.has(item.id) : false,
  }));
  await setMetadata("notificationInbox", updated);
  return updated.filter((item) => item.unread).length;
}

async function handleMessage(message) {
  switch (message?.type) {
    case "GET_FEED":
      return getFeed(message.payload);
    case "GET_FEED_BUNDLE":
      return getFeedBundle(message.payload);
    case "FILL_FEED_GAP":
      return fillFeedGap(message.payload || {});
    case "GET_STATUS":
      return databaseStats();
    case "GET_TAXONOMY":
      return getCategoryTaxonomy(message.payload || {});
    case "GET_API_USAGE":
      return (await getMetadata("apiUsageDaily")) || { date: isoDate(), requests: 0, costUsd: 0 };
    case "GET_SEARCH_HISTORY":
      return (await getMetadata("searchHistory")) || [];
    case "SET_MAX_TIME_FRAME":
    case "SET_MAX_TIMEFRAME": {
      const stored = await chrome.storage.sync.get("settings");
      const next = normalizeSettings({ ...(stored.settings || {}), maxTimeframeDays: message.payload?.days });
      await chrome.storage.sync.set({ settings: next });
      return { maxTimeframeDays: next.maxTimeframeDays, discoveryStarted: false };
    }
    case "CLEAR_FEED_CACHE":
      feedCorpusCache = null;
      return { ok: true };
    // Started without awaiting: a pass outlives the service worker's guaranteed
    // lifetime, and holding the message channel open for it fails with "the
    // message channel closed before a response was received". Callers watch
    // GET_REFRESH_STATE instead.
    case "REFRESH":
      refresh("manual").catch((error) => console.warn("Discovery pass failed", error));
      return { started: true, reason: "manual" };
    case "REBUILD":
      refresh("rebuild").catch((error) => console.warn("Discovery pass failed", error));
      return { started: true, reason: "rebuild" };
    case "GET_REFRESH_STATE":
      return (await getMetadata("refreshState")) || null;
    case "SETTINGS_CHANGED":
      feedCorpusCache = null;
      await scheduleAutoScan();
      await setMetadata("settingsChangedAt", new Date().toISOString());
      return { ok: true, discoveryStarted: false };
    case "CLEAR_DATA":
      await clearDatabase();
      feedCorpusCache = null;
      return { ok: true };
    case "GET_NOTIFICATIONS":
      return getNotifications();
    case "MARK_NOTIFICATIONS_READ": {
      const unread = await markNotificationsRead(message.payload?.ids || []);
      await refreshUnreadBadge();
      return unread;
    }
    case "GET_AUTO_SCAN_STATE":
      return {
        lastAutoScan: (await getMetadata("lastAutoScan")) || null,
        unread: (await getMetadata("notificationInbox") || []).filter((entry) => entry.unread).length,
      };
    case "RUN_AUTO_SCAN_IF_DUE":
      return runAutoScanIfDue("manual-check");
    case "CLEAR_NOTIFICATIONS":
      await setMetadata("notificationInbox", []);
      await refreshUnreadBadge();
      return { ok: true };
    default:
      throw new Error(`Unknown message type: ${message?.type}`);
  }
}

// The toolbar icon carries a red dot whenever the inbox holds unread papers, so
// a background pass is visible without opening anything.
const BADGE_COLOUR = "#c8362f";
async function refreshUnreadBadge() {
  if (!chrome.action?.setBadgeText) return 0;
  const inbox = (await getMetadata("notificationInbox")) || [];
  const unread = inbox.filter((entry) => entry.unread).length;
  try {
    await chrome.action.setBadgeBackgroundColor({ color: BADGE_COLOUR });
    if (chrome.action.setBadgeTextColor) await chrome.action.setBadgeTextColor({ color: "#ffffff" });
    await chrome.action.setBadgeText({ text: unread ? (unread > 99 ? "99+" : String(unread)) : "" });
  } catch (error) {
    console.warn("Could not update the toolbar badge", error);
  }
  return unread;
}

// Automatic scanning. A pass is due when the chosen interval has elapsed since
// the last automatic one; the alarm covers a browser left running for days and
// the startup check covers the interval passing while Chrome was closed.
const AUTO_SCAN_ALARM = "filteredresearch-auto-scan";
const AUTO_SCAN_CHECK_MINUTES = 30;

async function scheduleAutoScan() {
  if (!chrome.alarms) return;
  const { autoScanHours } = await loadSettings();
  await chrome.alarms.clear(AUTO_SCAN_ALARM);
  if (!normalizeAutoScanHours(autoScanHours)) return;
  // Checked more often than the interval so a due pass starts promptly rather
  // than waiting a whole further period.
  chrome.alarms.create(AUTO_SCAN_ALARM, { periodInMinutes: AUTO_SCAN_CHECK_MINUTES });
}

async function runAutoScanIfDue(reason = "alarm") {
  const settings = await loadSettings();
  const hours = normalizeAutoScanHours(settings.autoScanHours);
  if (!hours) return { ran: false, skipped: "automatic scanning is off" };
  const selection = selectionFromSettings(settings);
  if (!settings.apiKey || !(selection.fieldIds.length || selection.subfieldIds.length)) {
    return { ran: false, skipped: "needs an API key and a selected category" };
  }
  const lastScan = await getMetadata("lastAutoScan");
  if (!autoScanDue(hours, lastScan)) return { ran: false, skipped: "not due yet" };
  if (refreshPromise) return { ran: false, skipped: "a pass is already running" };

  await setMetadata("lastAutoScan", new Date().toISOString());
  try {
    const state = await refresh(`auto:${reason}`);
    await refreshUnreadBadge();
    return { ran: true, notificationsGenerated: state?.notificationsGenerated || 0 };
  } catch (error) {
    console.warn("Automatic scan failed", error);
    return { ran: false, skipped: error instanceof Error ? error.message : String(error) };
  }
}

chrome.runtime.onInstalled.addListener(() => {
  Promise.all([
    ensureDefaults(),
    chrome.sidePanel.setPanelBehavior({ openPanelOnActionClick: true }),
    getMetadata("refreshState").then((state) => state?.status === "running" ? setMetadata("refreshState", { ...state, status: "ready", interruptedAt: new Date().toISOString() }) : null),
    scheduleAutoScan(),
    refreshUnreadBadge(),
  ]).catch(console.error);
});

chrome.runtime.onStartup.addListener(() => {
  ensureDefaults()
    .then(() => Promise.all([scheduleAutoScan(), refreshUnreadBadge()]))
    .then(() => runAutoScanIfDue("startup"))
    .catch(console.error);
});

if (chrome.alarms?.onAlarm) {
  chrome.alarms.onAlarm.addListener((alarm) => {
    if (alarm.name !== AUTO_SCAN_ALARM) return;
    runAutoScanIfDue("alarm").catch(console.error);
  });
}

if (chrome.notifications?.onClicked) {
  chrome.notifications.onClicked.addListener((notificationId) => {
    if (notificationId.startsWith("paper:")) {
      const workId = notificationId.slice("paper:".length);
      getById("works", workId)
        .then((work) => chrome.tabs.create({ url: safePaperUrl(work || { id: workId }) }))
        .catch(console.error);
    } else {
      chrome.tabs
        .create({ url: chrome.runtime.getURL("src/notifications/notifications.html") })
        .catch(console.error);
    }
    chrome.notifications.clear(notificationId).catch(console.error);
  });
}

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  handleMessage(message)
    .then((result) => sendResponse({ ok: true, result }))
    .catch((error) =>
      sendResponse({
        ok: false,
        error: error instanceof Error ? error.message : String(error),
      }),
    );
  return true;
});

ensureDefaults().catch(console.error);
