import {
  legacySelection,
  normalizeFieldIds,
  normalizeSubfieldIds,
} from "./filters.js";

export const WINDOWS = Object.freeze({
  day: { label: "Past day", days: 1 },
  "3d": { label: "Past 3 days", days: 3 },
  week: { label: "Past week", days: 7 },
  "2w": { label: "Past 2 weeks", days: 14 },
  month: { label: "Past month", days: 30 },
  "3m": { label: "Past 3 months", days: 90 },
});

export const WINDOW_ORDER = Object.freeze(Object.keys(WINDOWS));

// Index depth drives how far back a discovery pass reaches. Depths beyond three
// months made every pass fetch far more works than the scoring stage could keep
// up with, so the ceiling matches the widest local view.
export const INDEX_DEPTHS = Object.freeze([
  { days: 1, label: "1 day", tier: "Light" },
  { days: 3, label: "3 days", tier: "Light" },
  { days: 7, label: "1 week", tier: "Moderate" },
  { days: 14, label: "2 weeks", tier: "Moderate" },
  { days: 30, label: "1 month", tier: "Intensive" },
  { days: 90, label: "3 months", tier: "Intensive" },
]);

export const MAX_INDEX_DEPTH_DAYS = 90;

// Automatic scanning. `0` keeps discovery entirely manual, which stays the
// default because a pass spends the user's own OpenAlex allowance.
export const AUTO_SCAN_CHOICES = Object.freeze([
  { hours: 0, label: "Only when I press refresh" },
  { hours: 3, label: "Every 3 hours" },
  { hours: 6, label: "Every 6 hours" },
  { hours: 24, label: "Every 24 hours" },
  { hours: 72, label: "Every 3 days" },
]);

export function normalizeAutoScanHours(value) {
  const hours = Number(value);
  return AUTO_SCAN_CHOICES.some((choice) => choice.hours === hours) ? hours : 0;
}

// Whether an automatic pass is due, given when the last one completed.
export function autoScanDue(autoScanHours, lastScanIso, now = Date.now()) {
  const hours = normalizeAutoScanHours(autoScanHours);
  if (!hours) return false;
  const last = Date.parse(lastScanIso || "");
  if (!Number.isFinite(last)) return true;
  return now - last >= hours * 60 * 60 * 1000;
}

export const DEFAULT_SETTINGS = Object.freeze({
  queries: [],
  selectedFields: [],
  selectedSubfields: [],
  selectedArxivGroups: [],
  selectedArxivCategories: [],
  englishOnly: true,
  notificationsEnabled: false,
  defaultWindow: "week",
  defaultSort: "balanced",
  noveltySelectivity: 70,
  authorshipSelectivity: 70,
  refreshHours: 6,
  broadSample: 1000,
  perQuery: 50_000,
  baselinePerSubfield: 320,
  historyYears: 3,
  maxQueries: 5,
  maxAuthors: 50_000,
  maxDiscoveryWorks: 1_000_000,
  maxReferenceWorks: 6_000,
  maxPeerComparisons: 320,
  incrementalLookbackDays: 2,
  maxTimeframeDays: 30,
  fullRebuildDays: 7,
  fullScanBudgetUsd: 0.95,
  incrementalScanBudgetUsd: 0.02,
  gapFillBudgetUsd: 0.1,
  autoScanHours: 0,
  strictInterestFilter: false,
});

export const INCREMENTAL_MARKERS = Object.freeze([
  "improved",
  "enhanced",
  "extension of",
  "variant of",
  "revisiting",
  "comparative study",
  "benchmarking",
  "fine-tuning",
  "fine tuning",
]);

// The saved index depth is a hard ceiling on every date view: a completed pass
// may physically hold older papers from a previous deeper depth, and those must
// not leak into the feed once the user narrows the scope. Kept here, and pure,
// so the clamp is testable instead of being re-derived inside the worker.
export function effectiveHorizonDays(settings = {}, coverage = null) {
  const depth = Number(settings.maxTimeframeDays);
  const ceiling = Number.isFinite(depth) && depth > 0 ? depth : DEFAULT_SETTINGS.maxTimeframeDays;
  const indexed = Number(coverage?.horizonDays);
  const reached = Number.isFinite(indexed) && indexed > 0 ? indexed : ceiling;
  return Math.max(1, Math.min(ceiling, reached));
}

// Date views wider than the index depth would silently show the same papers as
// the widest allowed view, so they are offered only up to the depth.
export function windowsWithin(days) {
  return WINDOW_ORDER.filter((name) => WINDOWS[name].days <= Math.max(1, Number(days) || 1));
}

export const SETTINGS_KEY = "settings";
export const SECRET_KEY = "openAlexApiKey";
export const REFRESH_ALARM = "filteredresearch-refresh";

export function normalizeSettings(value = {}) {
  const merged = { ...DEFAULT_SETTINGS, ...value };
  merged.noveltySelectivity =
    value.noveltySelectivity ?? value.minNovelty ?? DEFAULT_SETTINGS.noveltySelectivity;
  merged.authorshipSelectivity =
    value.authorshipSelectivity ?? value.minResearcher ?? DEFAULT_SETTINGS.authorshipSelectivity;
  merged.queries = Array.isArray(merged.queries)
    ? merged.queries
        .map((query) => String(query).trim())
        .filter(Boolean)
        .slice(0, DEFAULT_SETTINGS.maxQueries)
    : [];
  const legacy = legacySelection(value.selectedCategories);
  merged.selectedFields = normalizeFieldIds(value.selectedFields).length
    ? normalizeFieldIds(value.selectedFields)
    : legacy.fieldIds;
  merged.selectedSubfields = normalizeSubfieldIds(value.selectedSubfields).length
    ? normalizeSubfieldIds(value.selectedSubfields)
    : legacy.subfieldIds;
  merged.selectedArxivGroups = Array.isArray(value.selectedArxivGroups) ? [...new Set(value.selectedArxivGroups.map(String).filter(Boolean))] : [];
  merged.selectedArxivCategories = Array.isArray(value.selectedArxivCategories) ? [...new Set(value.selectedArxivCategories.map(String).filter(Boolean))] : [];
  if (!merged.selectedArxivGroups.length && !merged.selectedArxivCategories.length) {
    if (merged.selectedFields.includes("17")) merged.selectedArxivGroups = ["cs"];
    else if (merged.selectedSubfields.includes("1702")) merged.selectedArxivCategories = ["cs.AI"];
  }
  merged.defaultWindow = WINDOWS[merged.defaultWindow] ? merged.defaultWindow : "week";
  merged.defaultSort = ["balanced", "novelty", "researcher", "newest"].includes(
    merged.defaultSort,
  )
    ? merged.defaultSort
    : "balanced";

  const numericBounds = {
    noveltySelectivity: [1, 100],
    authorshipSelectivity: [1, 100],
    refreshHours: [1, 24],
    broadSample: [100, 2000],
    perQuery: [100, 100_000],
    baselinePerSubfield: [100, 500],
    historyYears: [1, 10],
    maxAuthors: [1000, 100_000],
    maxDiscoveryWorks: [5000, 1_000_000],
    maxReferenceWorks: [1000, 20_000],
    maxPeerComparisons: [50, 800],
    incrementalLookbackDays: [1, 7],
    maxTimeframeDays: [1, MAX_INDEX_DEPTH_DAYS],
    fullRebuildDays: [3, 30],
    fullScanBudgetUsd: [0.05, 0.95],
    incrementalScanBudgetUsd: [0.005, 0.1],
    gapFillBudgetUsd: [0.01, 0.4],
  };
  for (const [key, [minimum, maximum]] of Object.entries(numericBounds)) {
    const parsed = Number(merged[key]);
    merged[key] = Number.isFinite(parsed)
      ? Math.min(maximum, Math.max(minimum, parsed))
      : DEFAULT_SETTINGS[key];
  }
  // A saved 6-month or 1-year depth clamps into range above; snap it (and any
  // other stale value) onto a depth the picker can actually display.
  merged.maxTimeframeDays = INDEX_DEPTHS.reduce((closest, depth) =>
    Math.abs(depth.days - merged.maxTimeframeDays) < Math.abs(closest - merged.maxTimeframeDays)
      ? depth.days
      : closest,
  DEFAULT_SETTINGS.maxTimeframeDays);
  merged.perQuery = Math.max(merged.perQuery, DEFAULT_SETTINGS.perQuery);
  merged.maxDiscoveryWorks = Math.max(merged.maxDiscoveryWorks, DEFAULT_SETTINGS.maxDiscoveryWorks);
  merged.fullScanBudgetUsd = Math.max(merged.fullScanBudgetUsd, DEFAULT_SETTINGS.fullScanBudgetUsd);
  merged.strictInterestFilter = Boolean(merged.strictInterestFilter);
  merged.autoScanHours = normalizeAutoScanHours(merged.autoScanHours);
  merged.englishOnly = merged.englishOnly !== false;
  merged.notificationsEnabled = Boolean(merged.notificationsEnabled);
  delete merged.selectedCategories;
  delete merged.minNovelty;
  delete merged.minResearcher;
  return merged;
}

export async function loadSettings() {
  const stored = await chrome.storage.sync.get(SETTINGS_KEY);
  const local = await chrome.storage.local.get(SECRET_KEY);
  return {
    ...normalizeSettings(stored[SETTINGS_KEY]),
    apiKey: String(local[SECRET_KEY] || "").trim(),
  };
}

export async function saveSettings(settings) {
  const { apiKey = "", ...publicSettings } = settings;
  const normalized = normalizeSettings(publicSettings);
  await Promise.all([
    chrome.storage.sync.set({ [SETTINGS_KEY]: normalized }),
    chrome.storage.local.set({ [SECRET_KEY]: String(apiKey).trim() }),
  ]);
  return normalized;
}
