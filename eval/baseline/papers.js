import { normalizeSearchText } from "./filters.js";
import { firstReleaseDate } from "./openalex.js";
export function normalizeDoi(value) { return String(value || "").toLowerCase().replace(/^https?:\/\/(?:dx\.)?doi\.org\//, "").replace(/^doi:\s*/, "").trim(); }
export function canonicalPaperKey(work) {
  const title = normalizeSearchText(work.title);
  const authors = (work.authorships || []).slice(0, 3).map((item) => normalizeSearchText(item.name)).filter(Boolean).join("|");
  if (title && authors) return `title:${title}|authors:${authors}`;
  const doi = normalizeDoi(work.doi); if (doi) return `doi:${doi}`;
  if (work.arxivId) return `arxiv:${String(work.arxivId).replace(/v\d+$/i, "").toLowerCase()}`;
  return `title:${title}|id:${work.id}`;
}
function fallbackSource(work) { return { name: work.sourceName || "OpenAlex record", url: work.url || work.doi || `https://openalex.org/${work.id}`, doi: work.doi || null }; }
export function groupDuplicatePapers(works = []) {
  const groups = new Map();
  for (const work of works) { const key = canonicalPaperKey(work); groups.set(key, [...(groups.get(key) || []), work]); }
  return [...groups.values()].map((group) => {
    const best = [...group].sort((a, b) => String(b.abstract || "").length - String(a.abstract || "").length)[0];
    const rawSources = group.flatMap((work) => work.sources?.length ? work.sources : [fallbackSource(work)]);
    const sources = [...new Map(rawSources.map((source) => [`${normalizeSearchText(source.name)}|${String(source.url || "").toLowerCase()}`, source])).values()];
    const publicationDate = group.map((item) => item.publicationDate).filter(Boolean).sort()[0] || best.publicationDate;
    // Recency is judged on the earliest public appearance across every merged
    // record, so a preprint later re-published by a journal keeps its original
    // age instead of resurfacing as brand-new research.
    const released = group
      .map((item) => item.firstReleaseDate || firstReleaseDate(item))
      .filter(Boolean)
      .sort()[0];
    return { ...best, publicationDate,
      firstReleaseDate: [released, firstReleaseDate({ ...best, publicationDate })].filter(Boolean).sort()[0] || publicationDate,
      noveltyScore: Math.max(...group.map((item) => Number(item.noveltyScore || 0))), researcherScore: Math.max(...group.map((item) => Number(item.researcherScore || 0))),
      discoveryScore: Math.max(...group.map((item) => Number(item.discoveryScore || 0))), sources, sourceName: sources[0]?.name || best.sourceName,
      duplicateIds: group.map((item) => item.id), duplicateCount: group.length };
  });
}
