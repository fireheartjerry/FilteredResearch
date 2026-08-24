import { normalizeSearchText } from "./filters.js";

const ORGS = [
  ["Anthropic", "anthropic", "#d97738"], ["OpenAI", "openai", "#202b29"], ["Google DeepMind", "deepmind", "#4285f4"], ["Meta AI", "meta ai|fair", "#1877f2"],
  ["Microsoft Research", "microsoft research", "#737373"], ["NVIDIA Research", "nvidia research|nvidia", "#76b900"], ["Apple ML", "apple machine learning|apple", "#555555"],
  ["Amazon Science", "amazon science|amazon", "#ff9900"], ["xAI", "xai", "#333333"], ["Mistral AI", "mistral ai", "#f08c32"], ["Cohere", "cohere", "#39594d"],
  ["AI2", "allen institute for ai|allen institute for artificial intelligence", "#5c6ac4"], ["Stanford HAI", "stanford institute for human centered artificial intelligence|stanford hai", "#8c1515"],
  ["MIT CSAIL", "mit csail|computer science and artificial intelligence laboratory", "#a31f34"], ["BAIR", "berkeley artificial intelligence research|bair", "#003262"],
  ["CMU ML", "carnegie mellon", "#c41230"], ["Mila", "mila", "#6b4eff"], ["Vector Institute", "vector institute", "#00a6a6"], ["Tsinghua AIR", "institute for ai industry research|tsinghua", "#6f2c91"],
  ["Shanghai AI Lab", "shanghai artificial intelligence laboratory|shanghai ai laboratory", "#d12f2f"], ["IBM Research", "ibm research", "#0f62fe"], ["Google Research", "google research", "#4285f4"],
  ["Toyota Research", "toyota research institute", "#eb0a1e"], ["Hugging Face", "hugging face", "#d5a400"], ["Salesforce AI", "salesforce ai research|salesforce research", "#0d9dda"],
];
const PEOPLE = [
  "Yann LeCun", "Geoffrey Hinton", "Yoshua Bengio", "Demis Hassabis", "David Silver", "Fei Fei Li", "Andrew Ng", "Judea Pearl", "Michael I Jordan", "Jurgen Schmidhuber",
  "Ilya Sutskever", "Dario Amodei", "Percy Liang", "Christopher Manning", "Kaiming He", "Timnit Gebru", "Joelle Pineau", "Pieter Abbeel", "Chelsea Finn", "Sergey Levine",
  "Ian Goodfellow", "Oriol Vinyals", "Noam Brown", "Richard Sutton", "Ruslan Salakhutdinov",
];
export const PROMINENCE_SEED_SIZE = ORGS.length + PEOPLE.length;
export const PROMINENCE_CATALOG_SIZE = PROMINENCE_SEED_SIZE;
function includesAlias(text, aliases) { const padded = ` ${text} `; return aliases.split("|").some((alias) => padded.includes(` ${normalizeSearchText(alias)} `)); }

export function buildProminentResearcherRoster(authors = []) {
  const seededNames = new Set(PEOPLE.map(normalizeSearchText));
  return [...authors]
    .filter((author) => author?.id && author?.name && !seededNames.has(normalizeSearchText(author.name)))
    .filter((author) => Number(author.hIndex || 0) >= 50 && Number(author.citedByCount || 0) >= 10_000)
    .sort((left, right) =>
      Number(right.hIndex || 0) - Number(left.hIndex || 0) ||
      Number(right.citedByCount || 0) - Number(left.citedByCount || 0) ||
      String(left.name).localeCompare(String(right.name)),
    )
    .map((author) => ({
      authorId: author.id,
      label: author.name,
      color: "#477c73",
      hIndex: Number(author.hIndex || 0),
      citedByCount: Number(author.citedByCount || 0),
    }));
}

export function prominenceMarkers(work, researcherRoster = []) {
  const markers = [];
  const affiliations = normalizeSearchText((work.authorships || []).flatMap((a) => [...(a.institutions || []), ...(a.rawAffiliations || [])]).join(" "));
  for (const [label, aliases, color] of ORGS) if (includesAlias(affiliations, aliases)) markers.push({ type: "organization", label, color });
  const authorNames = new Set((work.authorships || []).map((a) => normalizeSearchText(a.name)));
  for (const name of PEOPLE) if (authorNames.has(normalizeSearchText(name))) markers.push({ type: "researcher", label: name, color: "#477c73" });
  const authorIds = new Set((work.authorships || []).map((authorship) => authorship.authorId).filter(Boolean));
  for (const researcher of researcherRoster) {
    if (authorIds.has(researcher.authorId)) {
      markers.push({ type: "researcher", label: researcher.label, color: researcher.color || "#477c73" });
    }
  }
  const seen = new Set();
  return markers.filter((marker) => {
    const key = `${marker.type}:${normalizeSearchText(marker.label)}`;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  }).slice(0, 2);
}
export function annotateProminence(work, researcherRoster = []) { const prominence = prominenceMarkers(work, researcherRoster); return { ...work, prominence, authorshipOverride: prominence.length > 0 }; }
