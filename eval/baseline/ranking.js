export const SELECTIVITY_ANCHORS = Object.freeze([
  { value: 1, topFraction: 1, label: "Nearly every matched paper" },
  { value: 20, topFraction: 0.75, label: "Broad · top 75%" },
  { value: 40, topFraction: 0.5, label: "Focused · top 50%" },
  { value: 60, topFraction: 0.2, label: "Selective · top 20%" },
  { value: 80, topFraction: 0.05, label: "Strong · top 5%" },
  { value: 90, topFraction: 0.01, label: "Rare · top 1%" },
  { value: 100, topFraction: 0.0002, label: "Extreme · top 0.02%" },
]);

function clamp(value, minimum, maximum) {
  return Math.min(maximum, Math.max(minimum, value));
}

export function selectivityToTopFraction(value) {
  const selected = clamp(Number(value) || 1, 1, 100);
  const exact = SELECTIVITY_ANCHORS.find((anchor) => anchor.value === selected);
  if (exact) return exact.topFraction;
  const upperIndex = SELECTIVITY_ANCHORS.findIndex((anchor) => anchor.value >= selected);
  if (upperIndex <= 0) return SELECTIVITY_ANCHORS[0].topFraction;
  const lower = SELECTIVITY_ANCHORS[upperIndex - 1];
  const upper = SELECTIVITY_ANCHORS[upperIndex];
  const progress = (selected - lower.value) / (upper.value - lower.value);
  const logFraction =
    Math.log10(lower.topFraction) +
    progress * (Math.log10(upper.topFraction) - Math.log10(lower.topFraction));
  return 10 ** logFraction;
}

export function describeSelectivity(value) {
  const fraction = selectivityToTopFraction(value);
  if (fraction >= 0.999) return "Nearly every matched paper";
  const percentage = fraction * 100;
  const formatted = percentage >= 10 ? Math.round(percentage) : Number(percentage.toPrecision(2));
  return `Approximately the top ${formatted}% on this signal`;
}

function cutoffForTopFraction(values, fraction) {
  if (!values.length || fraction >= 0.999) return Number.NEGATIVE_INFINITY;
  const sorted = values.map(Number).sort((left, right) => right - left);
  const keepCount = Math.max(1, Math.ceil(sorted.length * fraction));
  return sorted[Math.min(sorted.length - 1, keepCount - 1)];
}

export function applySelectivity(works, settings = {}, { includeAll = false } = {}) {
  works = (Array.isArray(works) ? works : []).filter((w) => w && typeof w === "object");
  if (!settings || typeof settings !== "object") settings = {};
  const noveltyFraction = selectivityToTopFraction(settings.noveltySelectivity);
  const authorshipFraction = selectivityToTopFraction(settings.authorshipSelectivity);
  const noveltyCutoff = cutoffForTopFraction(
    works.map((work) => work.noveltyScore || 0),
    noveltyFraction,
  );
  const authorshipCutoff = cutoffForTopFraction(
    works.map((work) => work.researcherScore || 0),
    authorshipFraction,
  );
  const selected = includeAll
    ? [...works]
    : works.filter(
        (work) =>
          (work.noveltyScore || 0) >= noveltyCutoff &&
          (Number(settings.noveltySelectivity || 1) <= 1 || (work.noveltyScore || 0) >= Number(settings.noveltySelectivity)) &&
          (((work.researcherScore || 0) >= authorshipCutoff &&
            (Number(settings.authorshipSelectivity || 1) <= 1 || (work.researcherScore || 0) >= Number(settings.authorshipSelectivity))) ||
            work.authorshipOverride),
      );
  return {
    works: selected,
    cutoffs: { novelty: noveltyCutoff, authorship: authorshipCutoff },
    topFractions: { novelty: noveltyFraction, authorship: authorshipFraction },
  };
}
