// Do any cheap, unused pieces of metadata predict disruption? Measured on the
// tuning split before anything is added to the fusion.
import { loadCorpus } from "../lib/adapters.mjs";
import { buildLabels, disruptionClasses } from "../lib/labels.mjs";
import { spearman, auc } from "../lib/metrics.mjs";

const corpus = loadCorpus();
const labels = buildLabels(corpus.rawCandidates);
const authors = new Map(corpus.authors.map((a) => [a.id, a]));
const tune = corpus.rawCandidates.filter((w) => w.split === "tune");
const labelled = tune.filter((w) => labels.get(w.id)?.noveltyLabel !== null);
const classes = disruptionClasses(tune, labels);

const hIndexes = (w) => (w.authorships || []).map((a) => authors.get(a.authorId)?.hIndex).filter((v) => Number.isFinite(v));
const CANDIDATES = {
  institutionCount: (w) => new Set((w.authorships || []).flatMap((a) => a.institutions || [])).size,
  firstAuthorH: (w) => authors.get((w.authorships || [])[0]?.authorId)?.hIndex ?? null,
  minAuthorH: (w) => { const h = hIndexes(w); return h.length ? Math.min(...h) : null; },
  maxAuthorH: (w) => { const h = hIndexes(w); return h.length ? Math.max(...h) : null; },
  titleLength: (w) => (w.title || "").length,
  abstractLength: (w) => (w.abstract || "").length,
  titleHasColon: (w) => ((w.title || "").includes(":") ? 1 : 0),
  isPreprint: (w) => (w.workType === "preprint" ? 1 : 0),
  authorCount: (w) => (w.authorships || []).length,
};

console.log("signal              n   spearman      AUC   (both directions shown as-is)");
for (const [name, fn] of Object.entries(CANDIDATES)) {
  const present = labelled.filter((w) => Number.isFinite(fn(w)));
  if (present.length < 60) { console.log(`${name.padEnd(20)} ${present.length} too few`); continue; }
  const rho = spearman(present.map(fn), present.map((w) => labels.get(w.id).noveltyLabel));
  const sep = auc(
    classes.positives.filter((w) => Number.isFinite(fn(w))).map(fn),
    classes.negatives.filter((w) => w.workType !== "review" && Number.isFinite(fn(w))).map(fn),
  );
  console.log(`${name.padEnd(20)} ${String(present.length).padStart(4)}  ${rho.toFixed(4).padStart(8)}  ${sep.toFixed(4).padStart(7)}`);
}
