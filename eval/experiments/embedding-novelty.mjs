// Does a real semantic space help NOVELTY, not just relevance? The corpus-local
// random-indexing space is weak (same-topic AUC 0.67) and text crowding measured
// against disruption at chance. If a proper model changes that, bundling buys C1
// as well as C3 and the case is decisive; if not, it buys relevance only.
import { loadCorpus } from "../lib/adapters.mjs";
import { buildLabels, disruptionClasses } from "../lib/labels.mjs";
import { spearman, auc } from "../lib/metrics.mjs";

const SPIKE_DIR = process.env.FR_SPIKE_DIR;
if (!SPIKE_DIR) { console.log("Set FR_SPIKE_DIR."); process.exit(0); }
const NORMALISED = SPIKE_DIR.split("\\").join("/");
const t = await import(`file:///${NORMALISED}/node_modules/@xenova/transformers/src/transformers.js`);
t.env.allowLocalModels = true; t.env.allowRemoteModels = true;
t.env.localModelPath = `${NORMALISED}/models/`;
t.env.cacheDir = `${NORMALISED}/hfcache`;

const corpus = loadCorpus();
const labels = buildLabels(corpus.rawCandidates);
const byId = new Map(corpus.rawCandidates.map((w) => [w.id, w]));
const field = corpus.rawCandidates[0].benchField;
// One field keeps the embedding cost sane and matches how a cohort is scored.
const cands = corpus.candidates.filter((w) => byId.get(w.id)?.benchField === field);
const peers = corpus.peers.filter((p) => p.benchField === field).slice(0, 700);
console.log(`${cands.length} candidates, ${peers.length} peers`);

const pipe = await t.pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2", { quantized: true });
const text = (w) => `${w.title || ""}. ${String(w.abstract || "").slice(0, 900)}`;
async function embed(list) {
  const out = [];
  for (let i = 0; i < list.length; i += 32) {
    out.push(...(await pipe(list.slice(i, i + 32).map(text), { pooling: "mean", normalize: true })).tolist());
  }
  return out;
}
const started = Date.now();
const cv = await embed(cands);
const pv = await embed(peers);
console.log(`embedded in ${((Date.now() - started) / 1000).toFixed(1)}s`);

const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
// Crowding: nearest peer plus the density of the neighbourhood, same shape as the
// shipped measure but in a space that actually encodes meaning.
const crowding = cands.map((w, i) => {
  const sims = pv.map((p) => dot(cv[i], p)).sort((a, b) => b - a);
  return 0.6 * sims[0] + 0.4 * (sims.slice(0, 5).reduce((s, x) => s + x, 0) / 5);
});

const tune = cands.map((w, i) => ({ w: byId.get(w.id), crowd: crowding[i] })).filter((x) => x.w);
const labelled = tune.filter((x) => labels.get(x.w.id)?.noveltyLabel !== null);
const classes = disruptionClasses(tune.map((x) => x.w), labels);
const crowdOf = new Map(tune.map((x) => [x.w.id, x.crowd]));

for (const sign of [1, -1]) {
  const rho = spearman(labelled.map((x) => sign * x.crowd), labelled.map((x) => labels.get(x.w.id).noveltyLabel));
  const sep = auc(
    classes.positives.filter((w) => crowdOf.has(w.id)).map((w) => sign * crowdOf.get(w.id)),
    classes.negatives.filter((w) => w.workType !== "review" && crowdOf.has(w.id)).map((w) => sign * crowdOf.get(w.id)),
  );
  console.log(`crowding ${sign > 0 ? "as-is  " : "inverted"}  spearman ${rho.toFixed(4)}  AUC(vs derivative) ${sep.toFixed(4)}   n=${labelled.length}`);
}
