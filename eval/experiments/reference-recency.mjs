// Does "how much of this paper's foundation is recent field work" predict
// disruption? The peer corpus is 2018-2020, so a reference that IS a peer is
// recent by construction. Measured on the tuning split before fusing anything.
import { loadCorpus } from "../lib/adapters.mjs";
import { buildLabels, disruptionClasses } from "../lib/labels.mjs";
import { spearman, auc } from "../lib/metrics.mjs";

const corpus = loadCorpus();
const labels = buildLabels(corpus.rawCandidates);
const peerIds = new Set(corpus.peers.map((p) => p.id));
const peerRefs = new Map();
for (const p of corpus.peers) for (const r of p.referencedWorks || []) peerRefs.set(r, (peerRefs.get(r) || 0) + 1);

const tune = corpus.rawCandidates.filter((w) => w.split === "tune");
const labelled = tune.filter((w) => labels.get(w.id)?.noveltyLabel !== null);
const classes = disruptionClasses(tune, labels);

const CANDIDATES = {
  refsThatArePeers: (w) => { const r = w.referencedWorks || []; return r.length >= 3 ? r.filter((x) => peerIds.has(x)).length / r.length : null; },
  refsCitedOnce: (w) => { const r = w.referencedWorks || []; return r.length >= 3 ? r.filter((x) => (peerRefs.get(x) || 0) === 1).length / r.length : null; },
  refsUncited: (w) => { const r = w.referencedWorks || []; return r.length >= 3 ? r.filter((x) => !peerRefs.has(x)).length / r.length : null; },
  maxRefPopularity: (w) => { const r = w.referencedWorks || []; return r.length >= 3 ? Math.max(0, ...r.map((x) => peerRefs.get(x) || 0)) : null; },
  meanRefPopularity: (w) => { const r = w.referencedWorks || []; if (r.length < 3) return null; return r.reduce((s, x) => s + (peerRefs.get(x) || 0), 0) / r.length; },
};

console.log("signal                 n   spearman     AUC(vs derivative)");
for (const [name, fn] of Object.entries(CANDIDATES)) {
  const present = labelled.filter((w) => Number.isFinite(fn(w)));
  if (present.length < 60) { console.log(`${name.padEnd(22)} ${present.length} too few`); continue; }
  const rho = spearman(present.map(fn), present.map((w) => labels.get(w.id).noveltyLabel));
  const sep = auc(
    classes.positives.filter((w) => Number.isFinite(fn(w))).map(fn),
    classes.negatives.filter((w) => w.workType !== "review" && Number.isFinite(fn(w))).map(fn),
  );
  console.log(`${name.padEnd(22)} ${String(present.length).padStart(4)}  ${rho.toFixed(4).padStart(8)}  ${sep.toFixed(4).padStart(8)}`);
}
