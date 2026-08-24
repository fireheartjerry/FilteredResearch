// Two cheap ways to get more out of the same encoder, measured on TUNE.
import { loadCorpus } from "../lib/adapters.mjs";
import { buildRelevanceQueries } from "../lib/labels.mjs";
import { ndcgAt, mean } from "../lib/metrics.mjs";

const corpus = loadCorpus();
const relevanceCorpus = loadCorpus({ forRelevance: true });
const splitOf = new Map(corpus.rawCandidates.map((w) => [w.id, w.split]));
const SPLIT = process.env.FR_SPLIT || "tune";
const docs = relevanceCorpus.candidates.filter((w) => splitOf.get(w.id) === SPLIT);
const ids = new Set(docs.map((w) => w.id));
const queries = buildRelevanceQueries(corpus.rawCandidates, { minimumRelevant: 10, maximumQueries: 60 })
  .map((e) => { const g = new Map([...e.gains].filter(([id]) => ids.has(id))); return { ...e, gains: g, relevantCount: [...g.values()].filter((x) => x >= 2).length }; })
  .filter((e) => e.relevantCount >= 6);
console.log(`split ${SPLIT}: ${docs.length} docs, ${queries.length} queries`);

const t = await import("@xenova/transformers");
t.env.allowRemoteModels = true; t.env.allowLocalModels = false;
t.env.cacheDir = "./vendor/.cache";
const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
const pipe = await t.pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2", { quantized: true });

async function encode(list) {
  const out = [];
  for (let i = 0; i < list.length; i += 32) out.push(...(await pipe(list.slice(i, i + 32), { pooling: "mean", normalize: true })).tolist());
  return out;
}
const whole = await encode(docs.map((w) => `${w.title || ""}. ${String(w.abstract || "").slice(0, 900)}`));
const titles = await encode(docs.map((w) => w.title || ""));
const abstracts = await encode(docs.map((w) => String(w.abstract || "").slice(0, 900) || w.title || ""));
const qv = await encode(queries.map((q) => q.query));

function score(label, sim) {
  const per = queries.map((entry, qi) => {
    const ranked = docs.map((w, di) => ({ w, s: sim(qi, di) })).sort((a, b) => b.s - a.s).map((x) => x.w);
    return ndcgAt(ranked, (w) => entry.gains.get(w.id) || 0, 20);
  });
  console.log(`${label.padEnd(38)} nDCG@20 ${mean(per).toFixed(4)}`);
}

score("whole document (current)", (q, d) => dot(qv[q], whole[d]));
score("title only", (q, d) => dot(qv[q], titles[d]));
score("abstract only", (q, d) => dot(qv[q], abstracts[d]));
score("max(title, abstract)", (q, d) => Math.max(dot(qv[q], titles[d]), dot(qv[q], abstracts[d])));
for (const w of [0.3, 0.5, 0.7]) {
  score(`title*${w} + abstract*${(1 - w).toFixed(1)}`, (q, d) => w * dot(qv[q], titles[d]) + (1 - w) * dot(qv[q], abstracts[d]));
}
for (const w of [0.3, 0.5]) {
  score(`whole*${1 - w} + max(t,a)*${w}`, (q, d) => (1 - w) * dot(qv[q], whole[d]) + w * Math.max(dot(qv[q], titles[d]), dot(qv[q], abstracts[d])));
}
