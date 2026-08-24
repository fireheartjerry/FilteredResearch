import { loadCorpus } from "../lib/adapters.mjs";
import { seededRandom } from "../lib/metrics.mjs";
import { contentTerms, textOf, lexiconFrom, bm25Weights } from "../../src/shared/text.js";
import { buildSemanticSpace, documentVector, tokensForSpace } from "../../src/shared/semantic.js";

const corpus = loadCorpus();
const random = seededRandom(4242);
const grow = (works, target) => { const out = []; let n = 0; while (out.length < target) { const s = works[Math.floor(random() * works.length)]; out.push({ ...s, id: `${s.id}_b${n++}` }); } return out; };
const candidates = grow(corpus.candidates, 10000);
const peers = grow(corpus.peers, 3200);
const all = [...peers, ...candidates];
const mb = () => Math.round(process.memoryUsage().rss / 1048576);
const heap = () => Math.round(process.memoryUsage().heapUsed / 1048576);
console.log(`after corpus build            rss ${mb()} MB  heap ${heap()} MB`);

const documentFrequency = new Map();
let total = 0;
for (const w of all) { const t = contentTerms(textOf(w)); total += t.length; for (const term of new Set(t)) documentFrequency.set(term, (documentFrequency.get(term) || 0) + 1); }
console.log(`after df pass (${documentFrequency.size} terms)   rss ${mb()} MB  heap ${heap()} MB`);

const lexicon = lexiconFrom(documentFrequency, all.length);
const avg = total / all.length;
console.log(`after lexicon (${lexicon.ids.size} kept)      rss ${mb()} MB  heap ${heap()} MB`);

const ordered = [...all].sort((a, b) => String(a.id).localeCompare(String(b.id)));
const space = buildSemanticSpace((function* () { for (const w of ordered) yield tokensForSpace(textOf(w)); })(), lexicon, { documentCount: ordered.length });
console.log(`after semantic space (${space.vocabulary.size}) rss ${mb()} MB  heap ${heap()} MB`);

const lex = new Map(); const sem = new Map();
for (const w of ordered) { const t = contentTerms(textOf(w)); lex.set(w, bm25Weights(t, lexicon, avg)); sem.set(w, documentVector(t, space, lexicon, avg)); }
console.log(`after vectors                 rss ${mb()} MB  heap ${heap()} MB`);
