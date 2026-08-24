import { loadCorpus } from "../lib/adapters.mjs";
import { seededRandom } from "../lib/metrics.mjs";
import { buildNoveltyModel, buildCohortCache, groupPeers, cohortFor } from "../../src/shared/novelty.js";

const corpus = loadCorpus();
const random = seededRandom(4242);
const grow = (w, n) => { const o = []; let i = 0; while (o.length < n) { const s = w[Math.floor(random() * w.length)]; o.push({ ...s, id: `${s.id}_b${i++}` }); } return o; };
const candidates = grow(corpus.candidates, 10000);
const peers = grow(corpus.peers, 3200);
const mb = () => Math.round(process.memoryUsage().rss / 1048576);
const hp = () => Math.round(process.memoryUsage().heapUsed / 1048576);
console.log(`corpus                 rss ${mb()}  heap ${hp()}`);

const model = buildNoveltyModel(candidates, peers, {});
if (global.gc) global.gc();
console.log(`model                  rss ${mb()}  heap ${hp()}`);

const groups = groupPeers(peers);
const cohort = cohortFor(candidates[0], model, groups);
console.log(`cohort peers: ${cohort.peers.length}`);

const cache = buildCohortCache(cohort.peers, model, "2021-01-01");
if (global.gc) global.gc();
console.log(`one cohort cache       rss ${mb()}  heap ${hp()}`);
console.log(`  postings entries ${cache.postings.peerIndex.length}`);
console.log(`  referenceIndex keys ${cache.referenceIndex.size}`);
console.log(`  coCitation pairs ${cache.coCitation.pairCount.size}, cited ${cache.coCitation.citedCount.size}`);
