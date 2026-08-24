// Builds the cached benchmark. Run once, online:  node eval/fetch-fixtures.mjs
// Everything downstream (npm run eval, eval:adversarial, eval:calibration,
// eval:robustness, bench) then runs fully offline against eval/fixtures/.
//
// Design rule that keeps the benchmark honest: a record's `outcome` block holds
// only post-publication information (forward citations, who cited it, and
// whether those citers also cited its references). No ranker is ever handed it.
//
// Both network stages checkpoint to disk. The citation stage is roughly a
// thousand requests against a rate-limited public API, so a 429 that kills the
// process must cost one request, not twenty minutes.

import { gzipSync, gunzipSync } from "node:zlib";
import { mkdirSync, writeFileSync, readFileSync, existsSync, appendFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { oaGet, mapLimit, shortId, projectWork, WORK_SELECT, BUDGET, BudgetExhausted } from "./lib/oa.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const FIXTURES = join(HERE, "fixtures");
const CITERS_LOG = join(FIXTURES, "citers.jsonl");

const SEED = 20210101;
const FIELDS = [
  { id: "17", name: "Computer Science" },
  { id: "27", name: "Medicine" },
  { id: "31", name: "Physics and Astronomy" },
  { id: "25", name: "Materials Science" },
];
const WINDOWS = [
  { key: "2021Q1", from: "2021-01-01", to: "2021-03-31" },
  { key: "2021Q2", from: "2021-04-01", to: "2021-06-30" },
  { key: "2021Q3", from: "2021-07-01", to: "2021-09-30" },
];
const PEER_WINDOW = { from: "2018-01-01", to: "2020-12-31" };
// A fixed five-year forward window, identical for every focal paper, so the
// disruption label is not quietly generous to the older ones.
const CITATION_HORIZON = "2026-01-01";

const CANDIDATES_PER_CELL = 170;
const REVIEWS_PER_FIELD = 70;
const PEERS_PER_FIELD = 900;
const MAX_CITERS = 200;

const queries_log = [];

function write(name, value) {
  mkdirSync(FIXTURES, { recursive: true });
  const path = join(FIXTURES, `${name}.json.gz`);
  writeFileSync(path, gzipSync(Buffer.from(JSON.stringify(value)), { level: 9 }));
  return path;
}

function readCached(name) {
  const path = join(FIXTURES, `${name}.json.gz`);
  return existsSync(path) ? JSON.parse(gunzipSync(readFileSync(path)).toString()) : null;
}

async function sampled(filter, size, select = WORK_SELECT) {
  const out = [];
  const perPage = 200;
  for (let page = 1; out.length < size; page += 1) {
    const params = { filter, sample: size, seed: SEED, per_page: Math.min(perPage, size), page, select };
    queries_log.push({ endpoint: "works", ...params });
    const payload = await oaGet("works", params);
    const results = payload.results || [];
    out.push(...results);
    if (results.length < Math.min(perPage, size) || page >= 25) break;
  }
  return out.slice(0, size);
}

function usable(work) {
  return Boolean(work.title) && Boolean(work.publicationDate);
}

// A stable, seeded assignment so the tuning half and the reported half never
// drift between runs or between machines.
function splitOf(id) {
  let hash = 2166136261;
  for (let index = 0; index < id.length; index += 1) {
    hash ^= id.charCodeAt(index);
    hash = Math.imul(hash, 16777619) >>> 0;
  }
  return hash % 100 < 40 ? "tune" : "test";
}

async function fetchWorksStage() {
  const cached = readCached("works-stage");
  if (cached) {
    console.log("Reusing cached works stage.");
    return cached;
  }

  console.log("Fetching candidates...");
  const candidates = [];
  for (const field of FIELDS) {
    for (const window of WINDOWS) {
      const filter = `primary_topic.field.id:fields/${field.id},from_publication_date:${window.from},to_publication_date:${window.to},type:article|preprint|review,language:en`;
      const raw = await sampled(filter, CANDIDATES_PER_CELL);
      for (const payload of raw) {
        const work = projectWork(payload);
        if (!usable(work)) continue;
        candidates.push({ ...work, benchField: field.id, benchWindow: window.key, split: splitOf(work.id) });
      }
      console.log(`  ${field.name} ${window.key}: ${raw.length}`);
    }
  }

  console.log("Fetching review-type records (known-negative class for novelty)...");
  const seen = new Set(candidates.map((work) => work.id));
  for (const field of FIELDS) {
    const filter = `primary_topic.field.id:fields/${field.id},from_publication_date:2021-01-01,to_publication_date:2021-09-30,type:review,language:en`;
    const raw = await sampled(filter, REVIEWS_PER_FIELD);
    for (const payload of raw) {
      const work = projectWork(payload);
      if (!usable(work) || seen.has(work.id)) continue;
      seen.add(work.id);
      candidates.push({ ...work, benchField: field.id, benchWindow: "2021Q1", split: splitOf(work.id), oversampledReview: true });
    }
    console.log(`  ${field.name} reviews: ${raw.length}`);
  }

  console.log("Fetching peer corpus (pre-2021, the only history a ranker may see)...");
  const peers = [];
  for (const field of FIELDS) {
    const filter = `primary_topic.field.id:fields/${field.id},from_publication_date:${PEER_WINDOW.from},to_publication_date:${PEER_WINDOW.to},type:article|preprint|review,language:en`;
    const raw = await sampled(filter, PEERS_PER_FIELD);
    for (const payload of raw) {
      const work = projectWork(payload);
      if (!usable(work)) continue;
      peers.push({ ...work, benchField: field.id, isBaseline: true });
    }
    console.log(`  ${field.name} peers: ${raw.length}`);
  }

  console.log("Fetching author profiles...");
  const author_ids = [...new Set(
    candidates.flatMap((work) => (work.authorships || []).slice(0, 4).map((authorship) => authorship.authorId)).filter(Boolean),
  )];
  const batches = [];
  for (let index = 0; index < author_ids.length; index += 50) batches.push(author_ids.slice(index, index + 50));
  const authors = [];
  await mapLimit(batches, 4, async (batch) => {
    const payload = await oaGet("authors", {
      filter: `openalex_id:${batch.join("|")}`,
      per_page: 50,
      select: "id,display_name,orcid,works_count,cited_by_count,summary_stats,last_known_institutions",
    });
    for (const entry of payload.results || []) {
      const stats = entry.summary_stats || {};
      authors.push({
        id: shortId(entry.id),
        name: entry.display_name || "Unknown",
        orcid: entry.orcid || null,
        worksCount: Number(entry.works_count || 0),
        citedByCount: Number(entry.cited_by_count || 0),
        hIndex: Number(stats.h_index || 0),
        i10Index: Number(stats.i10_index || 0),
        twoYearMeanCitedness: Number(stats["2yr_mean_citedness"] || 0),
        lastInstitution: entry.last_known_institutions?.[0]?.display_name || null,
      });
    }
  });
  console.log(`  ${authors.length} author profiles`);

  const stage = { candidates, peers, authors, queries: queries_log.slice(0, 60) };
  write("works-stage", stage);
  return stage;
}

async function fetchCiters(workId) {
  const payload = await oaGet("works", {
    filter: `cites:${workId},to_publication_date:${CITATION_HORIZON}`,
    per_page: MAX_CITERS,
    select: "id,referenced_works,publication_date",
  });
  return (payload.results || []).map((citer) => ({
    id: shortId(citer.id),
    referencedWorks: (citer.referenced_works || []).map(shortId),
  }));
}

// Bornmann & Tekles DI_nok: of the papers citing this one, the share that ignore
// its intellectual predecessors entirely. Work that displaces what came before
// gets cited *instead of* its references; work that extends them gets cited
// *alongside* them. n_r is omitted deliberately -- collecting it would cost a
// request per reference per paper, and the nok variant is the published,
// citable simplification rather than an invention of ours.
function disruptionIndex(references, citers) {
  const known = new Set(references || []);
  if (!known.size || citers.length < 5) return null;
  let forward = 0;
  let backward = 0;
  for (const citer of citers) {
    if ((citer.referencedWorks || []).some((id) => known.has(id))) backward += 1;
    else forward += 1;
  }
  const total = forward + backward;
  if (!total) return null;
  return { di: (forward - backward) / total, citersConsidered: total, forward, backward };
}

function loadCiterCheckpoint() {
  if (!existsSync(CITERS_LOG)) return new Map();
  const entries = new Map();
  for (const line of readFileSync(CITERS_LOG, "utf8").split("\n")) {
    if (!line.trim()) continue;
    try {
      const parsed = JSON.parse(line);
      entries.set(parsed.id, parsed.disruption);
    } catch {
      // A half-written final line from a killed process is simply refetched.
    }
  }
  return entries;
}

async function main() {
  mkdirSync(FIXTURES, { recursive: true });
  const { candidates, peers, authors, queries } = await fetchWorksStage();

  console.log("Fetching citation outcomes for disruption labels...");
  const labelable = candidates.filter(
    (work) => work.outcome.citedByCount >= 5 && (work.referencedWorks || []).length >= 5,
  );
  const checkpoint = loadCiterCheckpoint();
  const pending = labelable.filter((work) => !checkpoint.has(work.id));
  console.log(`  ${labelable.length} of ${candidates.length} candidates qualify; ${checkpoint.size} already cached, ${pending.length} to fetch`);

  let done = 0;
  await mapLimit(pending, 2, async (work) => {
    if (BUDGET.exhausted) return;
    try {
      const citers = await fetchCiters(work.id);
      const measure = disruptionIndex(work.referencedWorks, citers);
      checkpoint.set(work.id, measure);
      appendFileSync(CITERS_LOG, `${JSON.stringify({ id: work.id, disruption: measure })}\n`);
      done += 1;
      if (done % 50 === 0) console.log(`  ${done}/${pending.length} (budget ${BUDGET.spent}/${BUDGET.max})`);
    } catch (error) {
      // A spent daily allowance is an expected stopping point, not a failure:
      // everything fetched so far is already on disk.
      if (!(error instanceof BudgetExhausted)) throw error;
    }
  });
  if (BUDGET.exhausted) {
    console.log(`  Stopped early: OpenAlex free budget spent. ${checkpoint.size} labels cached; rerun after midnight UTC to top up.`);
  }

  for (const work of candidates) {
    const measure = checkpoint.get(work.id);
    if (measure) work.outcome.disruption = measure;
  }

  const manifest = {
    builtAt: new Date().toISOString(),
    seed: SEED,
    fields: FIELDS,
    windows: WINDOWS,
    peerWindow: PEER_WINDOW,
    citationHorizon: CITATION_HORIZON,
    counts: {
      candidates: candidates.length,
      withDisruption: candidates.filter((work) => work.outcome.disruption).length,
      reviews: candidates.filter((work) => work.workType === "review").length,
      withAbstract: candidates.filter((work) => work.abstract).length,
      withReferences: candidates.filter((work) => (work.referencedWorks || []).length >= 5).length,
      peers: peers.length,
      authors: authors.length,
      tune: candidates.filter((work) => work.split === "tune").length,
      test: candidates.filter((work) => work.split === "test").length,
    },
    queries: queries || queries_log.slice(0, 60),
    note: "outcome.* fields are post-publication only and are never passed to a ranker.",
  };

  write("candidates", candidates);
  write("peers", peers);
  write("authors", authors);
  writeFileSync(join(FIXTURES, "manifest.json"), `${JSON.stringify(manifest, null, 2)}\n`);
  console.log("\nWrote fixtures:", JSON.stringify(manifest.counts, null, 2));
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
