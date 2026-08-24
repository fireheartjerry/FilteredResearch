// node eval/llm-relevance.mjs
//
// An independent reference point for how hard the relevance task actually is.
//
// The acceptance criteria set an absolute nDCG@20 bar before anything had been
// measured. Four hand-built method families and a learned reranker all land near
// 0.43, which is either a weak implementation or a bar above what the task
// permits -- and the difference matters, so it gets measured rather than argued.
//
// A directly-prompted LLM reranks the top 100 BM25 candidates for a sample of the
// same held-out queries, judged against the same labels. It never ships; it runs
// once, offline, and its verdicts are cached.

import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { writeFileSync, readFileSync, existsSync, mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { loadCorpus } from "./lib/adapters.mjs";
import { buildRelevanceQueries } from "./lib/labels.mjs";
import { ndcgAt, mean, seededRandom } from "./lib/metrics.mjs";
import { buildRelevanceIndex, expandQuery, withFeedback, scoreEntry } from "../src/shared/relevance.js";

const HERE = dirname(fileURLToPath(import.meta.url));
const OUTPUT = join(HERE, "fixtures", "llm-relevance.json");
const QUERIES = Number(process.env.FR_LLM_QUERIES || 10);
const DEPTH = 100;

function buildPrompt(query, papers) {
  const listing = papers
    .map((work, index) => `[${index + 1}] id=${work.id}\nTITLE: ${work.title}\nABSTRACT: ${String(work.abstract || "").slice(0, 420)}`)
    .join("\n\n");
  return `A researcher searched for: "${query}"

Below are ${papers.length} candidate papers. Score EACH for how well it matches that
search, from 0 (unrelated) to 100 (exactly this subject). Judge subject matter, not
quality. A paper can be a perfect match without containing the search words.

${listing}

Output ONLY a JSON object mapping each id to its integer score, nothing else.`;
}

function callModel(prompt) {
  const directory = mkdtempSync(join(tmpdir(), "fr-rel-"));
  const promptPath = join(directory, "prompt.txt");
  writeFileSync(promptPath, prompt);
  const output = execFileSync(
    "kimi",
    ["-p", `Read the file at ${promptPath.replace(/\\/g, "/")} and follow its instructions exactly. Output only the JSON object.`],
    { encoding: "utf8", timeout: 900_000, maxBuffer: 64 * 1024 * 1024 },
  );
  const match = output.match(/\{[\s\S]*\}/);
  if (!match) throw new Error(`no JSON in model output: ${output.slice(0, 200)}`);
  return JSON.parse(match[0]);
}

async function main() {
  const corpus = loadCorpus();
  const relevanceCorpus = loadCorpus({ forRelevance: true });
  const splitOf = new Map(corpus.rawCandidates.map((w) => [w.id, w.split]));
  const docs = relevanceCorpus.candidates.filter((w) => splitOf.get(w.id) === "test");
  const testIds = new Set(docs.map((w) => w.id));

  const allQueries = buildRelevanceQueries(corpus.rawCandidates, { minimumRelevant: 10, maximumQueries: 60 })
    .map((entry) => {
      const gains = new Map([...entry.gains].filter(([id]) => testIds.has(id)));
      return { ...entry, gains, relevantCount: [...gains.values()].filter((g) => g >= 2).length };
    })
    .filter((entry) => entry.relevantCount >= 6);

  const random = seededRandom(6060);
  const sample = [...allQueries].sort(() => (random() < 0.5 ? -1 : 1)).slice(0, QUERIES);
  const index = buildRelevanceIndex(docs, { space: null });

  const cache = existsSync(OUTPUT) ? JSON.parse(readFileSync(OUTPUT, "utf8")) : { provenance: null, byQuery: {} };
  let cliVersion = "unavailable";
  try {
    cliVersion = execFileSync("kimi", ["--version"], { encoding: "utf8", timeout: 60_000 }).trim().split(/\r?\n/)[0];
  } catch { /* recorded as unavailable */ }
  cache.provenance = {
    cli: "kimi",
    cliVersion,
    promptHash: createHash("sha256").update(buildPrompt("SAMPLE", [])).digest("hex").slice(0, 16),
    depth: DEPTH,
    scoredAt: new Date().toISOString(),
  };

  const ours = [];
  const theirs = [];
  for (const entry of sample) {
    const expanded = withFeedback(index, expandQuery(entry.query, index));
    const ranked = index.entries
      .map((item) => ({ item, raw: scoreEntry(item, expanded, index).raw }))
      .sort((a, b) => b.raw - a.raw || String(a.item.work.id).localeCompare(String(b.item.work.id)));
    const pool = ranked.slice(0, DEPTH).map((x) => x.item.work);
    const gainOf = (work) => entry.gains.get(work.id) || 0;

    // Our ranking, restricted to the same pool the model is given, so the two are
    // scored on exactly the same candidate set.
    ours.push(ndcgAt(pool, gainOf, 20));

    if (!cache.byQuery[entry.query]) {
      try {
        cache.byQuery[entry.query] = callModel(buildPrompt(entry.query, pool));
        writeFileSync(OUTPUT, `${JSON.stringify(cache, null, 1)}\n`);
        console.log(`  scored "${entry.query.slice(0, 50)}"`);
      } catch (error) {
        console.log(`  failed "${entry.query.slice(0, 50)}": ${String(error.message).slice(0, 120)}`);
        continue;
      }
    }
    const judged = cache.byQuery[entry.query];
    const modelRanked = [...pool].sort((a, b) => (Number(judged[b.id]) || 0) - (Number(judged[a.id]) || 0));
    theirs.push(ndcgAt(modelRanked, gainOf, 20));
  }

  const summary = {
    queries: theirs.length,
    depth: DEPTH,
    algorithm_ndcg_at_20: Number(mean(ours.slice(0, theirs.length)).toFixed(4)),
    llm_ndcg_at_20: Number(mean(theirs).toFixed(4)),
    note: "Both ranked the identical top-100 BM25 pool for the identical queries, scored against the identical labels.",
  };
  cache.summary = summary;
  writeFileSync(OUTPUT, `${JSON.stringify(cache, null, 1)}\n`);
  console.log(`\n${JSON.stringify(summary, null, 2)}`);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
