// node eval/llm-baseline.mjs
//
// The user's opening claim was that a directly-prompted LLM would out-rank this
// algorithm. That is a measurable claim, so it gets measured: the same held-out
// papers, the same labels, the same metrics.
//
// The LLM never ships. It runs once, offline, through a local CLI, and its
// verdicts are cached to eval/fixtures/llm-novelty.json so `npm run eval` can
// report the comparison without a network call or a penny of runtime cost.

import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { writeFileSync, readFileSync, existsSync, mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { loadCorpus } from "./lib/adapters.mjs";
import { buildLabels } from "./lib/labels.mjs";
import { seededRandom } from "./lib/metrics.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const OUTPUT = join(HERE, "fixtures", "llm-novelty.json");
const SAMPLE = Number(process.env.FR_LLM_SAMPLE || 300);
const BATCH = 25;

function buildPrompt(batch) {
  const papers = batch
    .map((work, index) => `[${index + 1}] id=${work.id}\nTITLE: ${work.title}\nABSTRACT: ${String(work.abstract || "").slice(0, 900)}`)
    .join("\n\n");
  return `You are an expert research scientist screening recent papers.

For EACH paper below, judge how likely it is to be genuinely NOVEL and DISRUPTIVE -- work that
opens a new direction and makes prior approaches less relevant -- as opposed to incremental,
derivative, or consolidating work (surveys, benchmarks, minor variants, applications of an
established method to one more dataset).

Score each from 0 to 100. Use the whole range. Judge only from the title and abstract.
Do not reward good writing, prestige, or apparent importance -- only novelty of the idea.

${papers}

Output ONLY a JSON object mapping each id to its integer score, nothing else. Example:
{"W123": 42, "W456": 88}`;
}

function callModel(prompt) {
  const directory = mkdtempSync(join(tmpdir(), "fr-llm-"));
  const promptPath = join(directory, "prompt.txt");
  writeFileSync(promptPath, prompt);
  // The prompt goes by file path, never as an argument: it is long, it contains
  // quotes and braces, and Windows shells corrupt both without saying so.
  const output = execFileSync(
    "kimi",
    ["-p", `Read the file at ${promptPath.replace(/\\/g, "/")} and follow its instructions exactly. Output only the JSON object.`],
    { encoding: "utf8", timeout: 600_000, maxBuffer: 32 * 1024 * 1024 },
  );
  const match = output.match(/\{[\s\S]*\}/);
  if (!match) throw new Error(`no JSON in model output: ${output.slice(0, 300)}`);
  return JSON.parse(match[0]);
}

async function main() {
  const corpus = loadCorpus();
  const labels = buildLabels(corpus.rawCandidates);
  const eligible = corpus.rawCandidates.filter(
    (work) => work.split === "test" && work.abstract && labels.get(work.id)?.noveltyLabel !== null,
  );
  const random = seededRandom(20260824);
  const pool = [...eligible].sort((left, right) => (random() < 0.5 ? -1 : 1)).slice(0, SAMPLE);
  console.log(`Scoring ${pool.length} held-out papers with a directly-prompted LLM, ${BATCH} at a time...`);

  // Provenance. Without it the comparison is not reproducible by anyone,
  // including whoever runs this next month. It identifies the CLI and the prompt,
  // not a pinned model version -- the CLI resolves its own model, and that is a
  // real limitation of the comparison rather than something this can paper over.
  let cliVersion = "unknown";
  try {
    cliVersion = execFileSync("kimi", ["--version"], { encoding: "utf8", timeout: 60_000 }).trim().split(/\r?\n/)[0];
  } catch {
    cliVersion = "unavailable";
  }
  const promptHash = createHash("sha256").update(buildPrompt([{ id: "SAMPLE", title: "", abstract: "" }])).digest("hex").slice(0, 16);
  const provenance = { cli: "kimi", cliVersion, promptHash, batchSize: BATCH, scoredAt: new Date().toISOString() };

  const existing = existsSync(OUTPUT) ? JSON.parse(readFileSync(OUTPUT, "utf8")) : {};
  const scores = existing.scores || existing;
  delete scores.provenance;
  const pending = pool.filter((work) => scores[work.id] === undefined);
  console.log(`  ${Object.keys(scores).length} already cached, ${pending.length} to score`);

  for (let start = 0; start < pending.length; start += BATCH) {
    const batch = pending.slice(start, start + BATCH);
    try {
      const result = callModel(buildPrompt(batch));
      let accepted = 0;
      for (const work of batch) {
        const value = Number(result[work.id]);
        if (Number.isFinite(value)) {
          scores[work.id] = Math.max(0, Math.min(100, value));
          accepted += 1;
        }
      }
      writeFileSync(OUTPUT, `${JSON.stringify({ provenance, scores }, null, 1)}\n`);
      console.log(`  batch ${Math.floor(start / BATCH) + 1}: ${accepted}/${batch.length} scored (${Object.keys(scores).length} total)`);
    } catch (error) {
      console.log(`  batch ${Math.floor(start / BATCH) + 1} failed: ${String(error.message).slice(0, 160)}`);
    }
  }
  writeFileSync(OUTPUT, `${JSON.stringify({ provenance, scores }, null, 1)}\n`);
  console.log(`\nWrote ${OUTPUT} with ${Object.keys(scores).length} LLM judgements from ${cliVersion}.`);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
