# The evaluation harness

Everything the ranking algorithm claims about itself is measured here, on papers
whose outcomes are now known, against the version it replaced.

```bash
npm run eval              # the primary benchmark (offline)
npm run eval:adversarial  # construct validity: is it measuring novelty at all
npm run eval:calibration  # does 70 mean the same thing everywhere
npm run eval:robustness   # degenerate input
npm run bench             # wall-clock and peak memory
```

All five run **offline** from `eval/fixtures/`. Only `node eval/fetch-fixtures.mjs`
touches the network, and it is meant to be run once.

## What the labels are, and why they cannot be reached

A ranker sees a paper exactly as the extension would: title, abstract, topics,
authorships, reference ids, publication date. The labels are built from things
that did not exist when the paper was published.

| Label | Built from | Used for |
|---|---|---|
| **Disruption (DI₅)** | Every paper that cited this one through 2026, and whether those citers also cited *its* references | Primary novelty label |
| **Forward citations** | Citation count to 2026, percentile-ranked within the paper's own (field, quarter) cohort | Secondary novelty label; sole combined-ordering label |
| **`type: review`** | OpenAlex document type | Known-negative class for consolidation |
| **Topic assignment** | OpenAlex `primary_topic` / `topics` | Relevance judgements |

**Disruption** is Bornmann & Tekles' `DI_nok`: of the papers citing this one, the
share that ignore its own references entirely. Work that displaces what came
before gets cited *instead of* its predecessors; work that extends them gets cited
*alongside* them. Requires at least 5 citations and 5 references, which 1,026 of
the 2,267 sampled papers meet. The `n_r` term of the full CD index is omitted —
collecting it would cost one request per reference per paper — so this is the
published simplification, not an invention of ours.

The forward window is fixed at 2021→2026 for every paper, so an older paper in the
sample is not quietly given longer to accumulate evidence.

**Leakage.** `outcome.*` is stripped in `lib/adapters.mjs::sanitize` before any
ranker sees a record. For the relevance benchmark the labels come from OpenAlex
topics, so `stripTopicEvidence` additionally removes every topic name *and* id —
the names because a ranker could match a query against them, the ids because a
ranker could cluster on them and propagate relevance between papers that share a
label. Relevance is therefore measured on text alone.

**`type: review` is deliberately not a feature.** It is what the benchmark defines
the negative class with, so feeding it back into the algorithm would make the
measurement congratulate itself. The consolidation detector reads structure
instead, and `eval:adversarial` separately checks that switching off the keyword
patterns barely moves the separation.

## Splits

Every paper is assigned to `tune` (40%) or `test` (60%) by a seeded hash of its
OpenAlex id (`fetch-fixtures.mjs::splitOf`). The assignment is deterministic, so
it is identical on any machine and across every run.

- **`tune`** fits things: the novelty fusion weights (`eval/tune-weights.mjs`),
  the consolidation coefficients (`eval/experiments/consolidation-fit.mjs`), the
  consolidation weight (`eval/experiments/consolidation-sweep.mjs`) and the
  discovery blend (`eval/experiments/discovery-blend.mjs`).
- **`test`** is what gets reported, and nothing is fitted on it.

The fitted weights are then shrunk toward a uniform prior over every signal that
correlates positively on its own, because coordinate ascent on 400 labelled papers
put large weights on two correlated statistics and zeroed the rest — a pattern
that moved between runs, which is fitting noise rather than signal.

## The baseline

`eval/baseline/` holds v1.0.1 exactly as it was, copied out of git history with
`git show main:src/shared/<file>.js`. It is never edited to make a comparison look
better, and `test/eval-baseline.test.js` fails if any of those files stops matching
what `main` contains.

## The LLM comparison

The premise behind this rewrite was that a directly-prompted LLM would out-rank
the algorithm. `eval/llm-baseline.mjs` measures that: a general-purpose model
reads title and abstract for 300 held-out papers and scores each for novelty, and
its ranking is scored against the same labels on the same papers. It runs once,
offline, through a local CLI; the verdicts are cached in
`fixtures/llm-novelty.json` and the model is never called at runtime.

## Regenerating the fixtures

```bash
node eval/fetch-fixtures.mjs
```

`fixtures/manifest.json` records the seed, the fields, the windows, the citation
horizon and the exact queries used, so a third party can rebuild the same corpus.
The run checkpoints to `fixtures/citers.jsonl` and can be resumed.

Note that OpenAlex now meters its free tier at **1,000 requests per day**, reset at
midnight UTC. The fetcher counts its own spend and stops cleanly when the
allowance runs out rather than thrashing on retries; a full build takes two days
of free allowance, or one request budget if you have a paid key.

## What these numbers are not

The benchmark covers four fields, three quarters of 2021, and English-language
records. Disruption is a citation-graph statistic, not a judgement of scientific
merit; a paper can be genuinely novel and never cited. Every metric here is a
comparison between rankers on one corpus, not a claim about research quality.
