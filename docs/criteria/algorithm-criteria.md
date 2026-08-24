# Filtered Research — ranking algorithm criteria

**Status:** APPROVED by the user. Weights shifted toward novelty (C1, C2) and real-world usefulness (C3, C5) at the user's request; disruption (CD5) is the primary label, forward citations secondary.
**Target:** > 95 / 100. **Baseline (v1.0.1, unmodified):** measured in Round 0.
**Budget:** up to 6 rounds; stop early on 95+; stop and report on plateau (< 2 points movement two rounds running).

## What is being improved

The ranking pipeline that decides which recent papers surface and in what order:
`src/shared/scoring.js`, `src/shared/filters.js`, `src/shared/ranking.js`,
`src/shared/prominence.js`, and the OpenAlex retrieval/enrichment in
`src/shared/openalex.js` + `src/background/service-worker.js`.

The user's stated problem: a directly-prompted LLM would out-rank this algorithm
today. The goal is an algorithm that does not lose that comparison.

## Constraints (these outrank any reviewer finding)

- **Zero runtime cost to the user.** No paid API, no hosted-LLM call at runtime,
  OpenAlex free tier only. Offline LLM spend by the developer — to build the
  benchmark, generate labels, or distill weights that then ship — is allowed.
- **Local-first.** No data leaves the device beyond the OpenAlex requests that
  already exist. No new host permissions, no remote executable code (MV3).
- **UI files are near-off-limits.** `src/sidepanel/`, `src/options/`,
  `src/notifications/` may take only additive edits needed to display new
  evidence. No redesign, no behavioural rewrite.
- **Staging only.** All work lands on `staging/algorithm-rewrite`. Never merged
  to `main`, never pushed without explicit permission.
- The benchmark must run **offline** from cached fixtures, so any scorer can run it.

## Out of scope

Visual design, the service worker's lifecycle/progress machinery, the
notification inbox, Web Store listing copy, and anything in `COMPLIANCE.md` /
`PRIVACY.md` / `SECURITY.md`.

## Gates — binary, reported beside the score, never averaged into it

| # | Gate | check: |
|---|---|---|
| G1 | Existing + new unit tests pass | `npm test` exits 0 |
| G2 | Static syntax check passes | `npm run check` exits 0 |
| G3 | Extension still packages | `npm run package` exits 0 and `dist/filteredresearch-extension/manifest.json` exists |
| G4 | Benchmark runs offline | set `FR_EVAL_OFFLINE=1`, run `npm run eval`; exits 0 and writes `eval/report.json` |
| G5 | No new capabilities | `git diff main -- manifest.json` adds no `permissions`, `host_permissions`, or `content_scripts`; no new remote endpoint outside `api.openalex.org` appears in `src/` |
| G6 | Not catastrophically slower | `npm run bench` reports 10,000-candidate wall-clock at **at most 3x** the v1.0.1 baseline recorded in Round 0 |
| G7 | UI untouched beyond additive | `git diff main --stat -- src/sidepanel src/options src/notifications` shows only additions for new evidence fields |
| G8 | Staging discipline | current branch is not `main`; nothing pushed to any remote |

A failed gate makes the round a net loss regardless of points.

## Criteria — 100 points

### C1 — Novelty ranking quality against ground truth · 22 pts

Does the novelty score rank genuinely novel work above derivative work, measured
on papers whose outcome is now known?

**How to check it:** run `npm run eval`. Read `eval/report.json` →
`novelty.ndcg_at_50`, `novelty.spearman`, `novelty.auc_disruptive_vs_derivative`
on the held-out benchmark (papers from a past window, scored using only metadata
that existed at publication time, evaluated against later-observed outcomes: the
CD5 disruption index, field-year-normalised forward citations, and OpenAlex
`type: review` as a known-negative class). The report also carries `baseline.*`
for the unmodified v1.0.1 algorithm on the identical split.

- 22 pts: nDCG@50 >= 0.75 **and** AUC >= 0.80 **and** at least +0.20 absolute nDCG over baseline
- 16 pts: nDCG@50 >= 0.65, AUC >= 0.72, at least +0.12 over baseline
- 11 pts: nDCG@50 >= 0.55 and clearly beats baseline
- 5 pts: measurably beats baseline but under 0.55
- 0 pts: no better than baseline, or no benchmark exists

### C2 — Novelty construct validity · 14 pts

The score must measure novelty, not vocabulary weirdness. Today's
`distinctivenessOf` is mean IDF over unique terms, which rewards typos, chemical
names, transliterations, and obscure jargon.

**How to check it:** run `npm run eval:adversarial`. For each probe:

1. **Junk-token injection** — append 30 rare nonsense tokens to an abstract. Novelty must rise by **< 3 points**.
2. **Paraphrase invariance** — restate a paper with synonyms, no new ideas. Novelty must move by **< 8 points**.
3. **True-duplicate detection** — a near-copy of an existing peer must land in the **bottom 10%**.
4. **Survey handling** — real OpenAlex reviews rank below matched research articles by **>= 20 points median**, without a keyword regex doing the work: delete the regex list, re-run, AUC must drop by **< 0.05**.
5. **Thin records** — non-English and short-abstract records must not receive inflated novelty.

14 pts = all five pass. Deduct 3 per failed probe.

### C3 — Relevance ranking quality · 15 pts

Today `matchesResearchFilters` returns a boolean and interests rank nothing,
though `SPEC.md` and `README.md` both claim they do.

**How to check it:** `eval/report.json` → `relevance.ndcg_at_20`,
`relevance.mrr`, `relevance.recall_at_100` over a labelled query set (queries
from OpenAlex topic names and real researcher phrasings; relevant = papers whose
own `primary_topic`/`topics` match, held out from the ranker). Plus a
qualitative set — `RAG`, `LLM agents`, `diffusion`, `protein folding`, `causal
inference`, `sparse autoencoder` — each with a hand-checked top-10.

- 15 pts: a graded relevance score exists, is used in ordering, nDCG@20 >= 0.70, all six qualitative top-10s defensible
- 10 pts: graded score exists, nDCG@20 >= 0.55
- 5 pts: graded score exists but weak, or not wired into ordering
- 0 pts: still boolean

### C4 — Authorship and combined ranking · 5 pts

**How to check it:** `eval/report.json` → `authorship.spearman` against
field-normalised author standing, and `combined.ndcg_at_50` for the score that
actually orders the feed. The combined ranker must beat both of its parts alone
on `combined.ndcg_at_50`. The h-index blend must be field-normalised (h = 40
means different things in medicine and mathematics):
`authorship.field_bias_gap` must be **< 0.15**.

### C5 — Calibration and cross-field stability · 12 pts

A 70 must mean the same thing in AI and in materials science, this month and last.

**How to check it:** `npm run eval:calibration`, over >= 4 fields x >= 3 time windows:

- decile occupancy: each 10-point band holds 6–14% of papers, **no band above 20%**
- cross-field drift: **mean drift < 6 points**, sigma ratio within 0.7–1.4
- corpus-size stability: doubling the corpus moves a fixed paper by **< 5 points**
- rank stability under a 10% random resample: Spearman **>= 0.90**

12 pts = all four. Deduct 3 each.

### C6 — Robustness on degenerate input · 8 pts

**How to check it:** `npm run eval:robustness` over the adversarial fixture set —
empty abstract, title-only, zero peers, one peer, all-identical corpus, a
12,000-character abstract, malformed dates, missing `topics`, missing
`authorships`, non-UTF8 escapes, a paper that is its own peer, and a corpus where
every paper shares one publication date. Requirement: **no throw, no NaN, no
undefined score, nothing outside 1–100**, and every case emits evidence
explaining its low confidence. 8 pts = clean sweep; 0 pts = any throw or NaN.

### C7 — Performance and memory · 7 pts

**How to check it:** `npm run bench` on 10,000 candidates x 320 peers, reporting
wall-clock and peak RSS against the v1.0.1 numbers from Round 0.

- 7 pts: at most 1.25x baseline wall-clock and 1.5x peak memory
- 5 pts: at most 1.75x wall-clock
- 3 pts: at most 2.5x wall-clock
- 0 pts: above 3x (also fails G6)

### C8 — Explainability · 7 pts

Every score must stay defensible to a researcher looking at one card.

**How to check it:** for 10 randomly sampled scored papers, the
`noveltyEvidence` / `relevanceEvidence` objects must let a reader answer, without
reading source: *what was this compared against, which signal moved the score,
and by how much?* That requires a per-signal contribution breakdown (signal name
→ signed points contributed), not just raw intermediates. 7 pts = all ten
defensible; deduct 1 each (floor 0).

### C9 — Benchmark integrity · 5 pts

The benchmark produces every other number, so it must not be gameable.

**How to check it:** confirm in `eval/README.md` and in code that (a) labels come
from data the algorithm never sees — post-publication citation outcomes, not the
text it scores; (b) the tuning split and the reported split are disjoint, seeded,
and fixed; (c) the v1.0.1 baseline is the real original code imported unmodified
from git history, not a reimplementation; (d) fixtures are cached alongside the
query that produced them so a third party can regenerate them. 5 pts = all four;
deduct 1.25 each.

### C10 — Integration and code quality · 3 pts

**How to check it:** the new pipeline is actually called by
`src/background/service-worker.js` rather than sitting dead beside the old path;
the old implementation is removed, not left behind; `SCORING_VERSION` is bumped
so stored scores invalidate; unit tests cover the new signals; no function runs
long without cause; naming follows the repo's existing conventions.

### C11 — Documentation truthfulness · 2 pts

**How to check it:** read `docs/SCORING.md` and `SPEC.md` against the code. Every
formula, weight, and claim must match, and claims the algorithm cannot support
must be removed. Note the current drift: the docs claim "78% cosine idea-distance
/ 14% unseen phrases / 8% cross-field", weights that appear nowhere in
`scoring.js`. That class of drift scores 0.

## Calibration check for the scorer

Unmodified v1.0.1 should score **low** on C1, C2, C3, C9 (no benchmark exists,
relevance is boolean, distinctiveness is vocabulary noise) and **well** on C7 and
partially on C6 (it is fast, and it does guard NaN). A scorer that returns high
marks on C1 or C3 for the unmodified code is miscalibrated.

## Score log

| Round | Total | Gates | Note |
|---|---|---|---|
| 0 (baseline) | **35 / 100** | all pass | Unmodified v1.0.1. Scored by an independent `kimi -p` process against a clean `main` worktree. Novelty rank-correlation with real disruption: -0.0485. |
