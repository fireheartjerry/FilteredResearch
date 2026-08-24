# Round log — ranking algorithm

Criteria: [algorithm-criteria.md](algorithm-criteria.md).
Reviewer and scorer both run as independent `kimi -p` processes, meeting the work
cold. Scoring never sees the reviewer's findings, the changelog, or any prior
score.

---

## Round 0 — baseline 35/100

**Scorer:** independent, against a clean `main` worktree.

The unmodified v1.0.1 algorithm, measured on a benchmark built for the purpose:
2,267 papers published in 2021 across four fields, 1,026 of them carrying a
disruption label computed from who cited them through 2026.

| | v1.0.1 |
|---|---|
| novelty nDCG@50 | 0.292 |
| AUC vs derivative articles | **0.489** (below chance) |
| Spearman vs disruption | **−0.057** |
| relevance nDCG@20 | 0.025 |
| construct-validity probes | 2/5 |
| calibration checks | 2/4 |

The headline finding: the novelty score had no relationship with what actually
disrupted its field, and against derivative research it was worse than a coin
flip. Interest phrases ranked nothing despite two documents saying they did.

---

## Round 1 — 35 → 73.75 (+38.75)

**Reviewer:** independent, blind (no criteria). Returned 12 findings.
**Accepted:** 8 · **Deferred:** 3 · **Rejected:** 1

### Built

Novelty rewritten as ten signals fused over the reference graph, the text and the
paper's own shape, each ranked inside its field cohort. Relevance rewritten as
BM25 with acronym, phrase and corpus-derived expansion, and wired into the feed
ordering. A benchmark, an adversarial suite, a calibration suite, a robustness
suite and a performance bench, all offline.

### The review's two critical findings, both correct

- **Scores depended on input order.** Anchor selection strode through the peer
  array and the shortlist took stratified picks from it, both in arrival order.
  Reshuffling the corpus changed 2,247 of 2,267 scores by up to 39 points — while
  `SPEC.md` claimed order-independence and a unit test "verified" it by shuffling
  only the candidates. Fixed by canonical ordering; the test now shuffles both.
- **Peer anchors were measured against an index containing themselves.** Anchors
  seed the reference-signal distributions, and every anchor's own bibliography
  was inside the co-citation and coupling indexes it was scored against —
  `strongestCoupling` was exactly 1.0 for 100% of them. Those signals carry three
  quarters of the fusion weight. Now measured leave-one-out.

### Also accepted

- Co-citation index scoped to the paper's own field rather than the whole corpus.
- AUC decomposed: pooling reviews into the negative class flattered a
  discriminator explicitly built to catch reviews. The honest number is against
  derivative *research*.
- Oversampled reviews were stamped with the wrong publication quarter, putting
  them in the wrong citation cohort.
- Fusion weights now emitted by the tuner rather than hand-transcribed. This
  removed `crowding`, whose measured correlation with disruption is negative.
- Emerging terminology always requires two independent users, closing a hole that
  only opened on small first-run corpora.
- The LLM baseline carries provenance and a usability flag.

### Rejected

- *"Score papers one per prompt to fix the LLM baseline."* The baseline's problem
  is not batching — its Spearman is −0.228, so it is a broken instrument. Fixed
  by flagging it as unusable rather than by spending 300 more calls to obtain a
  better-measured broken instrument.

### Deferred

- Per-field lexicon and semantic space — attempted in round 2, measured worse.
- One-per-prompt LLM rescoring — cost without a decision riding on it.
- Per-field predictive-validity reporting — worth doing, not attempted.

---

## Round 2 — 73.75 → 79.25 (+5.5)

Driven by the round-1 scorecard rather than a fresh review.

- **Relevance +0.039 nDCG@20** from pseudo-relevance feedback. Semantic query
  expansion and a hybrid lexical-vector score were both tried and both made it
  worse; feedback was the only method of three that helped.
- **The qualitative query panel was unanswerable.** It asked a 2021 corpus about
  retrieval-augmented generation, LLM agents and sparse autoencoders — zero
  matching papers, so its top-10 lists were noise by construction. Replaced with
  queries the corpus can answer, and the report now prints per-query match counts.
- **Documented weights pinned to the code by a test.** They had gone stale again,
  which is the same class of drift this rewrite was partly meant to fix.
- **Reverted after measuring:** per-field vocabulary history (did not improve the
  stability it targeted, and cost 0.037 nDCG).

---

## Round 3 — final

- Relevance corpus stripped of taxonomy ids, closing a leakage route the ranker
  was not using but could have.
- Posting lists flattened to compressed sparse row.
- `SPEC.md` signal count corrected and pinned.
- **Reverted after measuring:** fitting the fusion against derivative articles
  alone. More principled, better on tune (AUC 0.704), and 0.018 worse held out —
  too few negatives for the estimate to be stable.

---

## Where it stops, and why

Two criteria were amended mid-loop, both because they set absolute thresholds
before anything had been measured:

- **C1** originally required nDCG@50 ≥ 0.75. The achievable range on this
  benchmark is under 0.5 for any ranker, so every result scored 5 and the
  criterion could not distinguish good from bad. Re-anchored on reference points
  that exist independently of this implementation.
- **C4** originally required the combined ordering to beat both of its parts on a
  citation label. A sweep over every blend showed that impossible — authorship
  predicts citations by construction, so adding novelty can only add noise. The
  criterion as written rewarded turning the feed into a prestige list.

**C3 has the same defect and was deliberately left alone.** Its 10-point anchor
requires relevance nDCG@20 ≥ 0.55; four method families were tried (tuned BM25,
semantic query expansion, hybrid lexical-vector scoring, pseudo-relevance
feedback) and the ceiling is ~0.42. Amending a third criterion downward, in my
own favour, after the number it produces had become the thing being optimised, is
exactly the move that would hollow out the score. The eight points stay lost and
the reason is recorded here instead.

That caps the reachable total near 90 rather than the 95 originally targeted. The
remaining gap is a property of the rubric, not of the work.
