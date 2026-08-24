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

## Rounds 4–7 — 81 → 77 → 79 → 81

Rounds 4 and 5 are the useful ones to read, because the score went *down* while
the work only improved. Both drops were ambiguities in criteria I had written:

- **C2** hinged on "novelty must move by < 8 points" without naming a statistic.
  One scorer read the mean (4.6, passes), the next read the p95 (25.4, fails) —
  three points of swing on identical code. Naming the statistic fixed it. The p95
  tail is real and is now reported rather than buried: on a record with no
  reference list, terminology carries about eighty per cent of the available
  fusion weight. Two fixes for it were implemented and measured — a steeper
  evidence-breadth term, and a confidence-weighted final re-rank — and neither
  moved the p95 by a point, so both were reverted.
- **C6** said scores must stay inside 1–100 while the code, the docs and `clamp`
  all use 0–100, and a paper with no identifiable authors legitimately scores 0.

Round 7 was worth more than its two points. The memory number had been reported
from a single unbounded run of both rankers in one process, which is wrong three
ways: resident memory never shrinks, so whichever ranker ran second was charged
for the first; V8 grows its heap against spare machine memory, so this algorithm
measured 348, 498 and 639 MB on three identical runs while the baseline sat at
exactly 272 every time; and the extension runs in a service worker that has no
such memory to be opportunistic with. Measured properly — three runs each, own
process, bounded heap, medians — it is **1.11×**, reproducible to a megabyte, and
the previously reported 2.3× was mostly an artefact of the measurement.

Two genuine reductions landed alongside it: the pipeline no longer builds three
successive ten-thousand-element arrays of spread copies, and the lexicon's raw
frequency table is released once the IDF is derived instead of held for the pass.

Also tried and reverted in these rounds, with the evidence left in the comments:
team juniority as a fusion signal (AUC 0.64 alone, made the model worse in
combination), and dropping the text-derived features from the consolidation model
(review detection fell and corpus stability got *worse*, so the theory was wrong).

---

## Round 8 — four more attempts on the last two gaps, all reverted

The remaining points sit in C5 (corpus stability) and C1 (separation against
derivative research). Both were attacked directly; neither moved. Recorded
because a list of what does *not* work is the more useful half of this log.

**C5 — corpus stability.** Decomposing the drift showed the peer set and the
confidence term are perfectly stable, and the movement enters through the fused
signal value itself. Four fixes followed from that, and all four made it worse:

| Attempt | Corpus drift | Cost |
|---|---|---|
| Per-field vocabulary history | 6.8 → 6.8 | −0.037 nDCG |
| Drop text features from the consolidation model | 6.8 → 7.5 | review AUC 0.694 → 0.674 |
| Fixed reference scale for the consolidation penalty | 6.8 → 7.9 | −0.016 nDCG |
| Rate-based rather than absolute "new to the field" | 6.8 → **11.8** | −0.025 nDCG |

The sensitivity is intrinsic to scoring a paper against its corpus. Every removal
of corpus-relativity costs the accuracy that corpus-relativity buys.

**C1 — separation.** Two new families of signal were measured on the tuning split
and both looked excellent alone:

- *Team juniority* (the least-established author's h-index, inverted): AUC 0.64,
  and it points the way the team-composition literature says it should.
- *Reference popularity* (how heavily cited a paper's sources are, mean and peak):
  rank correlation 0.145, the **strongest of any signal measured here**.

Fused, both made the model worse. Reference popularity raised nDCG@50 to 0.373
while dropping separation against derivative research from 0.654 to 0.638 — the
tuner optimises the pooled AUC it is given, and the pooled figure includes reviews,
so it happily traded away the thing the criterion actually reads. Both are computed
and reported as evidence; neither is fused.

---

## Round 9 — the bundled-model question, settled

The constraints permit shipping a model inside the extension, which is not the
same as calling a hosted one, and the first reading of them missed that. Measured
rather than argued, with a 22 MB quantized MiniLM on the same held-out queries:

| | nDCG@20 |
|---|---|
| Lexical, as shipped | 0.433 |
| Hybrid | 0.565 |
| Embeddings alone | **0.638** |

So the relevance points are real and reachable inside the stated constraints. Two
things stop it happening in this branch, and both are worth knowing:

**It buys relevance only.** Pointed at novelty, crowding in that same space
separates disruptive from derivative research at **AUC 0.501** — chance, and no
better than the corpus-local space it would replace. Text distance carries no
disruption signal however good the embedding is, which is exactly what the shipped
weights already encode by giving `crowding` zero weight.

**It costs a different product.** 27.7 ms per document makes a 10,000-paper pass
277 s against a baseline of 11 s, and the package goes from 0.37 MB to roughly
28 MB. There is a design that works — embed once at index time, persist, reuse,
which puts a feed window at 6.9 s — but that is an architecture change, and it is
a product call rather than a scoring one.

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

**C3 looked like the same defect and turned out not to be.** Its 10-point anchor
requires relevance nDCG@20 ≥ 0.55, and five method families capped out near 0.45:
tuned BM25, pseudo-relevance feedback, glossary and semantic query expansion, a
hybrid lexical-vector score, and a learned reranker over eight retrieval features.
The obvious conclusion was that the threshold was unreachable and should be
re-anchored like C1's.

Measuring the reference point first showed that conclusion was wrong. A
directly-prompted LLM reranking the identical top-100 pool for the identical
queries reaches **0.659**, against 0.406 for the shipped ranker on that pool. The
bar is achievable. What is unreachable is the bar under a "no hosted model at
runtime, must fit in a service worker" constraint — and to check that the space
itself was not the limit, the semantic space was rebuilt with PPMI and
power-iteration refinement, lifting same-topic discrimination from AUC 0.67 to
0.73; relevance moved 0.003. The gap is comprehension.

So the threshold stands and the eight points stay lost. The remaining gap is a
property of the constraints, not of the rubric and not of the effort — which is a
more useful thing for the reader to know than a re-anchored number would have
been.
