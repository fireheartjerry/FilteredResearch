# Scoring model v3

Filtered Research computes every score locally from OpenAlex metadata. It does not
call a language model, it downloads no weights, and it does not claim to judge
scientific merit. These are screening heuristics for deciding what to read next.

Everything below is measured. See [../eval/README.md](../eval/README.md) for how,
and `eval/report.json` for the current numbers.

## Why this replaced v2

The previous model asked one question — how far is this paper's wording from other
papers' wording — and answered it with TF-IDF cosine against the first 320 older
records in insertion order. Measured against 1,026 papers whose disruptiveness is
now known from the citation graph, its rank correlation with reality was **−0.05**:
no relationship at all. Its interest matching was a boolean, so searching `RAG`
returned Tagalog relative-clause processing above every retrieval-augmented
generation paper in the index.

## Novelty

Ten signals, each turned into a rank inside the paper's own field cohort before
being combined, so no signal can dominate through having a wider range and so a
score means the same thing in materials science as in machine learning.

Only the signals a paper actually has evidence for are combined; the weights are
renormalised over what is present. A record with no reference list is scored on
what it does have, and its confidence reflects that rather than being quietly
imputed to the middle.

| Weight | Signal | What it asks |
|---:|---|---|
| 0.232 | `referenceCount` | Breadth of citation — surveys cite far more |
| 0.196 | `emergentDensity` | Vocabulary the field did not have before |
| 0.142 | `canonShare` | How much of its bibliography is the field's standard canon |
| 0.108 | `pairSupport` | How often the field already cites its sources together |
| 0.088 | `unfamiliarReferenceFraction` | Share of sources this field never cites |
| 0.075 | `unseenPairFraction` | Share of its source pairings nobody has made before |
| 0.062 | `strongestCoupling` | Strongest bibliography overlap with an existing paper |
| 0.053 | `coupledPeerFraction` | How many papers share its sources |
| 0.036 | `genericness` | How much of the paper is just the field's average |
| 0.009 | `topicConcentration` | Whether it sits on one topic or spreads across many |

Text crowding is deliberately absent from this table: measured against real
outcomes its rank correlation with disruption is negative, so the shrink rule
excludes it. It still catches near-verbatim restatements, through the ceiling
described below rather than as a term in the sum.

Weights are fitted by coordinate ascent on a tuning split and then shrunk toward a
uniform prior over every positively-correlated signal. The reported metrics come
from a disjoint test split. Nothing is fitted on the numbers that get reported.

### Text comparison

Similarity uses two views and takes the stronger as evidence that the work already
exists:

- **Semantic.** Terms get a fixed sparse random signature; a term's meaning vector
  is the sum of the signatures of the words it appears beside. Terms used in the
  same contexts end up pointing the same way, which is what makes a paraphrase
  look like a paraphrase. The corpus mean is projected out of every document
  vector — without that step, unrelated papers score 0.44 similarity and
  same-topic pairs separate from different-topic pairs at AUC 0.53, barely above a
  coin flip; with it, 0.63.
- **Lexical.** BM25 over unigrams and bigrams, which is what carries a corpus too
  small or too narrow for the co-occurrence statistics to have settled.

A term seen in fewer than a handful of documents never enters the vocabulary, so
typos, OCR debris and injected nonsense contribute nothing rather than reading as
maximally informative — which is exactly what v2's mean-IDF "distinctiveness" did.

### Consolidation

Two terms, and the structural one carries it:

- **Structural**, from reference count, topic concentration, centrality, canon
  share, coupling, team size, crowding and genericness. Logistic coefficients
  fitted against OpenAlex's `type: review` on the tuning split; held-out
  separation is **AUC 0.69**. The document-type field is deliberately *not* an
  input — it is what the benchmark defines the negative class with.
- **Lexical**, a small list of patterns like "a systematic review" and "a
  comparative study". Capped at 6 points, and `eval:adversarial` verifies that
  switching it off drops review separation by less than 0.05 AUC.

### Duplicates

Text similarity acts as a ceiling on the score rather than as another additive
term. A near-verbatim restatement of existing work is the least novel thing that
can enter a corpus whatever its bibliography looks like, and measured against real
outcomes text distance carries no disruption signal — so paying for it across the
whole distribution would cost accuracy everywhere to catch a rare case. As a
ceiling it costs nothing and catches it completely.

### Confidence and calibration

Scores are shrunk toward the middle in proportion to how much evidence exists:
abstract length, peer count, and how many signals were available. The final number
is then a rank within the paper's own field cohort — pooled with a sample of the
field's own papers measured the same way, so a quiet month cannot promote its
least ordinary paper to the top of the range — blended 85/15 with the raw value.

The re-rank happens *after* the penalties and the shrink, not before. Applied to an
already-calibrated number, those steps pulled the population toward the middle and
left the top and bottom deciles nearly empty.

## Authorship

Per-author career score, unchanged from v2:

```text
45% h-index   25% total citations   15% two-year mean citedness
10% works count   5% ORCID presence
```

Middle authors take a 0.86 role multiplier; the paper takes
`82% × strongest author + 18% × median enriched author`.

New in v3: the result is **field-normalised**. An h-index of 40 is unremarkable in
medicine and exceptional in mathematics, so the raw score is converted to a
percentile among the authors publishing in the same field and blended 75/25 with
the raw value. Measured field-to-field drift falls from 0.118 to 0.107 of the
scale.

This is called **Authorship**, not "researcher quality". Bibliometrics carry
field, career-stage, identity-resolution and citation-culture biases. A high score
makes no claim about the paper.

## Relevance

Interest phrases now rank results, which is what `SPEC.md` always said they did.
BM25 over title and abstract, with title terms weighted 1.2×, a proximity bonus for
query words that stay together, and a coverage term so a paper matching every query
term beats one matching a single term repeatedly.

Queries expand three ways: a glossary of the abbreviations researchers actually
type (`RAG` → retrieval augmented generation, and back), the derived acronym of a
multi-word query, and the nearest terms in the corpus's own semantic space at
reduced weight.

Queries are also asked twice: the strongest terms of the best few answers are
harvested and folded back in at reduced weight (pseudo-relevance feedback), which
is what lets a phrase find papers that describe the same thing in the field's own
jargon.

### The bundled sentence model

Lexical retrieval has a ceiling on this task and it is not high. A reader asking
about "quantum and electron transport phenomena" wants papers that say "ballistic
conduction in nanowires", and no amount of term weighting bridges that. Five
families were measured against the gap -- tuned BM25, pseudo-relevance feedback,
glossary and corpus-derived query expansion, a hybrid lexical-vector score, and a
learned reranker over eight retrieval features -- and all capped near nDCG@20 0.45.

The extension therefore ships a small sentence-embedding model
(`all-MiniLM-L6-v2`, quantized, 22 MB) and orders interest-matched results by
semantic similarity, with the lexical score as the fallback and the tie-break.
Nothing is downloaded at runtime: the weights and the inference runtime are files
inside the extension, which is what Manifest V3 requires.

Two things worth knowing about that choice:

- **It is used for relevance only.** Pointed at novelty, similarity in the same
  space separates disruptive from derivative research at AUC 0.501 -- chance --
  so the scoring pass never touches it and never pays for it.
- **There is no cheaper version.** Distilling the model into shipped word vectors
  would have cost 2 MB instead of 22; measured, it reaches 0.465, because
  averaging static word vectors discards what the model actually knows.

Embedding a paper costs about 28 ms, so vectors are cached by work id: a paper is
embedded once, the first time it is ranked, and reused afterwards. An install with
no interest phrases never loads the model at all.

Three retrieval-specific encoders of the same size were measured against this
general-purpose one and all lost: bge-small 0.606, gte-small 0.616, e5-small
0.527, against 0.638. Encoding title and abstract separately and taking the better
match gains 0.002 for three times the inference, and is not used.

Measured on held-out topic judgements, nDCG@20 goes from **0.03 to 0.64**, MRR from
0.04 to 0.90, recall@100 from 0.06 to 0.90. Without the model present the ranker
degrades to lexical order and scores 0.43, which is still seventeen times the
0.025 it replaced.

## Ordering and selectivity

Admission is unchanged: novelty and authorship percentile cutoffs, applied with
AND, with the documented logarithmic anchors.

```text
1 → nearly all       40 → top 50%       80 → top 5%
20 → top 75%         60 → top 20%       90 → top 1%
100 → top 0.02%
```

Among admitted papers the default order is

```text
discovery = 0.55 × novelty + 0.25 × authorship
          + 0.20 × max(novelty, authorship)
```

This is a viewing order, not an admission rule. It is deliberately novelty-led:
measured against forward citations, *no* blend of the two beats authorship on its
own, because author standing is what predicts who gets cited. Ranking the feed
that way would make it a prestige list, which is the opposite of what it is for.

## Reproducibility

Records store `SCORING_VERSION`, the full evidence object, and the scoring time.
Changing the scoring version invalidates stored scores and a refresh rescores the
feed, because scores from different scales are not comparable. Scoring is
deterministic: the same corpus produces identical numbers on any machine, and the
order papers arrive in does not affect any score.

## Limits

Novelty is measured against a local sample of a field, not against all of science.
A high score means "unlike what this index holds", never "this has never been
done". Disruption labels come from a citation graph and inherit its biases.
Equations, figures, datasets and the actual correctness of a result are not read
at all.
