# Filtered Research v1.0.1 specification

## Product contract

Filtered Research is an open-source, local-first Chrome extension for finding recent papers that are both unusual within their field -- judged from the reference graph, the text and the shape of the work -- and associated with an established authorship track record. It is not a publisher crawler, paper archive, citation recommender, peer-review substitute, or AI research product.

The primary user chooses an OpenAlex field/subfield and an index depth in the sidebar. Depth options are 1 day and 3 days (Light), 1 week and 2 weeks (Moderate), and 1 month and 3 months (Intensive). Three months is the ceiling: deeper passes retrieved far more works than the scoring stage could keep up with, which made discovery unusably slow. Preferences are local-only. Discovery starts through an explicit user refresh, or on the automatic interval the user chooses.

## Functional requirements

### Discovery

- Use OpenAlex article/preprint metadata over HTTPS.
- Identify the extension to OpenAlex with a fixed contact address on every request, so callers land in the polite pool rather than the anonymous one. The address describes the software, is the same for every install, and carries no user data.
- A focused manual scan uses cursor paging until the API result is exhausted or 1,000,000 unique works are retrieved.
- Report API total, records downloaded, unique papers after duplicate merging, truncation, coverage percentage, estimated cost, and progress.
- Coverage percentage is records downloaded over the API total. It is never computed from post-deduplication paper counts, which understated complete passes.
- A category lane carries no keyword; the selected scope is indexed exhaustively so interests can be changed later without re-fetching.
- Require a user-owned OpenAlex key for exhaustive indexing. Never ship a shared key.
- Without a key, retrieve up to 1,000 works and present the result as a working sample, naming a free key as the way to remove the limit. Never bundle a developer key: a key reaches OpenAlex only when the user supplied it, so the publisher can never fund another person's usage.
- If no taxonomy field is selected, use a rotating cross-disciplinary preview rather than claiming exhaustive global coverage.
- Install, settings saves, and depth changes make no discovery requests. Startup makes one only when automatic scanning is enabled and its interval has elapsed. Group records by normalized title plus author identity even across different DOIs and preserve up to one year locally.
- A stored index depth beyond the supported ceiling migrates onto the nearest supported depth instead of being rejected.
- Full discovery runs only after explicit user action and the prior saved feed remains available until completion.
- A pass is started without holding a message channel open, because it runs longer than a Manifest V3 service worker is guaranteed to live. Callers watch the progress state the worker persists, which also keeps the worker alive; a dropped poll never aborts the watch, and a pass that stops advancing is reported rather than awaited forever.
- Enforce a $0.95 full-pass guard by estimated OpenAlex request costs; show locally recorded daily usage in Settings.
- Retrieval is hybrid. The feed renders from the local index first, then a bounded gap fill retrieves what the last pass missed for the current window only, under its own budget and rate-limited per scope, so it never turns a render into a network wait.

### Filters

- Fetch and cache the OpenAlex field/subfield taxonomy for 30 days, with a Computer Science fallback.
- Use arXiv's official technical taxonomy internally while displaying clean category names; supplement it with OpenAlex general fields not covered by arXiv.
- A parent field checkbox means every child subfield. Individual subfields may be selected instead.
- Normalize taxonomy identifiers from both bare numeric IDs and current OpenAlex URL-shaped IDs.
- A selected category is the filter. Interest queries rank results and do not exclude papers, because requiring exact wording discarded most of an indexed subfield. Enabling strict interest filtering restores the AND behaviour.
- With no category selected, interest queries are the only scope and do filter.
- Interest matching considers title, abstract, and the paper's own topic, subfield and field labels; multiple phrases combine with OR.
- English-only is enabled by default. Missing/non-English metadata is rejected when enabled.

### Ranking

- Score novelty from ten signals over the reference graph, the text and the
  paper's own shape, each ranked inside the paper's field cohort before being
  combined. Only signals the record has evidence for are combined and the weights
  are renormalised over those, so a record without a reference list is scored on
  what it has rather than imputed to the middle.
- Fit the fusion weights, the consolidation coefficients and the discovery blend
  on a tuning split that is disjoint from every reported number, and ship them as
  constants. No fitting happens on the user's machine.
- Compare text two ways and take the stronger as evidence that the work already
  exists: a semantic space built from the corpus's own word co-occurrence with the
  corpus mean projected out, and BM25 over unigrams and bigrams.
- Never let a term seen in only one or two documents carry the score, so typos,
  OCR debris and injected vocabulary cannot buy novelty.
- Cap novelty by nearest-neighbour text similarity, so a near-verbatim
  restatement of existing work scores at the bottom whatever its bibliography
  looks like.
- Detect consolidation work structurally rather than by keyword. A keyword list
  remains as a small prior for records with no reference list, and the test suite
  verifies that removing it barely changes the separation.
- Express novelty as a rank within the paper's own field, pooled with a sample of
  that field's own papers measured identically, so a quiet month cannot promote
  its least ordinary paper to the top of the range. Apply that ranking after the
  penalties and the confidence shrink, not before.
- Fall back to a fixed calibration curve when a field has too few peers to
  describe a distribution.
- Treat a change of scoring version as invalidating stored scores; a refresh
  rescores saved papers so one calibration applies across the feed.
- Score authorship from transparent OpenAlex author metrics and role, normalised
  within the paper's field so one slider position means the same thing across the
  taxonomy.
- Rank results by graded relevance to the user's interest phrases, not by a
  boolean match. Interests still never exclude a paper when a category is chosen.
- Convert selectivity 1-100 into logarithmic target top-fractions using the
  documented anchors.
- Apply novelty and authorship percentile cutoffs with AND.
- Produce identical scores for identical input on any machine, regardless of the
  order records arrive in.
- Show both raw scores and evidence. Never label either as truth, quality, reputation, peer review, or proven novelty.

### Interface

- Side panel uses past day, 3 days, week, 2 weeks, month, and 3 month tabs.
- The saved index depth is a hard ceiling on every date view. Views wider than the depth are shown disabled rather than silently repeating the widest allowed view, and the worker reports the depth it applied so the picker can never display a scope the feed did not use.
- Which views are offered follows the saved depth, not how far the last pass reached. A view inside the depth that has not been indexed yet stays selectable, is marked, and prompts for a refresh.
- Narrowing the depth re-filters the cached bundle locally and issues no request; only widening fetches, because the cache does not yet hold the wider views.
- Recency means first public release. An arXiv identifier encodes the month the preprint was submitted; when that precedes the recorded publication month it is used instead, so a preprint later re-published by a journal keeps its true age.
- Cards show the first-release date and name the later re-publication date rather than presenting old work as new, and list each repository once.
- One request builds every date view from a single corpus scan, and the side panel serves tab switches from that response. No tab switch issues a message, so switching cannot be delayed by a terminated service worker.
- The feed render path reads no work or author records to report its counts; only the settings page computes full corpus totals.
- Feed responses are paginated (maximum 250 serialized works) so a 20,000-paper low-selectivity result does not produce a giant Chrome message.
- Settings use the existing white/grey/opal minimal-brutalist visual system, fine type, overlapping edge lines, and restrained rounded corners.
- Settings explain the selectivity anchors, own-key requirement, expected first-run cost/time, local page inspection, and scoring limitations.
- Text normalization decodes entities, strips markup/control characters, applies NFKC, and repairs conservative lowercase-to-uppercase word joins.

### Notifications and automatic scanning

- The extension declares no content script and reads no web page. Page highlighting was removed in v0.8.0.
- Automatic scanning is off by default and offers 3 hours, 6 hours, 24 hours, and 3 days. A pass runs when Chrome starts and the interval has elapsed, and while Chrome stays open an alarm checks periodically.
- An automatic pass requires a user-owned OpenAlex key and a selected category, never runs while another pass is in flight, and stamps its attempt before starting so a failure cannot retry in a loop.
- A timestamp in the future, from a clock change, must not make every wake-up look due.
- Papers that clear both bars collect in a local inbox. The toolbar icon carries a red unread count that follows the inbox and clears when it is read or emptied.
- Native desktop notifications remain an optional Chrome permission requested only after the user enables them.

## Storage

- `chrome.storage.sync`: non-secret settings.
- `chrome.storage.local`: user OpenAlex key only, restricted to trusted extension contexts.
- IndexedDB: works, baseline references, author profiles, refresh metadata, and notification inbox.
- Non-baseline candidates older than 400 days are pruned.
- Scope/scoring-version changes invalidate the baseline comparison set.
- A clear-data action deletes IndexedDB state; saving an empty key removes the key value.

## Security and compliance requirements

- MV3 only; no remote executable code, eval, telemetry, ads, accounts, or project-owned backend.
- No tabs, history, cookies, identity, clipboard, scripting, or all-sites permission.
- Insert all scholarly text with `textContent`, never `innerHTML` except fixed developer-authored skeleton markup.
- Treat all stored and remote metadata as possibly malformed: invalid dates must not reach `Intl`, and no asynchronous handler or fire-and-forget extension call may reject unhandled.
- Treat API data and page DOM as untrusted.
- Publish the privacy policy, accurate Chrome Web Store disclosures, OpenAlex source/terms notice, no-endorsement language, and heuristic limitations.
- Credit development assistance from OpenAI Codex without implying OpenAI endorsement.

## Acceptance tests

- A production work with `fieldId=https://openalex.org/fields/17` matches selected field `17`.
- AI with selectivity 1/1 retains every locally indexed AI paper regardless of low authorship score.
- 80/80 targets the top 5% on each signal and requires both; exceptional performance on only one axis is excluded.
- 100 targets top 0.02% per axis and ordinarily yields 0–2 papers for a ~10,000-paper month.
- Cursor paging has no duplicate IDs and reports truncation honestly.
- Joined text examples `EfficientTwo`, `PlatformAdvanced`, and `FromFilamentary` are repaired; escaped HTML is decoded.
- The manifest contains only named host patterns and optional notifications, and contains no all-URL/history/tabs permission.
- The exact packaged directory passes static checks, automated tests, and manual Chrome rendering/highlighting checks.
