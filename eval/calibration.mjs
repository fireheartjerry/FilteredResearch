// npm run eval:calibration -- does a 70 mean the same thing everywhere?
//
// A score is only useful if the user can set one selectivity slider and have it
// behave the same way in computer science as in materials science, this quarter
// as last quarter. These four checks are what that claim decomposes into.

import { writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { loadCorpus, currentAdapter } from "./lib/adapters.mjs";
import { mean, stdev, spearman, seededRandom } from "./lib/metrics.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));

function decileOccupancy(scores) {
  const bands = new Array(10).fill(0);
  for (const score of scores) {
    const index = Math.min(9, Math.max(0, Math.floor(score / 10)));
    bands[index] += 1;
  }
  return bands.map((count) => count / Math.max(1, scores.length));
}

async function main() {
  const adapter = await currentAdapter();
  const corpus = loadCorpus();
  const fields = [...new Set(corpus.rawCandidates.map((work) => work.benchField))];
  const windows = [...new Set(corpus.rawCandidates.map((work) => work.benchWindow))];
  const rawById = new Map(corpus.rawCandidates.map((work) => [work.id, work]));

  const cells = [];
  for (const field of fields) {
    for (const window of windows) {
      const candidates = corpus.candidates.filter((work) => {
        const source = rawById.get(work.id);
        return source?.benchField === field && source?.benchWindow === window;
      });
      const peers = corpus.peers.filter((peer) => peer.benchField === field);
      if (candidates.length < 40) continue;
      const scored = adapter.scoreBatch(candidates, peers, corpus.authors);
      cells.push({ field, window, scores: scored.map((work) => work.noveltyScore || 0), byId: new Map(scored.map((w) => [w.id, w.noveltyScore || 0])) });
    }
  }

  const pooled = cells.flatMap((cell) => cell.scores);
  const occupancy = decileOccupancy(pooled);
  const occupancy_ok = occupancy.every((share) => share >= 0.06 && share <= 0.14) && Math.max(...occupancy) <= 0.2;

  const cellMeans = cells.map((cell) => mean(cell.scores));
  const cellSigmas = cells.map((cell) => stdev(cell.scores)).filter((value) => value > 0);
  const meanDrift = Math.max(...cellMeans) - Math.min(...cellMeans);
  const sigmaRatio = Math.max(...cellSigmas) / Math.max(1e-9, Math.min(...cellSigmas));
  const drift_ok = meanDrift < 6 && sigmaRatio >= 0.7 && sigmaRatio <= 1.4;

  // Corpus-size stability: the same papers, scored against twice as much
  // history. A percentile-based scale should barely move.
  const field = fields[0];
  const focus = corpus.candidates.filter((work) => rawById.get(work.id)?.benchField === field);
  const halfPeers = corpus.peers.filter((peer) => peer.benchField === field);
  const doublePeers = [...halfPeers, ...corpus.peers.filter((peer) => peer.benchField !== field).slice(0, halfPeers.length)];
  const small = new Map(adapter.scoreBatch(focus, halfPeers, corpus.authors).map((w) => [w.id, w.noveltyScore || 0]));
  const large = new Map(adapter.scoreBatch(focus, doublePeers, corpus.authors).map((w) => [w.id, w.noveltyScore || 0]));
  const shifts = focus.map((work) => Math.abs((large.get(work.id) || 0) - (small.get(work.id) || 0)));
  const size_ok = mean(shifts) < 5;

  // Rank stability: drop a random tenth of the corpus and re-score. If the order
  // reshuffles, the score is reporting corpus composition, not the papers.
  const random = seededRandom(31337);
  const kept = focus.filter(() => random() > 0.1);
  const resampled = new Map(adapter.scoreBatch(kept, halfPeers, corpus.authors).map((w) => [w.id, w.noveltyScore || 0]));
  const commonIds = kept.map((work) => work.id);
  const rankStability = spearman(
    commonIds.map((id) => small.get(id) || 0),
    commonIds.map((id) => resampled.get(id) || 0),
  );
  const stability_ok = rankStability >= 0.9;

  const report = {
    generatedAt: new Date().toISOString(),
    cells: cells.length,
    fields: fields.length,
    windows: windows.length,
    decile_occupancy: occupancy.map((share) => Number(share.toFixed(3))),
    occupancy_ok,
    mean_drift_points: Number(meanDrift.toFixed(2)),
    sigma_ratio: Number(sigmaRatio.toFixed(3)),
    drift_ok,
    corpus_doubling_mean_shift: Number(mean(shifts).toFixed(2)),
    size_ok,
    resample_rank_spearman: Number(rankStability.toFixed(4)),
    stability_ok,
    per_cell: cells.map((cell) => ({ field: cell.field, window: cell.window, n: cell.scores.length, mean: Number(mean(cell.scores).toFixed(2)), sd: Number(stdev(cell.scores).toFixed(2)) })),
  };
  const passed = [occupancy_ok, drift_ok, size_ok, stability_ok].filter(Boolean).length;
  report.passed = passed;
  writeFileSync(join(HERE, "calibration.json"), `${JSON.stringify(report, null, 2)}\n`);

  console.log(`decile occupancy      ${report.decile_occupancy.join(" ")}   ${occupancy_ok ? "ok" : "FAIL"}`);
  console.log(`cross-cell mean drift ${report.mean_drift_points} pts, sigma ratio ${report.sigma_ratio}   ${drift_ok ? "ok" : "FAIL"}`);
  console.log(`corpus doubling shift ${report.corpus_doubling_mean_shift} pts   ${size_ok ? "ok" : "FAIL"}`);
  console.log(`resample rank rho     ${report.resample_rank_spearman}   ${stability_ok ? "ok" : "FAIL"}`);
  console.log(`\n${passed}/4 calibration checks pass\n`);
  process.exitCode = passed === 4 ? 0 : 1;
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
