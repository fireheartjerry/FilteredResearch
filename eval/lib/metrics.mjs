// Ranking metrics. Deliberately dependency-free and deterministic so the same
// numbers come out on any machine, including a scorer's.

export function dcg(gains) {
  let total = 0;
  for (let index = 0; index < gains.length; index += 1) {
    total += (2 ** gains[index] - 1) / Math.log2(index + 2);
  }
  return total;
}

// Graded nDCG. `ranked` is the order the algorithm produced; gainOf maps an item
// to its ground-truth gain. The ideal ordering is the same items sorted by gain,
// so a ranker is measured against the best it could possibly have done with the
// set it was given -- not against a hypothetical perfect corpus.
export function ndcgAt(ranked, gainOf, k) {
  if (!ranked.length) return 0;
  const gains = ranked.slice(0, k).map(gainOf);
  const ideal = ranked.map(gainOf).sort((left, right) => right - left).slice(0, k);
  const best = dcg(ideal);
  return best > 0 ? dcg(gains) / best : 0;
}

function ranksOf(values) {
  const order = values.map((value, index) => [value, index]).sort((left, right) => left[0] - right[0]);
  const ranks = new Array(values.length);
  let index = 0;
  while (index < order.length) {
    let end = index;
    while (end + 1 < order.length && order[end + 1][0] === order[index][0]) end += 1;
    // Ties share the average rank, otherwise a signal that emits many identical
    // scores gets credited for an ordering it never actually made.
    const shared = (index + end) / 2 + 1;
    for (let position = index; position <= end; position += 1) ranks[order[position][1]] = shared;
    index = end + 1;
  }
  return ranks;
}

export function spearman(left, right) {
  if (left.length !== right.length || left.length < 3) return 0;
  const a = ranksOf(left);
  const b = ranksOf(right);
  const n = a.length;
  const meanA = a.reduce((sum, value) => sum + value, 0) / n;
  const meanB = b.reduce((sum, value) => sum + value, 0) / n;
  let cov = 0;
  let varA = 0;
  let varB = 0;
  for (let index = 0; index < n; index += 1) {
    const da = a[index] - meanA;
    const db = b[index] - meanB;
    cov += da * db;
    varA += da * da;
    varB += db * db;
  }
  const denominator = Math.sqrt(varA * varB);
  return denominator ? cov / denominator : 0;
}

// Rank-based AUC (Mann-Whitney), so ties count as half a win rather than
// silently inflating the score.
export function auc(positiveScores, negativeScores) {
  if (!positiveScores.length || !negativeScores.length) return 0.5;
  const all = [...positiveScores, ...negativeScores];
  const ranks = ranksOf(all);
  let rankSum = 0;
  for (let index = 0; index < positiveScores.length; index += 1) rankSum += ranks[index];
  const n1 = positiveScores.length;
  const n2 = negativeScores.length;
  return (rankSum - (n1 * (n1 + 1)) / 2) / (n1 * n2);
}

export function meanReciprocalRank(rankedLists, isRelevant) {
  if (!rankedLists.length) return 0;
  let total = 0;
  for (const ranked of rankedLists) {
    const position = ranked.findIndex(isRelevant);
    if (position >= 0) total += 1 / (position + 1);
  }
  return total / rankedLists.length;
}

export function recallAt(ranked, isRelevant, k, totalRelevant) {
  if (!totalRelevant) return 0;
  let hits = 0;
  for (let index = 0; index < Math.min(k, ranked.length); index += 1) if (isRelevant(ranked[index])) hits += 1;
  return hits / totalRelevant;
}

export function precisionAt(ranked, isRelevant, k) {
  if (!ranked.length) return 0;
  const window = ranked.slice(0, k);
  return window.filter(isRelevant).length / window.length;
}

export function quantile(sortedValues, fraction) {
  if (!sortedValues.length) return 0;
  const position = (sortedValues.length - 1) * fraction;
  const low = Math.floor(position);
  const high = Math.ceil(position);
  if (low === high) return sortedValues[low];
  return sortedValues[low] + (position - low) * (sortedValues[high] - sortedValues[low]);
}

export function mean(values) {
  return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : 0;
}

export function stdev(values) {
  if (values.length < 2) return 0;
  const average = mean(values);
  return Math.sqrt(values.reduce((sum, value) => sum + (value - average) ** 2, 0) / (values.length - 1));
}

// Percentile of each value inside its own cohort, used to turn raw citation
// counts into a field-and-window-fair label.
export function cohortPercentiles(entries, cohortOf, valueOf) {
  const cohorts = new Map();
  for (const entry of entries) {
    const key = cohortOf(entry);
    if (!cohorts.has(key)) cohorts.set(key, []);
    cohorts.get(key).push(valueOf(entry));
  }
  for (const [key, values] of cohorts) cohorts.set(key, values.sort((left, right) => left - right));
  const result = new Map();
  for (const entry of entries) {
    const values = cohorts.get(cohortOf(entry));
    const value = valueOf(entry);
    let below = 0;
    let equal = 0;
    for (const other of values) {
      if (other < value) below += 1;
      else if (other === value) equal += 1;
    }
    result.set(entry, values.length ? (below + equal / 2) / values.length : 0);
  }
  return result;
}

// Deterministic PRNG so resampling experiments repeat exactly.
export function seededRandom(seed) {
  let state = seed >>> 0 || 1;
  return () => {
    state ^= state << 13;
    state >>>= 0;
    state ^= state >> 17;
    state ^= state << 5;
    state >>>= 0;
    return state / 4294967296;
  };
}
