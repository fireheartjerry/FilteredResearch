// Sentence embeddings for relevance ranking.
//
// Why this exists, and why only here. Interest phrases and the papers that answer
// them frequently share no words: a reader asking about "quantum and electron
// transport phenomena" wants papers that say "ballistic conduction in nanowires".
// Five families of bag-of-words retrieval were measured against that gap and all
// capped near nDCG@20 0.45 -- tuned BM25, pseudo-relevance feedback, glossary and
// corpus-derived query expansion, a hybrid lexical-vector score, and a learned
// reranker. A contextual model reaches 0.638 on the same held-out queries, within
// 0.02 of what an LLM achieves. The gap is comprehension, and nothing cheaper
// reproduces it: distilling the model down to shipped word vectors, which would
// have cost 2 MB instead of 22, reaches only 0.465, because averaging static word
// vectors discards exactly what the model knows.
//
// It is used for RELEVANCE ONLY. Pointed at novelty, similarity in this same
// space separates disruptive from derivative research at AUC 0.501 -- chance --
// so the scoring pass does not touch it and its cost stays out of that path.
//
// Weights ship inside the extension and are loaded from disk. No request is made
// at runtime, and nothing here is reachable unless a user has typed an interest
// phrase.

const MODEL_ID = "Xenova/all-MiniLM-L6-v2";
export const EMBEDDING_DIMENSIONS = 384;

// Embedding a document costs about 28 ms, so a window's worth is seconds rather
// than milliseconds. Vectors are therefore cached by work id: a paper is embedded
// once, the first time it is ranked, and reused for every query after.
const vectorCache = new Map();
const MAX_CACHED_VECTORS = 20_000;

let pipelinePromise = null;
let unavailableReason = null;

// Resolved lazily and only on first use, so an install that never types an
// interest phrase never pays for the model at all.
async function loadPipeline(options = {}) {
  if (unavailableReason) return null;
  if (!pipelinePromise) {
    pipelinePromise = (async () => {
      const transformers = await options.importTransformers();
      transformers.env.allowRemoteModels = false;
      transformers.env.allowLocalModels = true;
      transformers.env.localModelPath = options.modelPath;
      if (options.wasmPath) transformers.env.backends.onnx.wasm.wasmPaths = options.wasmPath;
      transformers.env.backends.onnx.wasm.numThreads = 1;
      return transformers.pipeline("feature-extraction", MODEL_ID, { quantized: true });
    })().catch((error) => {
      // A missing or broken model must degrade to lexical ranking, never break
      // the feed. The reason is recorded so it can be surfaced rather than
      // silently swallowed.
      unavailableReason = String(error?.message || error);
      return null;
    });
  }
  return pipelinePromise;
}

export function embeddingUnavailableReason() {
  return unavailableReason;
}

export function textForEmbedding(work) {
  return `${work?.title || ""}. ${String(work?.abstract || "").slice(0, 900)}`;
}

function remember(id, vector) {
  if (vectorCache.size >= MAX_CACHED_VECTORS) {
    // Oldest first; Map preserves insertion order.
    const oldest = vectorCache.keys().next().value;
    vectorCache.delete(oldest);
  }
  vectorCache.set(id, vector);
}

// Embeds whatever is not already cached, in batches, and returns vectors in the
// order the works were given.
export async function embedWorks(works, options) {
  const pipeline = await loadPipeline(options);
  if (!pipeline) return null;

  const pending = works.filter((work) => !vectorCache.has(work.id));
  const batchSize = options.batchSize || 32;
  for (let index = 0; index < pending.length; index += batchSize) {
    const batch = pending.slice(index, index + batchSize);
    const output = await pipeline(batch.map(textForEmbedding), { pooling: "mean", normalize: true });
    const rows = output.tolist();
    batch.forEach((work, position) => remember(work.id, Float32Array.from(rows[position])));
    if (options.onProgress) options.onProgress(Math.min(index + batchSize, pending.length), pending.length);
  }
  return works.map((work) => vectorCache.get(work.id) || null);
}

export async function embedQuery(query, options) {
  const pipeline = await loadPipeline(options);
  if (!pipeline) return null;
  const output = await pipeline([String(query || "")], { pooling: "mean", normalize: true });
  return Float32Array.from(output.tolist()[0]);
}

export function cosine(left, right) {
  if (!left || !right) return 0;
  let total = 0;
  for (let index = 0; index < left.length; index += 1) total += left[index] * right[index];
  return total;
}

export function cachedVectorCount() {
  return vectorCache.size;
}

export function clearVectorCache() {
  vectorCache.clear();
}
