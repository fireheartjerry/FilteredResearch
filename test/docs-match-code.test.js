import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import { NOVELTY_WEIGHTS } from "../src/shared/scoring.js";

// Documented weights drifted away from the shipped ones twice: v2's docs claimed
// a 78/14/8 split that appeared nowhere in the code, and the first draft of v3's
// table went stale the moment the tuner was re-run. A table a reader cannot trust
// is worse than no table, so it is pinned here rather than maintained by memory.
test("the documented novelty weights are the shipped novelty weights", async () => {
  const doc = await readFile(new URL("../docs/SCORING.md", import.meta.url), "utf8");

  for (const signal of NOVELTY_WEIGHTS) {
    assert.ok(
      doc.includes(`| ${signal.weight} | \`${signal.key}\``),
      `docs/SCORING.md is missing the shipped weight for ${signal.key} (${signal.weight})`,
    );
  }

  const documented = [...doc.matchAll(/^\| (\d\.\d+) \| `([a-zA-Z]+)`/gm)].map((match) => match[2]);
  assert.deepEqual(
    [...documented].sort(),
    NOVELTY_WEIGHTS.map((signal) => signal.key).sort(),
    "docs/SCORING.md lists a different set of signals than the code ships",
  );
  assert.ok(
    doc.includes(`${NOVELTY_WEIGHTS.length} signals`) || doc.includes("Ten signals"),
    "the stated signal count does not match the shipped one",
  );
});
