import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { readFile } from "node:fs/promises";
import test from "node:test";

// The benchmark's whole claim is that the new algorithm beats the old one. That
// only means something if the old one is the real old one, so these files are
// checked against git rather than trusted. An accidental edit to eval/baseline/
// -- or a deliberate one -- would otherwise silently move the number every
// improvement is measured against.
const BASELINE_FILES = ["scoring", "filters", "ranking", "defaults", "papers", "prominence", "openalex"];

test("the benchmark baseline is v1.0.1 exactly as it shipped", async () => {
  for (const name of BASELINE_FILES) {
    const copied = await readFile(new URL(`../eval/baseline/${name}.js`, import.meta.url), "utf8");
    const original = execFileSync("git", ["show", `main:src/shared/${name}.js`], {
      cwd: new URL("..", import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, "$1"),
      encoding: "utf8",
      maxBuffer: 8 * 1024 * 1024,
    });
    assert.equal(
      copied.replace(/\r\n/g, "\n"),
      original.replace(/\r\n/g, "\n"),
      `eval/baseline/${name}.js no longer matches main:src/shared/${name}.js`,
    );
  }
});
