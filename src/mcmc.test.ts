import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { getTarget, metropolisHastings } from "./mcmc.ts";

describe("mcmc", () => {
  it("accepts a reasonable fraction of proposals", () => {
    const result = metropolisHastings({
      target: getTarget("mixture"),
      nSamples: 600,
      stepSize: 0.4,
      seed: 9,
    });
    assert.ok(result.acceptanceRate > 0.15);
    assert.ok(result.acceptanceRate < 0.9);
    assert.equal(result.samples.length, 600);
  });

  it("spends time near at least one mode of the two-well target", () => {
    const result = metropolisHastings({
      target: getTarget("mixture"),
      nSamples: 1200,
      stepSize: 0.45,
      seed: 2,
      start: [-2, 0.4],
    });
    const tail = result.samples.slice(400);
    const meanX = tail.reduce((s, p) => s + p[0]!, 0) / tail.length;
    assert.ok(Math.abs(meanX - -2.2) < 1.2 || Math.abs(meanX - 2.1) < 1.2);
  });
});
