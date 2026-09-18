import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { generateElongated, pca, reconstruct } from "./pca.ts";
import { dot, norm } from "./math.ts";

describe("pca", () => {
  it("recovers an elongated cloud onto one dominant axis", () => {
    const X = generateElongated({ n: 200, seed: 4, stretch: 4 });
    const result = pca(X, 2);
    assert.equal(result.components.length, 2);
    assert.ok(result.explainedVarianceRatio[0]! > 0.8);
    const a = result.components[0]!;
    const b = result.components[1]!;
    assert.ok(Math.abs(dot(a, b)) < 1e-6);
    assert.ok(Math.abs(norm(a) - 1) < 1e-6);
  });

  it("reconstructs exactly at full rank", () => {
    const X = [
      [1, 2],
      [2, 4],
      [3, 5],
      [0, 1],
    ];
    const result = pca(X);
    const hat = reconstruct(result, result.components.length);
    for (let i = 0; i < X.length; i++) {
      for (let j = 0; j < 2; j++) {
        assert.ok(Math.abs(X[i]![j]! - hat[i]![j]!) < 1e-6);
      }
    }
  });
});
