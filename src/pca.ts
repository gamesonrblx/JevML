import { centerRows, covariance, jacobiEigen, matVec, mulberry32, gaussian } from "./math.ts";

export type PcaResult = {
  mean: number[];
  components: number[][];
  eigenvalues: number[];
  explainedVarianceRatio: number[];
  scores: number[][];
};

export function pca(X: number[][], k?: number): PcaResult {
  if (X.length === 0 || (X[0]?.length ?? 0) === 0) {
    return {
      mean: [],
      components: [],
      eigenvalues: [],
      explainedVarianceRatio: [],
      scores: [],
    };
  }
  const d = X[0]!.length;
  const { mean, centered } = centerRows(X);
  const C = covariance(centered);
  const { values, vectors } = jacobiEigen(C);
  const rank = Math.min(k ?? d, d, values.length);
  const components = vectors.slice(0, rank);
  const eigenvalues = values.slice(0, rank);
  const total = values.reduce((s, v) => s + Math.max(v, 0), 0) || 1;
  const explainedVarianceRatio = eigenvalues.map((v) => Math.max(v, 0) / total);
  const scores = centered.map((row) => components.map((pc) => matVec([pc], row)[0] ?? 0));
  return { mean, components, eigenvalues, explainedVarianceRatio, scores };
}

export function reconstruct(result: PcaResult, rank?: number): number[][] {
  const r = Math.min(rank ?? result.components.length, result.components.length);
  return result.scores.map((score) => {
    const x = result.mean.slice();
    for (let i = 0; i < r; i++) {
      const s = score[i] ?? 0;
      const pc = result.components[i] ?? [];
      for (let j = 0; j < x.length; j++) x[j]! += s * (pc[j] ?? 0);
    }
    return x;
  });
}

export function generateBlobs(opts: {
  n?: number;
  dim?: number;
  seed?: number;
  clusters?: number;
  spread?: number;
}): number[][] {
  const n = opts.n ?? 180;
  const dim = opts.dim ?? 2;
  const clusters = opts.clusters ?? 3;
  const spread = opts.spread ?? 0.55;
  const rng = mulberry32(opts.seed ?? 7);
  const centers: number[][] = Array.from({ length: clusters }, () =>
    Array.from({ length: dim }, () => (rng() - 0.5) * 8),
  );
  const X: number[][] = [];
  for (let i = 0; i < n; i++) {
    const c = centers[i % clusters]!;
    X.push(c.map((v) => v + gaussian(rng) * spread));
  }
  return X;
}

export function generateElongated(opts: { n?: number; seed?: number; stretch?: number }): number[][] {
  const n = opts.n ?? 180;
  const stretch = opts.stretch ?? 3.4;
  const rng = mulberry32(opts.seed ?? 11);
  const angle = 0.55;
  const cos = Math.cos(angle);
  const sin = Math.sin(angle);
  const X: number[][] = [];
  for (let i = 0; i < n; i++) {
    const a = gaussian(rng) * stretch;
    const b = gaussian(rng) * 0.55;
    X.push([cos * a - sin * b, sin * a + cos * b]);
  }
  return X;
}

export function parseNumericTable(text: string): number[][] {
  const rows: number[][] = [];
  for (const line of text.split(/\n+/)) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#")) continue;
    const parts = trimmed.split(/[\s,;]+/).filter(Boolean);
    const nums = parts.map(Number).filter((n) => Number.isFinite(n));
    if (nums.length >= 2) rows.push(nums);
  }
  const dim = rows[0]?.length ?? 0;
  return rows.filter((r) => r.length === dim);
}
