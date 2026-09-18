/** Small linear-algebra helpers for the JevML primitives. */

export type Rng = () => number;

export function mulberry32(seed: number): Rng {
  let t = seed >>> 0;
  return () => {
    t += 0x6d2b79f5;
    let x = Math.imul(t ^ (t >>> 15), 1 | t);
    x ^= x + Math.imul(x ^ (x >>> 7), 61 | x);
    return ((x ^ (x >>> 14)) >>> 0) / 4294967296;
  };
}

export function gaussian(rng: Rng): number {
  let u = 0;
  let v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

export function zeros(n: number): number[] {
  return Array.from({ length: n }, () => 0);
}

export function identity(n: number): number[][] {
  return Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)),
  );
}

export function transpose(A: number[][]): number[][] {
  const n = A.length;
  const m = A[0]?.length ?? 0;
  const T: number[][] = Array.from({ length: m }, () => Array<number>(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < m; j++) T[j]![i] = A[i]![j]!;
  }
  return T;
}

export function matVec(A: number[][], x: number[]): number[] {
  return A.map((row) => row.reduce((s, a, j) => s + a * (x[j] ?? 0), 0));
}

export function dot(a: number[], b: number[]): number {
  let s = 0;
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) s += a[i]! * b[i]!;
  return s;
}

export function norm(a: number[]): number {
  return Math.sqrt(dot(a, a));
}

export function add(a: number[], b: number[]): number[] {
  return a.map((v, i) => v + (b[i] ?? 0));
}

export function sub(a: number[], b: number[]): number[] {
  return a.map((v, i) => v - (b[i] ?? 0));
}

export function scale(a: number[], s: number): number[] {
  return a.map((v) => v * s);
}

export function meanVec(X: number[][]): number[] {
  const n = X.length;
  const d = X[0]?.length ?? 0;
  const m = zeros(d);
  for (const row of X) {
    for (let j = 0; j < d; j++) m[j]! += row[j] ?? 0;
  }
  return m.map((v) => v / Math.max(n, 1));
}

export function centerRows(X: number[][], mean?: number[]): { centered: number[][]; mean: number[] } {
  const mu = mean ?? meanVec(X);
  return { mean: mu, centered: X.map((row) => sub(row, mu)) };
}

export function clip(x: number, lo = 0, hi = 1): number {
  return Math.min(hi, Math.max(lo, x));
}

export function logSumExp(xs: number[]): number {
  const m = Math.max(...xs);
  if (!Number.isFinite(m)) return -Infinity;
  let s = 0;
  for (const x of xs) s += Math.exp(x - m);
  return m + Math.log(s);
}

/**
 * Jacobi eigenvalue decomposition of a symmetric matrix.
 * Returns eigenvalues (descending) and matching orthonormal eigenvectors as columns.
 */
export function jacobiEigen(A: number[][]): { values: number[]; vectors: number[][] } {
  const n = A.length;
  let M = A.map((row) => row.slice());
  const V = identity(n);

  for (let iter = 0; iter < 80 * n * n; iter++) {
    let p = 0;
    let q = 1;
    let max = 0;
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const a = Math.abs(M[i]![j]!);
        if (a > max) {
          max = a;
          p = i;
          q = j;
        }
      }
    }
    if (max < 1e-12) break;

    const app = M[p]![p]!;
    const aqq = M[q]![q]!;
    const apq = M[p]![q]!;
    const theta = 0.5 * Math.atan2(2 * apq, aqq - app);
    const c = Math.cos(theta);
    const s = Math.sin(theta);

    const N = M.map((row) => row.slice());
    for (let k = 0; k < n; k++) {
      if (k === p || k === q) continue;
      const mkp = M[k]![p]!;
      const mkq = M[k]![q]!;
      N[k]![p] = N[p]![k] = c * mkp - s * mkq;
      N[k]![q] = N[q]![k] = s * mkp + c * mkq;
    }
    N[p]![p] = c * c * app - 2 * s * c * apq + s * s * aqq;
    N[q]![q] = s * s * app + 2 * s * c * apq + c * c * aqq;
    N[p]![q] = 0;
    N[q]![p] = 0;
    M = N;

    for (let k = 0; k < n; k++) {
      const vkp = V[k]![p]!;
      const vkq = V[k]![q]!;
      V[k]![p] = c * vkp - s * vkq;
      V[k]![q] = s * vkp + c * vkq;
    }
  }

  const values = Array.from({ length: n }, (_, i) => M[i]![i]!);
  const order = values.map((_, i) => i).sort((i, j) => values[j]! - values[i]!);
  return {
    values: order.map((i) => values[i]!),
    vectors: order.map((j) => Array.from({ length: n }, (_, i) => V[i]![j]!)),
  };
}

export function covariance(centered: number[][]): number[][] {
  const n = centered.length;
  const d = centered[0]?.length ?? 0;
  const C: number[][] = Array.from({ length: d }, () => Array<number>(d).fill(0));
  const denom = Math.max(n - 1, 1);
  for (const row of centered) {
    for (let i = 0; i < d; i++) {
      const ri = row[i] ?? 0;
      for (let j = i; j < d; j++) {
        const v = (ri * (row[j] ?? 0)) / denom;
        C[i]![j]! += v;
        if (i !== j) C[j]![i]! += v;
      }
    }
  }
  return C;
}
