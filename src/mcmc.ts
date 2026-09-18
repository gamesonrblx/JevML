import { add, clip, gaussian, logSumExp, mulberry32, scale, type Rng } from "./math.ts";

export type TargetDensity = {
  id: string;
  name: string;
  logProb: (x: number[]) => number;
};

export type McmcResult = {
  samples: number[][];
  logProbs: number[];
  accepted: number;
  acceptanceRate: number;
  mean: number[];
};

function logMixGauss(
  x: number[],
  modes: { mu: number[]; logW: number; invVar: number }[],
): number {
  const terms = modes.map((m) => {
    let q = 0;
    for (let i = 0; i < x.length; i++) {
      const d = x[i]! - (m.mu[i] ?? 0);
      q += d * d * m.invVar;
    }
    return m.logW - 0.5 * q;
  });
  return logSumExp(terms);
}

export const TARGETS: TargetDensity[] = [
  {
    id: "mixture",
    name: "Two wells",
    logProb: (x) =>
      logMixGauss(x, [
        { mu: [-2.2, 0.4], logW: 0, invVar: 1.6 },
        { mu: [2.1, -0.3], logW: -0.15, invVar: 1.4 },
      ]),
  },
  {
    id: "banana",
    name: "Banana",
    logProb: (x) => {
      const a = x[0] ?? 0;
      const b = x[1] ?? 0;
      const t = b - 0.18 * a * a;
      return -0.5 * (a * a * 0.35 + t * t * 4.5);
    },
  },
  {
    id: "ring",
    name: "Ring",
    logProb: (x) => {
      const r = Math.hypot(x[0] ?? 0, x[1] ?? 0);
      const t = r - 2.4;
      return -0.5 * (t * t) * 14;
    },
  },
];

export function getTarget(id: string): TargetDensity {
  return TARGETS.find((t) => t.id === id) ?? TARGETS[0]!;
}

export function metropolisHastings(opts: {
  target: TargetDensity;
  start?: number[];
  nSamples?: number;
  stepSize?: number;
  seed?: number;
}): McmcResult {
  const nSamples = opts.nSamples ?? 800;
  const step = opts.stepSize ?? 0.35;
  const rng = mulberry32(opts.seed ?? 3);
  let x = (opts.start ?? [0, 0]).slice();
  let lp = opts.target.logProb(x);
  const samples: number[][] = [];
  const logProbs: number[] = [];
  let accepted = 0;

  for (let i = 0; i < nSamples; i++) {
    const prop = x.map((v) => v + gaussian(rng) * step);
    const lpProp = opts.target.logProb(prop);
    const logAlpha = Math.min(0, lpProp - lp);
    if (Math.log(rng() + 1e-12) < logAlpha) {
      x = prop;
      lp = lpProp;
      accepted++;
    }
    samples.push(x.slice());
    logProbs.push(lp);
  }

  const dim = x.length;
  const mean = Array.from({ length: dim }, (_, j) => {
    let s = 0;
    for (const p of samples) s += p[j] ?? 0;
    return s / samples.length;
  });

  return {
    samples,
    logProbs,
    accepted,
    acceptanceRate: accepted / nSamples,
    mean,
  };
}

export function densityGrid(
  target: TargetDensity,
  bounds: { x: [number, number]; y: [number, number] },
  res = 80,
): { z: number[][]; min: number; max: number } {
  const z: number[][] = [];
  let min = Infinity;
  let max = -Infinity;
  for (let iy = 0; iy < res; iy++) {
    const row: number[] = [];
    const y = bounds.y[1] - (iy / (res - 1)) * (bounds.y[1] - bounds.y[0]);
    for (let ix = 0; ix < res; ix++) {
      const x = bounds.x[0] + (ix / (res - 1)) * (bounds.x[1] - bounds.x[0]);
      const v = target.logProb([x, y]);
      row.push(v);
      if (v < min) min = v;
      if (v > max) max = v;
    }
    z.push(row);
  }
  return { z, min, max };
}

export function walkOnce(
  x: number[],
  lp: number,
  target: TargetDensity,
  step: number,
  rng: Rng,
): { x: number[]; lp: number; accepted: boolean } {
  const prop = add(x, scale([gaussian(rng), gaussian(rng)], step));
  const lpProp = target.logProb(prop);
  const accept = Math.log(rng() + 1e-12) < Math.min(0, lpProp - lp);
  if (accept) return { x: prop, lp: lpProp, accepted: true };
  return { x, lp, accepted: false };
}

export function essFromTrace(xs: number[]): number {
  const n = xs.length;
  if (n < 8) return n;
  const mean = xs.reduce((s, v) => s + v, 0) / n;
  const var0 =
    xs.reduce((s, v) => s + (v - mean) * (v - mean), 0) / Math.max(n - 1, 1);
  if (var0 <= 1e-12) return n;
  let acc = 0;
  const maxLag = Math.min(80, Math.floor(n / 4));
  for (let lag = 1; lag <= maxLag; lag++) {
    let c = 0;
    for (let i = 0; i < n - lag; i++) {
      c += (xs[i]! - mean) * (xs[i + lag]! - mean);
    }
    const rho = c / ((n - lag) * var0);
    if (rho <= 0) break;
    acc += rho;
  }
  return clip(n / (1 + 2 * acc), 1, n);
}
