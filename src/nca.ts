import { clip, mulberry32 } from "./math.ts";

export type GenomeId = "morphogen" | "spiral" | "grow" | "life" | "neural";

export type SimState = {
  width: number;
  height: number;
  kind: GenomeId;
  u: Float32Array;
  v: Float32Array;
  cells: Uint8Array;
  field: Float32Array;
  channels: number;
  generation: number;
};

export const GENOMES: { id: GenomeId; name: string; blurb: string }[] = [
  { id: "morphogen", name: "Morphogen", blurb: "Gray–Scott reaction-diffusion. Coral, worms, mitosis." },
  { id: "spiral", name: "Spiral", blurb: "FitzHugh–Nagumo excitable media. Waves that refuse to die." },
  { id: "grow", name: "Grow", blurb: "A seed with a local growth rule. Surface tension, then a body." },
  { id: "neural", name: "Neural", blurb: "Sobel perception plus a tiny MLP residual update." },
  { id: "life", name: "Life", blurb: "Conway. The original local-rule weather engine." },
];

export function createSim(kind: GenomeId, width: number, height: number, seed = 1): SimState {
  const n = width * height;
  const state: SimState = {
    width,
    height,
    kind,
    u: new Float32Array(n),
    v: new Float32Array(n),
    cells: new Uint8Array(n),
    field: new Float32Array(n * 4),
    channels: 4,
    generation: 0,
  };
  seedSim(state, seed);
  return state;
}

function idx(x: number, y: number, w: number, h: number): number {
  const xx = ((x % w) + w) % w;
  const yy = ((y % h) + h) % h;
  return yy * w + xx;
}

export function seedSim(state: SimState, seed = 1): void {
  const { width: w, height: h, kind } = state;
  const n = w * h;
  const rng = mulberry32(seed + kind.length * 17);
  state.generation = 0;
  state.u.fill(1);
  state.v.fill(0);
  state.cells.fill(0);
  state.field.fill(0);

  if (kind === "life") {
    for (let i = 0; i < n; i++) state.cells[i] = rng() > 0.78 ? 1 : 0;
    return;
  }

  if (kind === "morphogen") {
    for (let i = 0; i < n; i++) {
      state.u[i] = 1;
      state.v[i] = 0;
    }
    const rw = Math.max(4, Math.floor(w * 0.12));
    const rh = Math.max(4, Math.floor(h * 0.12));
    for (const [cx, cy] of [
      [Math.floor(w * 0.45), Math.floor(h * 0.48)],
      [Math.floor(w * 0.62), Math.floor(h * 0.4)],
    ] as const) {
      for (let y = -rh; y <= rh; y++) {
        for (let x = -rw; x <= rw; x++) {
          const i = idx(cx + x, cy + y, w, h);
          state.u[i] = 0.5 + rng() * 0.1;
          state.v[i] = 0.25 + rng() * 0.2;
        }
      }
    }
    return;
  }

  if (kind === "spiral") {
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const i = y * w + x;
        state.u[i] = x < w / 2 ? 1 : -1;
        state.v[i] = y / h - 0.4;
      }
    }
    return;
  }

  // grow + neural share a center seed
  const cx = Math.floor(w / 2);
  const cy = Math.floor(h / 2);
  const r = Math.max(2, Math.floor(Math.min(w, h) * 0.045));
  for (let y = -r; y <= r; y++) {
    for (let x = -r; x <= r; x++) {
      if (x * x + y * y > r * r) continue;
      const i = idx(cx + x, cy + y, w, h) * 4;
      state.field[i] = 0.55;
      state.field[i + 1] = 0.68;
      state.field[i + 2] = 0.58;
      state.field[i + 3] = 1;
    }
  }
}

export function paintSim(state: SimState, px: number, py: number, radius = 2.5): void {
  const { width: w, height: h, kind } = state;
  const r2 = radius * radius;
  for (let y = Math.floor(py - radius); y <= py + radius; y++) {
    for (let x = Math.floor(px - radius); x <= px + radius; x++) {
      const dx = x - px;
      const dy = y - py;
      if (dx * dx + dy * dy > r2) continue;
      const i = idx(x, y, w, h);
      if (kind === "life") {
        state.cells[i] = 1;
      } else if (kind === "morphogen") {
        state.u[i] = 0.4;
        state.v[i] = 0.6;
      } else if (kind === "spiral") {
        state.u[i] = 1;
        state.v[i] = 0.2;
      } else {
        const f = i * 4;
        state.field[f] = 0.6;
        state.field[f + 1] = 0.72;
        state.field[f + 2] = 0.58;
        state.field[f + 3] = 1;
      }
    }
  }
}

function lap5(grid: Float32Array, i: number, w: number, h: number): number {
  const x = i % w;
  const y = (i / w) | 0;
  const c = grid[i]!;
  return (
    grid[idx(x + 1, y, w, h)]! +
    grid[idx(x - 1, y, w, h)]! +
    grid[idx(x, y + 1, w, h)]! +
    grid[idx(x, y - 1, w, h)]! -
    4 * c
  );
}

function stepMorphogen(state: SimState): void {
  const { width: w, height: h, u, v } = state;
  const n = w * h;
  const Du = 0.16;
  const Dv = 0.08;
  const F = 0.035;
  const k = 0.06;
  const un = new Float32Array(n);
  const vn = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const uvv = u[i]! * v[i]! * v[i]!;
    un[i] = clip(u[i]! + Du * lap5(u, i, w, h) - uvv + F * (1 - u[i]!));
    vn[i] = clip(v[i]! + Dv * lap5(v, i, w, h) + uvv - (F + k) * v[i]!);
  }
  state.u.set(un);
  state.v.set(vn);
}

function stepSpiral(state: SimState): void {
  const { width: w, height: h, u, v } = state;
  const n = w * h;
  const a = 0.7;
  const b = 0.8;
  const e = 0.08;
  const D = 0.5;
  const dt = 0.15;
  const un = new Float32Array(n);
  const vn = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const z = u[i]!;
    const wch = v[i]!;
    un[i] = z + dt * (z - (z * z * z) / 3 - wch + D * lap5(u, i, w, h));
    vn[i] = wch + dt * e * (z + a - b * wch);
  }
  state.u.set(un);
  state.v.set(vn);
}

function stepLife(state: SimState): void {
  const { width: w, height: h, cells } = state;
  const n = w * h;
  const next = new Uint8Array(n);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let nb = 0;
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          if (dx === 0 && dy === 0) continue;
          nb += cells[idx(x + dx, y + dy, w, h)]!;
        }
      }
      const i = y * w + x;
      const alive = cells[i] === 1;
      next[i] = alive ? (nb === 2 || nb === 3 ? 1 : 0) : nb === 3 ? 1 : 0;
    }
  }
  state.cells.set(next);
}

function neighborStats(field: Float32Array, x: number, y: number, w: number, h: number) {
  let aSum = 0;
  let aMax = 0;
  let sx = 0;
  let sy = 0;
  for (let dy = -1; dy <= 1; dy++) {
    for (let dx = -1; dx <= 1; dx++) {
      const i = idx(x + dx, y + dy, w, h) * 4;
      const a = field[i + 3]!;
      aSum += a;
      if (a > aMax) aMax = a;
      sx += dx * a;
      sy += dy * a;
    }
  }
  return { aMean: aSum / 9, aMax, sx, sy };
}

function stepGrow(state: SimState): void {
  const { width: w, height: h, field } = state;
  const next = new Float32Array(field.length);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = (y * w + x) * 4;
      const r = field[i]!;
      const g = field[i + 1]!;
      const b = field[i + 2]!;
      const a = field[i + 3]!;
      const { aMean, aMax, sx, sy } = neighborStats(field, x, y, w, h);
      const edge = Math.hypot(sx, sy);
      const grow = 0.18 * aMean * (1.05 - a) - 0.035 * a + 0.02 * (aMax - a);
      const an = clip(a + grow);
      const tr = 0.52 + 0.1 * Math.tanh(edge);
      const tg = 0.66 + 0.08 * Math.tanh(aMean * 2 - 1);
      const tb = 0.54;
      const mix = 0.12 * an;
      next[i] = clip(r + (tr - r) * mix);
      next[i + 1] = clip(g + (tg - g) * mix);
      next[i + 2] = clip(b + (tb - b) * mix);
      next[i + 3] = an < 0.02 ? 0 : an;
      if (an < 0.02) {
        next[i] = 0;
        next[i + 1] = 0;
        next[i + 2] = 0;
      }
    }
  }
  state.field.set(next);
}

function stepNeural(state: SimState): void {
  const { width: w, height: h, field } = state;
  const next = new Float32Array(field.length);
  const fire = 0.55;
  const rng = mulberry32((state.generation * 9973 + 13) >>> 0);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = (y * w + x) * 4;
      const a = field[i + 3]!;
      const { aMean, aMax, sx, sy } = neighborStats(field, x, y, w, h);
      const aliveN = aMax > 0.1;
      if (!aliveN && a < 0.1) {
        next[i] = 0;
        next[i + 1] = 0;
        next[i + 2] = 0;
        next[i + 3] = 0;
        continue;
      }
      if (rng() > fire) {
        next[i] = field[i]!;
        next[i + 1] = field[i + 1]!;
        next[i + 2] = field[i + 2]!;
        next[i + 3] = a;
        continue;
      }
      // Tiny MLP: h = tanh(W · perception)
      const p = [
        field[i]!,
        field[i + 1]!,
        field[i + 2]!,
        a,
        aMean,
        sx * 0.35,
        sy * 0.35,
        aMax,
      ];
      const h0 = Math.tanh(0.9 * p[4]! + 0.4 * p[7]! - 0.35);
      const h1 = Math.tanh(0.6 * p[5]! - 0.6 * p[6]!);
      const h2 = Math.tanh(0.8 * (0.6 - p[3]!) + 0.3 * p[4]!);
      const da = 0.16 * h0 + 0.04 * h2 - 0.03 * p[3]!;
      const dr = 0.05 * (0.55 - p[0]!) + 0.03 * h1;
      const dg = 0.05 * (0.7 - p[1]!) - 0.02 * h1;
      const db = 0.04 * (0.56 - p[2]!);
      const an = clip(a + da);
      next[i] = clip(p[0]! + dr) * an;
      next[i + 1] = clip(p[1]! + dg) * an;
      next[i + 2] = clip(p[2]! + db) * an;
      next[i + 3] = an < 0.04 ? 0 : an;
    }
  }
  state.field.set(next);
}

export function stepSim(state: SimState, ticks = 1): void {
  for (let t = 0; t < ticks; t++) {
    switch (state.kind) {
      case "morphogen":
        stepMorphogen(state);
        break;
      case "spiral":
        stepSpiral(state);
        break;
      case "life":
        stepLife(state);
        break;
      case "neural":
        stepNeural(state);
        break;
      default:
        stepGrow(state);
    }
    state.generation++;
  }
}

export type Palette = { ink: [number, number, number]; paper: [number, number, number]; sage: [number, number, number] };

export const DEFAULT_PALETTE: Palette = {
  ink: [12, 13, 12],
  paper: [232, 230, 220],
  sage: [154, 175, 154],
};

export function renderSim(state: SimState, image: ImageData, palette: Palette = DEFAULT_PALETTE): void {
  const { width: w, height: h, kind } = state;
  const d = image.data;
  const lerp = (a: number[], b: number[], t: number) => [
    a[0]! + (b[0]! - a[0]!) * t,
    a[1]! + (b[1]! - a[1]!) * t,
    a[2]! + (b[2]! - a[2]!) * t,
  ];
  for (let i = 0; i < w * h; i++) {
    let col = palette.ink;
    if (kind === "life") {
      col = state.cells[i] ? palette.sage : palette.ink;
    } else if (kind === "morphogen") {
      const t = clip(state.v[i]! * 1.6);
      col = lerp(palette.ink, palette.sage, t) as [number, number, number];
      const u = state.u[i]!;
      col = lerp(col, palette.paper, clip((1 - u) * 0.35)) as [number, number, number];
    } else if (kind === "spiral") {
      const t = clip((state.u[i]! + 1.4) / 2.8);
      col = lerp(palette.ink, palette.sage, t) as [number, number, number];
    } else {
      const f = i * 4;
      const a = state.field[f + 3]!;
      col = lerp(palette.ink, [state.field[f]! * 255, state.field[f + 1]! * 255, state.field[f + 2]! * 255], a) as [
        number,
        number,
        number,
      ];
    }
    const o = i * 4;
    d[o] = col[0]!;
    d[o + 1] = col[1]!;
    d[o + 2] = col[2]!;
    d[o + 3] = 255;
  }
}
