import { mulberry32, type Rng } from "./math.ts";

export const ALPHABET =
  "abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,;:'\"!?-()";

const BASE_CORPUS = `
The shortest path through a high-dimensional cloud is rarely a straight line.
Patterns hide in covariance: stretch, rotate, and the world collapses onto a few axes.
When a density cannot be sampled, walk it. Propose, accept, refuse. The chain remembers.
Text can be treated as a lattice of symbols. Noise a letter, then ask the neighborhood
what should have been there. Repeat, and a sentence condenses out of static.
A cell knows only its neighbors. From that poverty of information, spiral, coral, and
creature still arise. Local rules, global weather.
Machine learning primitives are small verbs. Project. Sample. Denoise. Grow.
Jev chooses the verb that fits the sentence of the task.
Principal components recover the directions of greatest variance.
Metropolis-Hastings draws from densities known only as a black-box log probability.
Discrete diffusion refines a noisy string by iterative unmasking.
Neural cellular automata iterate a tiny network on a grid until form appears.
Reuse is the point. A primitive that is tested, named, and harnessed can be called again.
`;

export type NgramModel = {
  order: number;
  counts: Map<string, Map<string, number>>;
  unigram: Map<string, number>;
};

function pushCount(map: Map<string, number>, key: string, n = 1) {
  map.set(key, (map.get(key) ?? 0) + n);
}

export function trainNgram(corpus: string, order = 3): NgramModel {
  const counts = new Map<string, Map<string, number>>();
  const unigram = new Map<string, number>();
  const text = corpus.replace(/\s+/g, " ").trim();
  for (const ch of text) pushCount(unigram, ch);
  for (let i = 0; i < text.length; i++) {
    const next = text[i]!;
    for (let o = 1; o < order; o++) {
      if (i - o < 0) break;
      const ctx = text.slice(i - o, i);
      let m = counts.get(ctx);
      if (!m) {
        m = new Map();
        counts.set(ctx, m);
      }
      pushCount(m, next);
    }
  }
  return { order, counts, unigram };
}

const DEFAULT_MODEL = trainNgram(BASE_CORPUS, 3);

export function defaultModel(): NgramModel {
  return DEFAULT_MODEL;
}

export function modelWithPrompt(prompt: string): NgramModel {
  return trainNgram(`${BASE_CORPUS}\n${prompt}\n${prompt}\n${prompt}`, 3);
}

function sampleFrom(map: Map<string, number>, rng: Rng, temperature: number): string {
  const entries = [...map.entries()];
  if (entries.length === 0) return " ";
  const invT = 1 / Math.max(temperature, 0.05);
  const max = Math.max(...entries.map(([, n]) => n));
  const weights = entries.map(([, n]) => Math.pow(n / max, invT));
  let z = 0;
  for (const w of weights) z += w;
  let r = rng() * z;
  for (let i = 0; i < entries.length; i++) {
    r -= weights[i]!;
    if (r <= 0) return entries[i]![0];
  }
  return entries[entries.length - 1]![0];
}

function predict(model: NgramModel, left: string, rng: Rng, temperature: number): string {
  for (let o = model.order - 1; o >= 1; o--) {
    const ctx = left.slice(-o);
    const m = model.counts.get(ctx);
    if (m && m.size > 0) return sampleFrom(m, rng, temperature);
  }
  return sampleFrom(model.unigram, rng, temperature);
}

function randomChar(rng: Rng): string {
  const i = Math.floor(rng() * ALPHABET.length);
  return ALPHABET[i] ?? " ";
}

export type DiffuseMode = "restore" | "infill" | "generate";

export type DiffuseStep = {
  t: number;
  text: string;
  /** 0..1 how noisy each character still is. */
  noise: number[];
};

export function runTextDiffusion(opts: {
  target?: string;
  length?: number;
  steps?: number;
  seed?: number;
  mode?: DiffuseMode;
  temperature?: number;
}): DiffuseStep[] {
  const mode = opts.mode ?? "restore";
  const steps = Math.max(4, opts.steps ?? 18);
  const rng = mulberry32(opts.seed ?? 21);
  const target =
    (opts.target ?? "Local rules become weather.").replace(/\s+/g, " ").trim() ||
    "Local rules become weather.";
  const length = mode === "generate" ? opts.length ?? 48 : target.length;
  const model = modelWithPrompt(target);
  const temp0 = opts.temperature ?? 0.85;

  let chars: string[] = [];
  if (mode === "generate") {
    chars = Array.from({ length }, () => randomChar(rng));
  } else if (mode === "infill") {
    const a = Math.floor(target.length * 0.28);
    const b = Math.ceil(target.length * 0.72);
    chars = target.split("").map((ch, i) => (i >= a && i < b ? randomChar(rng) : ch));
  } else {
    chars = Array.from({ length }, () => randomChar(rng));
  }

  const frozen =
    mode === "infill"
      ? target.split("").map((_, i) => {
          const a = Math.floor(target.length * 0.28);
          const b = Math.ceil(target.length * 0.72);
          return i < a || i >= b;
        })
      : chars.map(() => false);

  const out: DiffuseStep[] = [
    { t: 1, text: chars.join(""), noise: chars.map((_, i) => (frozen[i] ? 0 : 1)) },
  ];

  for (let s = 1; s <= steps; s++) {
    const t = 1 - s / steps;
    const temperature = temp0 * (0.25 + 0.75 * t);
    const replaceP = Math.pow(t, 0.65);
    for (let i = 0; i < chars.length; i++) {
      if (frozen[i]) continue;
      if (rng() > replaceP && s > 1) continue;
      const left = chars.slice(Math.max(0, i - 3), i).join("");
      let next = predict(model, left, rng, temperature);
      if (mode !== "generate" && rng() < (1 - t) * 0.55) {
        next = target[i] ?? next;
      }
      chars[i] = next;
    }
    const noise = chars.map((ch, i) => {
      if (frozen[i]) return 0;
      if (mode === "generate") return t;
      return ch === (target[i] ?? "") ? t * 0.25 : Math.min(1, t + 0.35);
    });
    out.push({ t, text: chars.join(""), noise });
  }
  return out;
}
