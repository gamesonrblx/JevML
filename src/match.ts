import { PRIMITIVES } from "./registry.ts";
import type { HarnessSelection } from "./types.ts";

const WEIGHT: Record<string, Partial<Record<string, number>>> = {
  pca: {
    pca: 8,
    principal: 6,
    component: 5,
    variance: 4,
    dimension: 4,
    compress: 4,
    reduce: 3,
    tabular: 3,
    correlat: 3,
    axis: 3,
    pattern: 2,
    cluster: 1,
    sensor: 2,
    matrix: 2,
  },
  mcmc: {
    mcmc: 8,
    posterior: 6,
    bayes: 6,
    sample: 5,
    density: 4,
    likelihood: 4,
    metropolis: 6,
    energy: 3,
    multimodal: 4,
    infer: 3,
    walk: 2,
    chain: 3,
    distribution: 3,
  },
  "text-diffusion": {
    text: 6,
    sentence: 4,
    word: 3,
    diffus: 7,
    denois: 6,
    infill: 6,
    mask: 4,
    edit: 3,
    generat: 3,
    restor: 5,
    language: 3,
    string: 3,
    character: 2,
  },
  nca: {
    cell: 6,
    automata: 7,
    morphogen: 6,
    grow: 4,
    local: 3,
    grid: 4,
    pattern: 3,
    texture: 4,
    seed: 2,
    spiral: 3,
    life: 4,
    reaction: 4,
    neighbor: 4,
    organism: 3,
    simulate: 3,
  },
};

function tokens(task: string): string[] {
  return task
    .toLowerCase()
    .split(/[^a-z0-9+]+/)
    .filter((t) => t.length > 1);
}

export function scoreTask(task: string): { id: string; score: number }[] {
  const toks = tokens(task);
  return PRIMITIVES.map((p) => {
    const w = WEIGHT[p.id] ?? {};
    let s = 0;
    for (const t of toks) {
      for (const [k, v] of Object.entries(w)) {
        if (t.includes(k) || k.includes(t)) s += v ?? 0;
      }
      for (const tag of p.tags) {
        if (t === tag || tag.includes(t)) s += 1.5;
      }
    }
    return { id: p.id, score: s };
  }).sort((a, b) => b.score - a.score);
}

export function matchLocal(task: string): HarnessSelection {
  const ranked = scoreTask(task);
  const best = ranked[0] ?? { id: "pca", score: 0 };
  const primitive = PRIMITIVES.find((p) => p.id === best.id)!;
  const second = ranked[1];
  const conf =
    best.score <= 0
      ? 0.34
      : Math.min(0.86, 0.42 + best.score / (best.score + (second?.score ?? 0) + 6));
  return {
    primitiveId: primitive.id,
    confidence: Number(conf.toFixed(2)),
    rationale: `Matched on task language against ${primitive.name}'s tags and use-cases. ${primitive.whenToUse.split(".")[0]}.`,
    suggestedParams: defaultParams(primitive.id),
    alternatives: ranked.slice(1, 3).map((r) => ({
      primitiveId: r.id,
      why: PRIMITIVES.find((p) => p.id === r.id)?.summary ?? "",
    })),
    source: "local",
  };
}

export function defaultParams(id: string): Record<string, string | number> {
  switch (id) {
    case "pca":
      return { dataset: "elongated", k: 2 };
    case "mcmc":
      return { target: "mixture", stepSize: 0.35 };
    case "text-diffusion":
      return { mode: "restore", steps: 18 };
    case "nca":
      return { genome: "morphogen" };
    default:
      return {};
  }
}
