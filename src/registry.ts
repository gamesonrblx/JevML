import type { PrimitiveMeta } from "./types.ts";

export const PRIMITIVES: PrimitiveMeta[] = [
  {
    id: "pca",
    name: "PCA",
    kind: "transform",
    summary: "Finds the most important patterns in data.",
    description:
      "Principal component analysis centers a table, builds its covariance, and rotates it onto orthogonal axes ordered by variance. Use it to compress, denoise, or see the grain of a dataset.",
    whenToUse:
      "Tabular or vector data where you want axes of variation, compression, reconstruction, or a first look at structure. Dimensionality reduction before a downstream model. Whitening. Visualizing correlated sensors.",
    whenNotToUse:
      "Sequences, graphs, or generative tasks. Nonlinear manifolds (consider a kernel or an autoencoder). Classification with a known label as the goal.",
    inputs: [
      { name: "X", type: "number[][]", description: "n × d matrix of observations." },
      { name: "k", type: "number?", description: "How many components to keep." },
    ],
    outputs: [
      { name: "components", type: "number[][]", description: "Principal axes, unit length, orthogonal." },
      { name: "scores", type: "number[][]", description: "Coordinates in the new basis." },
      { name: "explainedVarianceRatio", type: "number[]", description: "Share of variance on each axis." },
    ],
    tags: ["linear", "reduction", "covariance", "compression", "visualization", "tabular"],
    references: [{ title: "Pearson, 1901", href: "https://en.wikipedia.org/wiki/Principal_component_analysis" }],
    labPath: "/primitives/pca",
  },
  {
    id: "mcmc",
    name: "MCMC",
    kind: "sampler",
    summary: "Samples possible solutions from a density you can only evaluate.",
    description:
      "Metropolis–Hastings walks a parameter space with a symmetric proposal. Each step is accepted with the likelihood ratio, so the chain spends its time where the target is high. Use it when you can score a candidate but cannot draw one directly.",
    whenToUse:
      "Bayesian posteriors, inverse problems, multimodal densities, energy landscapes. Anytime the object of interest is a distribution, not a point.",
    whenNotToUse:
      "Point estimates with a convex loss (use an optimizer). Very high dimension without a better proposal. Streaming data that needs a closed-form update.",
    inputs: [
      { name: "logProb", type: "(x: number[]) => number", description: "Unnormalized log density." },
      { name: "stepSize", type: "number", description: "Gaussian proposal scale." },
      { name: "nSamples", type: "number", description: "Length of the chain." },
    ],
    outputs: [
      { name: "samples", type: "number[][]", description: "Draws from the target (after mixing)." },
      { name: "acceptanceRate", type: "number", description: "Fraction of proposals kept." },
    ],
    tags: ["bayesian", "sampling", "posterior", "inference", "energy", "stochastic"],
    references: [
      { title: "Metropolis et al., 1953", href: "https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm" },
    ],
    labPath: "/primitives/mcmc",
  },
  {
    id: "text-diffusion",
    name: "Text diffusion",
    kind: "generator",
    summary: "Generates or edits text through repeated refinement.",
    description:
      "A discrete diffusion process treats a string as a noisy lattice of symbols. The reverse process iteratively unmasks characters using a local language model, optionally guided by a target. Useful for infill, restoration, and stepwise editing.",
    whenToUse:
      "Restore corrupted text, fill a masked span, or show iterative refinement. Constrained generation where you want to see the path, not only the end.",
    whenNotToUse:
      "Long-form writing that needs a large language model. Tasks that are really classification or retrieval.",
    inputs: [
      { name: "target", type: "string", description: "Prompt or clean sentence for restore/infill." },
      { name: "mode", type: "'restore' | 'infill' | 'generate'", description: "Which reverse process to run." },
      { name: "steps", type: "number", description: "How many refinement rounds." },
    ],
    outputs: [
      { name: "steps", type: "DiffuseStep[]", description: "The string at each noise level." },
    ],
    tags: ["text", "generative", "denoising", "infill", "language", "iterative"],
    references: [
      { title: "Austin et al., D3PM", href: "https://arxiv.org/abs/2107.03006" },
    ],
    labPath: "/primitives/text-diffusion",
  },
  {
    id: "nca",
    name: "Neural cellular automata",
    kind: "simulator",
    summary: "Builds complex behavior from simple local rules.",
    description:
      "Each cell sees only a small neighborhood (identity + Sobel, or a Laplacian) and applies a tiny update — an MLP residual, a reaction–diffusion step, or a discrete life rule. Form, spiral, and morphogen appear without a global planner.",
    whenToUse:
      "Morphogenesis, texture, pattern formation, local multi-agent dynamics, growing a structure from a seed. Teaching the idea that complexity can be local.",
    whenNotToUse:
      "Global optimization, tabular prediction, or anything that needs long-range attention in one shot.",
    inputs: [
      { name: "genome", type: "GenomeId", description: "Which local rule: morphogen, spiral, grow, neural, life." },
      { name: "seed", type: "grid", description: "Initial field, or a painted blob." },
    ],
    outputs: [
      { name: "field", type: "grid", description: "The grid after n local updates." },
    ],
    tags: ["cellular", "morphogenesis", "local", "simulation", "texture", "self-organization"],
    references: [
      { title: "Mordvintsev et al., Growing NCA", href: "https://distill.pub/2020/growing-ca/" },
      { title: "Pearson, Gray–Scott", href: "https://www.karlsims.com/rd.html" },
    ],
    labPath: "/primitives/nca",
  },
];

export function getPrimitive(id: string): PrimitiveMeta | undefined {
  return PRIMITIVES.find((p) => p.id === id);
}

export function catalogForPrompt(): string {
  return PRIMITIVES.map(
    (p) =>
      `- ${p.id} (${p.kind}): ${p.summary}\n  use: ${p.whenToUse}\n  avoid: ${p.whenNotToUse}\n  tags: ${p.tags.join(", ")}`,
  ).join("\n");
}
