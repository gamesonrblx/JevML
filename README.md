# JevML

Reusable machine-learning primitives, designed so **Jev** (or any agent) can pick the right tool for the work.

| Primitive | Kind | Job |
| --- | --- | --- |
| **PCA** | transform | Find the most important patterns in data |
| **MCMC** | sampler | Sample possible solutions from a density you can only evaluate |
| **Text diffusion** | generator | Generate or edit text through repeated refinement |
| **Neural cellular automata** | simulator | Build complex behavior from simple local rules |

## Install

Copy `src/` into a TypeScript project, or import the modules directly:

```ts
import { pca, reconstruct } from "./src/pca.ts";
import { metropolisHastings, getTarget } from "./src/mcmc.ts";
import { runTextDiffusion } from "./src/text-diffusion.ts";
import { createSim, stepSim } from "./src/nca.ts";
import { PRIMITIVES, matchLocal } from "./src/index.ts";
```

No runtime dependencies. Node 22+ or any modern bundler.

## Harness

The catalog is the contract. Jev should:

1. Read `PRIMITIVES` in `src/registry.ts` (id, `whenToUse`, `whenNotToUse`, tags).
2. Choose the primitive that fits the task — `matchLocal(task)` is a keyword fallback; a larger model can use `catalogForPrompt()`.
3. Call the matching function with suggested parameters.

```ts
import { matchLocal, getPrimitive } from "./src/index.ts";

const choice = matchLocal("recover the axes of variation in a sensor table");
const meta = getPrimitive(choice.primitiveId);
// choice.primitiveId === "pca"
```

A running lab also exposes:

- `GET /api/primitives` — JSON catalog
- `POST /api/harness` `{ "task": "..." }` — selection with rationale

## Primitive notes

**PCA** — covariance + Jacobi eigendecomposition. Scores, explained variance, rank-`k` reconstruction.

**MCMC** — Metropolis–Hastings with a Gaussian proposal. Built-in targets: two wells, banana, ring.

**Text diffusion** — discrete unmasking with an n-gram reverse process. Modes: restore, infill, generate.

**NCA** — local update on a torus. Genomes: Gray–Scott morphogen, FitzHugh–Nagumo spirals, grow, a tiny MLP residual, and Life.

## Tests

```sh
node --experimental-strip-types --test src/*.test.ts
```

## License

MIT
