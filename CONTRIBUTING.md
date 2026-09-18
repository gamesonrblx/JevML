# Contribute a primitive

JevML is a catalog of small verbs, not a framework.

1. **One job.** Project, sample, denoise, grow — not an entire pipeline.
2. **Pure functions.** Inputs in, outputs out. Seeded RNGs. Deterministic when the seed is fixed.
3. **Register it.** Add a `PrimitiveMeta` entry in `src/registry.ts` with `whenToUse` / `whenNotToUse`. The harness reads this.
4. **Ship a test.** Reconstruction error, acceptance rate, a conserved quantity — something that fails if the math is wrong.
5. **Open a pull request.**

```ts
export type PrimitiveMeta = {
  id: string
  name: string
  kind: "transform" | "sampler" | "generator" | "simulator"
  summary: string
  whenToUse: string
  whenNotToUse: string
  tags: string[]
}
```
