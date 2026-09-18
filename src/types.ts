export type PrimitiveKind = "transform" | "sampler" | "generator" | "simulator";

export type ParamSpec = {
  name: string;
  type: string;
  description: string;
};

export type PrimitiveMeta = {
  id: string;
  name: string;
  kind: PrimitiveKind;
  summary: string;
  description: string;
  whenToUse: string;
  whenNotToUse: string;
  inputs: ParamSpec[];
  outputs: ParamSpec[];
  tags: string[];
  references: { title: string; href: string }[];
  /** Path in the JevML lab UI. */
  labPath: string;
};

export type HarnessSelection = {
  primitiveId: string;
  confidence: number;
  rationale: string;
  suggestedParams: Record<string, string | number>;
  alternatives: { primitiveId: string; why: string }[];
  source: "jev" | "local";
};
