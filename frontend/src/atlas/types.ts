export interface AtlasNode {
  id: string;
  name: string;
  x: number;
  y: number;
  degree: number;
  weighted_degree: number;
  community_id: number;
  hierarchy: unknown;
}

export interface AtlasEdge {
  source: string;
  target: string;
  weight: number;
}

export interface AtlasSnapshot {
  cutoff_year: number;
  schema_version: string;
  metadata: Record<string, unknown>;
  nodes: AtlasNode[];
  edges: AtlasEdge[];
}

export interface AtlasState {
  cutoff_year: number;
  file: string;
  node_count: number;
  edge_count: number;
}

export interface AtlasManifest {
  atlas_type: string;
  community_semantics: string;
  coordinate_system: string;
  layout_policy: string;
  prediction_overlay_included: boolean;
  schema_version: string;
  states: AtlasState[];
}