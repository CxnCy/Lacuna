export interface AtlasNode {
  id: string;
  label: string;
  x: number;
  y: number;
  size: number;
  domain?: string;
}

export interface AtlasEdge {
  source: string;
  target: string;
  weight?: number;
}

export interface AtlasSnapshot {
  year: number;
  nodes: AtlasNode[];
  edges: AtlasEdge[];
}

export interface AtlasManifest {
  snapshots: {
    year: number;
    file: string;
  }[];
}