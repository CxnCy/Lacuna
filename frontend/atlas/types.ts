/**
 * Lacuna Atlas Types
 *
 * Shared frontend representation of atlas artifacts generated
 * by the backend export pipeline.
 */

export interface AtlasNode {
  id: string;
  label: string;

  x: number;
  y: number;

  size: number;

  community: number | null;
  hierarchy: string | null;
}

export interface AtlasEdge {
  source: string;
  target: string;
  weight: number;
}

export interface AtlasSnapshot {
  year: number;

  nodes: AtlasNode[];

  edges: AtlasEdge[];
}

export interface AtlasManifestEntry {
  year: number;
  file: string;
}

export interface AtlasManifest {
  snapshots: AtlasManifestEntry[];
}