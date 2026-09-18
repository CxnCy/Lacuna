import type { AtlasSnapshot } from "../types";

interface AtlasStageProps {
  atlas: AtlasSnapshot;
}

export function AtlasStage({ atlas }: AtlasStageProps) {
  console.log(
    `Loaded atlas ${atlas.cutoff_year}: ${atlas.nodes.length} nodes, ${atlas.edges.length} edges`
  );

  return null;
}