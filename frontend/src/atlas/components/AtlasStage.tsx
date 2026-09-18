import { useMemo } from "react";
import { extend } from "@pixi/react";
import { Container, Graphics } from "pixi.js";

import type { AtlasSnapshot } from "../types";

// Register Pixi display objects for @pixi/react v8.
extend({
  Container,
  Graphics,
});

interface AtlasStageProps {
  atlas: AtlasSnapshot;
}

export function AtlasStage({ atlas }: AtlasStageProps) {
  const nodes = useMemo(() => atlas.nodes, [atlas.nodes]);
  const edges = useMemo(() => atlas.edges, [atlas.edges]);

  return (
    <pixiContainer>
      {/* Temporary render test */}
      <pixiGraphics
        draw={(graphics) => {
          graphics.clear();

          graphics
            .circle(400, 300, 30)
            .fill({
              color: 0xff0000,
            });
        }}
      />

      {/* Edges */}
      <pixiGraphics
        draw={(graphics) => {
          graphics.clear();

          for (const edge of edges) {
            const source = nodes.find((node) => node.id === edge.source);
            const target = nodes.find((node) => node.id === edge.target);

            if (!source || !target) {
              continue;
            }

            graphics
              .moveTo(source.x, source.y)
              .lineTo(target.x, target.y)
              .stroke({
                width: 1,
                color: 0xc8c3b8,
                alpha: Math.min(0.35, edge.weight * 0.05),
              });
          }
        }}
      />

      {/* Nodes */}
      <pixiGraphics
        draw={(graphics) => {
          graphics.clear();

          for (const node of nodes) {
            const radius = Math.max(
              2,
              Math.min(
                12,
                2 + Math.sqrt(node.weighted_degree || 0),
              ),
            );

            graphics
              .circle(node.x, node.y, radius)
              .fill({
                color: 0x3f4548,
                alpha: 0.9,
              });
          }
        }}
      />
    </pixiContainer>
  );
}