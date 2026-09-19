import { useEffect, useMemo, useRef, useState } from "react";
import { extend } from "@pixi/react";
import { Container, Graphics } from "pixi.js";

import type { AtlasSnapshot } from "../types";
import { Camera } from "../engine/Camera";

extend({
  Container,
  Graphics,
});

interface AtlasStageProps {
  atlas: AtlasSnapshot;
}

export function AtlasStage({ atlas }: AtlasStageProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const cameraRef = useRef(
    new Camera({
      zoom: 1,
      minZoom: 0.25,
      maxZoom: 8,
    }),
  );

  const [cameraVersion, setCameraVersion] = useState(0);

  const draggingRef = useRef(false);
  const lastPointerRef = useRef({ x: 0, y: 0 });

  const { nodeMap, bounds } = useMemo(() => {
    const nodeMap = new Map(atlas.nodes.map((node) => [node.id, node]));

    const xs = atlas.nodes.map((node) => node.x);
    const ys = atlas.nodes.map((node) => node.y);

    return {
      nodeMap,
      bounds: {
        minX: Math.min(...xs),
        maxX: Math.max(...xs),
        minY: Math.min(...ys),
        maxY: Math.max(...ys),
      },
    };
  }, [atlas.nodes]);

  /*
   * Centre the camera on the Atlas when the snapshot changes.
   */
  useEffect(() => {
    const centerX = (bounds.minX + bounds.maxX) / 2;
    const centerY = (bounds.minY + bounds.maxY) / 2;

    cameraRef.current.reset(centerX, centerY, 1);
    setCameraVersion((version) => version + 1);
  },[atlas.cutoff_year, bounds]);

  /*
   * Attach interaction directly to the Pixi canvas.
   *
   * Pointer drag = pan
   * Wheel = zoom
   */
  useEffect(() => {
    const canvas = document.querySelector("canvas");

    if (!canvas) {
      return;
    }

    canvasRef.current = canvas;

    const handlePointerDown = (event: PointerEvent) => {
      draggingRef.current = true;

      lastPointerRef.current = {
        x: event.clientX,
        y: event.clientY,
      };

      canvas.style.cursor = "grabbing";
      canvas.setPointerCapture?.(event.pointerId);
    };

    const handlePointerMove = (event: PointerEvent) => {
      if (!draggingRef.current) {
        return;
      }

      const dx = event.clientX - lastPointerRef.current.x;
      const dy = event.clientY - lastPointerRef.current.y;

      const camera = cameraRef.current;

      /*
       * Screen movement is converted back into world movement.
       */
      camera.pan(
        -dx / camera.scale,
        -dy / camera.scale,
      );

      lastPointerRef.current = {
        x: event.clientX,
        y: event.clientY,
      };

      setCameraVersion((version) => version + 1);
    };

    const stopDragging = () => {
      draggingRef.current = false;
      canvas.style.cursor = "grab";
    };

    const handleWheel = (event: WheelEvent) => {
      event.preventDefault();

      const rect = canvas.getBoundingClientRect();

      const screenPoint = {
        x: event.clientX - rect.left,
        y: event.clientY - rect.top,
      };

      const viewport = {
        width: rect.width,
        height: rect.height,
      };

      const camera = cameraRef.current;

      /*
       * Find the world point underneath the cursor before zooming.
       */
      const worldPoint = camera.screenToWorld(
        screenPoint,
        viewport,
      );

      /*
       * Smooth multiplicative zoom.
       */
      const zoomFactor = Math.exp(-event.deltaY * 0.0015);

      camera.zoomBy(zoomFactor);

      /*
       * Keep the same world point underneath the cursor.
       */
      const newWorldPoint = camera.screenToWorld(
        screenPoint,
        viewport,
      );

      camera.pan(
        worldPoint.x - newWorldPoint.x,
        worldPoint.y - newWorldPoint.y,
      );

      setCameraVersion((version) => version + 1);
    };

    canvas.style.cursor = "grab";
    canvas.style.touchAction = "none";

    canvas.addEventListener("pointerdown", handlePointerDown);
    canvas.addEventListener("pointermove", handlePointerMove);
    canvas.addEventListener("pointerup", stopDragging);
    canvas.addEventListener("pointercancel", stopDragging);
    canvas.addEventListener("pointerleave", stopDragging);
    canvas.addEventListener("wheel", handleWheel, {
      passive: false,
    });

    return () => {
      canvas.removeEventListener("pointerdown", handlePointerDown);
      canvas.removeEventListener("pointermove", handlePointerMove);
      canvas.removeEventListener("pointerup", stopDragging);
      canvas.removeEventListener("pointercancel", stopDragging);
      canvas.removeEventListener("pointerleave", stopDragging);
      canvas.removeEventListener("wheel", handleWheel);

      canvas.style.cursor = "";
      canvas.style.touchAction = "";
    };
  }, []);

  const project = (x: number, y: number) => {
    const canvas = canvasRef.current;

    const width = canvas?.clientWidth ?? window.innerWidth;
    const height = canvas?.clientHeight ?? window.innerHeight;

    return cameraRef.current.worldToScreen(
      { x, y },
      { width, height },
    );
  };

  /*
   * cameraVersion intentionally causes the Pixi drawing callbacks
   * to receive the latest camera state after interaction.
   */
  void cameraVersion;

  return (
    <pixiContainer>
      {/* Research relationships */}
      <pixiGraphics
        draw={(graphics) => {
          graphics.clear();

          for (const edge of atlas.edges) {
            const source = nodeMap.get(edge.source);
            const target = nodeMap.get(edge.target);

            if (!source || !target) {
              continue;
            }

            const sourcePoint = project(
              source.x,
              source.y,
            );

            const targetPoint = project(
              target.x,
              target.y,
            );

            graphics
              .moveTo(sourcePoint.x, sourcePoint.y)
              .lineTo(targetPoint.x, targetPoint.y)
              .stroke({
                width: 1,
                color: 0xc8c3b8,
                alpha: Math.min(
                  0.22,
                  edge.weight * 0.04,
                ),
              });
          }
        }}
      />

      {/* Research topics */}
      <pixiGraphics
        draw={(graphics) => {
          graphics.clear();

          for (const node of atlas.nodes) {
            const point = project(
              node.x,
              node.y,
            );

            const radius = Math.max(
              2,
              Math.min(
                10,
                2 +
                  Math.sqrt(
                    Math.max(
                      0,
                      node.weighted_degree || 0,
                    ),
                  ),
              ),
            );

            graphics
              .circle(
                point.x,
                point.y,
                radius,
              )
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