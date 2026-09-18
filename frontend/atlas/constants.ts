/**
 * Lacuna Atlas Renderer Constants
 *
 * Central place for renderer configuration.
 * These values will evolve as the Atlas gains
 * interaction, animation and styling.
 */

export const ATLAS = {
  /**
   * Default canvas background.
   *
   * Deep neutral—not pure black—to match
   * Lacuna's scientific aesthetic.
   */
  backgroundColor: 0x0f1115,

  /**
   * Node appearance.
   */
  node: {
    radius: 3,
    alpha: 0.9,
  },

  /**
   * Edge appearance.
   */
  edge: {
    alpha: 0.12,
    width: 1,
  },

  /**
   * Camera limits.
   */
  camera: {
    minZoom: 0.15,
    maxZoom: 8,
  },

  /**
   * Animation.
   */
  animation: {
    targetFPS: 60,
  },
} as const;

export const DEFAULT_NODE_COLOR = 0xd9d9d9;

export const DEFAULT_EDGE_COLOR = 0x5d6673;