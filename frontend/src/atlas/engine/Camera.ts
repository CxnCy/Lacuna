/**
 * Lacuna Atlas Camera
 *
 * Maintains the viewport transform between:
 *
 *   World space  →  Screen space
 *
 * The scientific atlas coordinates live in world space.
 * The camera controls how much of that geography is visible.
 *
 * Responsibilities:
 * - Pan
 * - Zoom
 * - Zoom around a cursor position
 * - Coordinate conversion
 * - Camera reset
 *
 * Rendering is intentionally kept outside this class.
 */

export interface Point {
  x: number;
  y: number;
}

export interface CameraOptions {
  /**
   * Initial horizontal camera position in world coordinates.
   */
  x?: number;

  /**
   * Initial vertical camera position in world coordinates.
   */
  y?: number;

  /**
   * Initial zoom level.
   */
  zoom?: number;

  /**
   * Minimum allowed zoom.
   */
  minZoom?: number;

  /**
   * Maximum allowed zoom.
   */
  maxZoom?: number;
}

export interface CameraState {
  x: number;
  y: number;
  zoom: number;
}

export class Camera {
  private x: number;
  private y: number;
  private zoom: number;

  private readonly minZoom: number;
  private readonly maxZoom: number;

  constructor(options: CameraOptions = {}) {
    this.x = options.x ?? 0;
    this.y = options.y ?? 0;

    this.zoom = options.zoom ?? 1;

    this.minZoom = options.minZoom ?? 0.1;
    this.maxZoom = options.maxZoom ?? 20;

    this.zoom = this.clampZoom(this.zoom);
  }

  /**
   * Current camera position in world coordinates.
   */
  get position(): Point {
    return {
      x: this.x,
      y: this.y,
    };
  }

  /**
   * Current zoom factor.
   */
  get scale(): number {
    return this.zoom;
  }

  /**
   * Complete camera state.
   */
  get state(): CameraState {
    return {
      x: this.x,
      y: this.y,
      zoom: this.zoom,
    };
  }

  /**
   * Move the camera by a world-space amount.
   *
   * Positive dx moves the camera right.
   * Positive dy moves the camera down.
   */
  pan(dx: number, dy: number): void {
    this.x += dx;
    this.y += dy;
  }

  /**
   * Set the camera position directly.
   */
  setPosition(x: number, y: number): void {
    this.x = x;
    this.y = y;
  }

  /**
   * Set zoom directly.
   */
  setZoom(zoom: number): void {
    this.zoom = this.clampZoom(zoom);
  }

  /**
   * Zoom by a multiplicative factor.
   *
   * Example:
   *
   *   zoomBy(1.2)  → zoom in by 20%
   *   zoomBy(0.8)  → zoom out by 20%
   */
  zoomBy(factor: number): void {
    if (!Number.isFinite(factor) || factor <= 0) {
      return;
    }

    this.zoom = this.clampZoom(this.zoom * factor);
  }

  /**
   * Zoom around a screen-space point.
   *
   * This is important for natural map interaction:
   * when the user places their cursor over a research area
   * and scrolls, that area remains underneath the cursor.
   *
   * screenPoint:
   *   Position of the cursor in screen coordinates.
   *
   * currentCameraTransform:
   *   Function that converts screen coordinates into world coordinates
   *   before the zoom occurs.
   */
  zoomAt(
    screenPoint: Point,
    factor: number,
    screenToWorld: (point: Point) => Point,
  ): void {
    if (!Number.isFinite(factor) || factor <= 0) {
      return;
    }

    const worldPointBeforeZoom = screenToWorld(screenPoint);

    const previousZoom = this.zoom;

    this.zoom = this.clampZoom(this.zoom * factor);

    if (this.zoom === previousZoom) {
      return;
    }

    /*
     * Keep the world point under the cursor stationary.
     *
     * The camera position therefore changes according to the
     * difference between the old and new zoom levels.
     */
    const zoomRatio = previousZoom / this.zoom;

    this.x =
      worldPointBeforeZoom.x -
      (worldPointBeforeZoom.x - this.x) * zoomRatio;

    this.y =
      worldPointBeforeZoom.y -
      (worldPointBeforeZoom.y - this.y) * zoomRatio;
  }

  /**
   * Convert a world-space coordinate into screen space.
   *
   * screen = (world - camera) * zoom
   */
  worldToScreen(
    worldPoint: Point,
    viewport: { width: number; height: number },
  ): Point {
    return {
      x:
        (worldPoint.x - this.x) * this.zoom +
        viewport.width / 2,

      y:
        (worldPoint.y - this.y) * this.zoom +
        viewport.height / 2,
    };
  }

  /**
   * Convert a screen-space coordinate into world space.
   *
   * This is the inverse of worldToScreen().
   */
  screenToWorld(
    screenPoint: Point,
    viewport: { width: number; height: number },
  ): Point {
    return {
      x:
        (screenPoint.x - viewport.width / 2) / this.zoom +
        this.x,

      y:
        (screenPoint.y - viewport.height / 2) / this.zoom +
        this.y,
    };
  }

  /**
   * Reset the camera to the supplied position and zoom.
   */
  reset(
    x = 0,
    y = 0,
    zoom = 1,
  ): void {
    this.x = x;
    this.y = y;
    this.zoom = this.clampZoom(zoom);
  }

  /**
   * Restore a previously captured camera state.
   */
  restore(state: CameraState): void {
    this.x = state.x;
    this.y = state.y;
    this.zoom = this.clampZoom(state.zoom);
  }

  /**
   * Capture the current camera state.
   */
  snapshot(): CameraState {
    return {
      x: this.x,
      y: this.y,
      zoom: this.zoom,
    };
  }

  /**
   * Clamp zoom to the configured range.
   */
  private clampZoom(zoom: number): number {
    return Math.min(
      this.maxZoom,
      Math.max(this.minZoom, zoom),
    );
  }
}