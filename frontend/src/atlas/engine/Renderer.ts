/**
 * Lacuna Atlas Renderer
 *
 * Owns the PixiJS rendering application.
 *
 * Responsibilities:
 * - Create and manage the Pixi application
 * - Attach the canvas to the DOM
 * - Resize the renderer with its container
 * - Expose the Pixi stage
 * - Start and stop rendering
 *
 * Camera mathematics intentionally lives in Camera.ts.
 * Atlas drawing intentionally lives in AtlasStage.tsx.
 */

import { Application, Container } from "pixi.js";

export interface RendererOptions {
  backgroundColor?: number;
  antialias?: boolean;
  resolution?: number;
}

export class Renderer {
  private readonly app: Application;
  private readonly options: RendererOptions;

  private initialized = false;

  constructor(options: RendererOptions = {}) {
    this.options = {
      backgroundColor: 0xf4f1e8,
      antialias: true,
      resolution: window.devicePixelRatio || 1,
      ...options,
    };

    this.app = new Application();
  }

  /**
   * Pixi application instance.
   */
  get application(): Application {
    return this.app;
  }

  /**
   * Root Pixi stage.
   */
  get stage(): Container {
    return this.app.stage;
  }

  /**
   * Render canvas.
   */
  get canvas(): HTMLCanvasElement {
    return this.app.canvas;
  }

  /**
   * Initialise Pixi and attach the canvas to a DOM element.
   */
  async initialize(container: HTMLElement): Promise<void> {
    if (this.initialized) {
      return;
    }

    await this.app.init({
      background: this.options.backgroundColor,
      antialias: this.options.antialias,
      resolution: this.options.resolution,
      resizeTo: container,
    });

    container.appendChild(this.app.canvas);

    this.app.canvas.style.display = "block";
    this.app.canvas.style.width = "100%";
    this.app.canvas.style.height = "100%";

    this.initialized = true;
  }

  /**
   * Resize the renderer to match its container.
   */
  resize(width: number, height: number): void {
    if (!this.initialized) {
      return;
    }

    this.app.renderer.resize(width, height);
  }

  /**
   * Start Pixi's render loop.
   */
  start(): void {
    if (!this.initialized) {
      return;
    }

    this.app.start();
  }

  /**
   * Stop Pixi's render loop.
   */
  stop(): void {
    if (!this.initialized) {
      return;
    }

    this.app.stop();
  }

  /**
   * Destroy the renderer and release Pixi resources.
   */
  destroy(): void {
    if (!this.initialized) {
      return;
    }

    this.app.destroy(true, {
      children: true,
      texture: true,
    });

    this.initialized = false;
  }
}