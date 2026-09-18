import { useEffect, useState } from "react";
import { loadManifest, loadSnapshot } from "../services/atlasLoader";
import type { AtlasManifest, AtlasSnapshot } from "../types";

export function useAtlas() {
  const [manifest, setManifest] = useState<AtlasManifest | null>(null);
  const [snapshot, setSnapshot] = useState<AtlasSnapshot | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function initialise() {
      try {
        const manifestData = await loadManifest();
        setManifest(manifestData);

        const latest = manifestData.states.at(-1);

        if (!latest) {
          throw new Error("Atlas manifest contains no states.");
        }

        const snapshotData = await loadSnapshot(latest.file);
        setSnapshot(snapshotData);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Unknown error");
      } finally {
        setLoading(false);
      }
    }

    initialise();
  }, []);

  return {
    manifest,
    snapshot,
    loading,
    error,
  };
}