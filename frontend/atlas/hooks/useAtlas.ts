import { useEffect, useState } from "react";

import { loadLatestSnapshot } from "../services/atlasLoader";
import type { AtlasSnapshot } from "../types";

export function useAtlas() {
  const [atlas, setAtlas] = useState<AtlasSnapshot | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function load() {
      try {
        setLoading(true);

        const snapshot = await loadLatestSnapshot();

        if (!cancelled) {
          setAtlas(snapshot);
          setError(null);
        }
      } catch (err) {
        if (!cancelled) {
          setError(err as Error);
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    }

    load();

    return () => {
      cancelled = true;
    };
  }, []);

  return {
    atlas,
    loading,
    error,
  };
}