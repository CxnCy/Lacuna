import { Application } from "@pixi/react";

import { ATLAS } from "../constants";
import { useAtlas } from "../hooks/useAtlas";
import { AtlasStage } from "./AtlasStage";

export function AtlasCanvas() {
  const { snapshot, loading, error } = useAtlas();

  if (loading) {
    return <div>Loading Atlas…</div>;
  }

  if (error) {
    return <div>{error}</div>;
  }

  if (!snapshot) {
    return <div>No atlas loaded.</div>;
  }

  return (
    <Application
      resizeTo={window}
      background={ATLAS.backgroundColor}
      antialias
    >
      <AtlasStage atlas={snapshot} />
    </Application>
  );
}