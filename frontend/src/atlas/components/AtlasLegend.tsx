import React from "react";

export interface AtlasLegendItem {
  label: string;
  description: string;
  size: number;
}

interface AtlasLegendProps {
  items?: AtlasLegendItem[];
}

const DEFAULT_ITEMS: AtlasLegendItem[] = [
  {
    label: "Research topic",
    description: "Individual research areas",
    size: 7,
  },
  {
    label: "Dense research",
    description: "Higher research activity",
    size: 13,
  },
  {
    label: "Research frontier",
    description: "Emerging activity",
    size: 19,
  },
];

export function AtlasLegend({
  items = DEFAULT_ITEMS,
}: AtlasLegendProps) {
  return (
    <aside
      aria-label="Atlas legend"
      style={{
        position: "absolute",
        left: 20,
        bottom: 20,
        padding: "14px 16px",
        background: "rgba(250, 248, 243, 0.94)",
        border: "1px solid rgba(40, 40, 40, 0.14)",
        borderRadius: 6,
        boxShadow: "0 4px 18px rgba(0, 0, 0, 0.08)",
        fontFamily: "inherit",
        color: "#303030",
        minWidth: 210,
      }}
    >
      <div
        style={{
          fontSize: 11,
          fontWeight: 700,
          letterSpacing: "0.12em",
          textTransform: "uppercase",
          marginBottom: 12,
        }}
      >
        Atlas key
      </div>

      {items.map((item) => (
        <div
          key={item.label}
          style={{
            display: "flex",
            alignItems: "center",
            gap: 10,
            marginBottom: 9,
          }}
        >
          <span
            aria-hidden="true"
            style={{
              width: item.size,
              height: item.size,
              minWidth: item.size,
              borderRadius: "50%",
              background: "#4d5558",
              display: "inline-block",
            }}
          />

          <div>
            <div
              style={{
                fontSize: 12,
                fontWeight: 600,
                lineHeight: 1.2,
              }}
            >
              {item.label}
            </div>

            <div
              style={{
                fontSize: 10,
                color: "#747474",
                marginTop: 2,
              }}
            >
              {item.description}
            </div>
          </div>
        </div>
      ))}
    </aside>
  );
}

export default AtlasLegend;