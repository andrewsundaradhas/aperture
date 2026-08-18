"use client";

import { useEffect, useState } from "react";
import { API_BASE } from "@/lib/api";

type HeatmapDoc = {
  simulated: boolean;
  grid_size: number;
  frames: { t: number; grid: number[][] }[];
};

function heatColor(v: number): string {
  // simple viridis-ish ramp
  const stops = [
    [13, 8, 135],
    [126, 3, 168],
    [204, 71, 120],
    [248, 149, 64],
    [240, 249, 33],
  ];
  const x = Math.max(0, Math.min(1, v)) * (stops.length - 1);
  const i = Math.floor(x);
  const f = x - i;
  const a = stops[i];
  const b = stops[Math.min(i + 1, stops.length - 1)];
  const c = a.map((av, k) => Math.round(av + (b[k] - av) * f));
  return `rgb(${c[0]}, ${c[1]}, ${c[2]})`;
}

export function Heatmap({ uri }: { uri: string | null }) {
  const [doc, setDoc] = useState<HeatmapDoc | null>(null);
  const [frame, setFrame] = useState(0);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!uri) return;
    const url = uri.startsWith("http") ? uri : `${API_BASE}${uri}`;
    fetch(url)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`${r.status}`))))
      .then(setDoc)
      .catch((e) => setError(String(e)));
  }, [uri]);

  if (!uri) return <div className="text-sm text-muted">No attribution yet — run it above.</div>;
  if (error) return <div className="text-sm text-motor">Failed to load heatmap: {error}</div>;
  if (!doc) return <div className="text-sm text-muted">Loading heatmap…</div>;

  const grid = doc.frames[frame]?.grid ?? [];
  const max = Math.max(...grid.flat(), 1e-9);

  return (
    <div className="space-y-3">
      <div
        className="grid gap-px bg-edge rounded overflow-hidden mx-auto"
        style={{ gridTemplateColumns: `repeat(${doc.grid_size}, 1fr)`, maxWidth: 280 }}
      >
        {grid.flatMap((row, y) =>
          row.map((v, x) => (
            <div key={`${y}-${x}`} style={{ background: heatColor(v / max), aspectRatio: "1 / 1" }} />
          ))
        )}
      </div>
      {doc.frames.length > 1 && (
        <input
          type="range"
          min={0}
          max={doc.frames.length - 1}
          value={frame}
          onChange={(e) => setFrame(Number(e.target.value))}
          className="w-full accent-accent"
        />
      )}
      <div className="text-xs text-muted flex justify-between">
        <span>frame t={doc.frames[frame]?.t}</span>
        {doc.simulated && <span className="text-perception">simulated (no GPU worker) — see notebook</span>}
      </div>
    </div>
  );
}
