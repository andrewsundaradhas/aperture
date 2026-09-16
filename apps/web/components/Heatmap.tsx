"use client";

import { useEffect, useMemo, useState } from "react";
import { blobUrl } from "@/lib/api";

type HeatmapDoc = {
  simulated: boolean;
  grid_size: number;
  frames: { t: number; grid: number[][] }[];
};

// A single-hue sequential ramp, pale sage → forest ink, monotonic in lightness so "darker =
// more attention" reads without a key. A multi-hue ramp (viridis and friends) would be a
// rainbow in a system that has exactly one hue, and rainbows imply category boundaries that
// a continuous attention weight does not have.
const STOPS = [
  [240, 244, 241],
  [168, 207, 176],
  [77, 165, 109],
  [11, 107, 69],
  [9, 53, 46],
];

function heatColor(v: number): string {
  const x = Math.max(0, Math.min(1, v)) * (STOPS.length - 1);
  const i = Math.floor(x);
  const f = x - i;
  const a = STOPS[i];
  const b = STOPS[Math.min(i + 1, STOPS.length - 1)];
  const c = a.map((av, k) => Math.round(av + (b[k] - av) * f));
  return `rgb(${c[0]}, ${c[1]}, ${c[2]})`;
}

export function Heatmap({ uri }: { uri: string | null }) {
  const [doc, setDoc] = useState<HeatmapDoc | null>(null);
  const [frame, setFrame] = useState(0);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!uri) return;
    setError(null);
    setFrame(0);
    // Storage hands back `local://` / `r2://` URIs — blobUrl routes them through the API's
    // blob endpoint. Concatenating them onto the base URL yields an unfetchable string.
    fetch(blobUrl(uri))
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
      .then(setDoc)
      .catch((e) => setError(e instanceof Error ? e.message : String(e)));
  }, [uri]);

  const gradient = useMemo(
    () => `linear-gradient(to right, ${STOPS.map((s) => `rgb(${s.join(",")})`).join(", ")})`,
    [],
  );

  if (!uri)
    return (
      <div className="flex h-[200px] flex-col items-center justify-center gap-1 text-center">
        <div className="text-body text-slate-smoke">No attention map yet</div>
        <div className="muoto text-caption text-slate-smoke">Run attribution to generate one.</div>
      </div>
    );
  if (error)
    return (
      <div className="flex h-[200px] items-center justify-center text-body text-alarm">
        Couldn’t load the attention map: {error}
      </div>
    );
  if (!doc)
    return (
      <div className="space-y-3">
        <div className="skeleton mx-auto aspect-square w-[240px] rounded-card" />
        <div className="skeleton mx-auto h-2 w-32" />
      </div>
    );

  const current = doc.frames[frame];
  const grid = current?.grid ?? [];
  const max = Math.max(...grid.flat(), 1e-9);

  return (
    <div className="space-y-3">
      <div
        className="mx-auto grid gap-px overflow-hidden rounded-tag bg-lichen"
        style={{ gridTemplateColumns: `repeat(${doc.grid_size}, 1fr)`, maxWidth: 280 }}
        role="img"
        aria-label={`Attention rollout at frame ${current?.t ?? 0}, ${doc.grid_size}×${doc.grid_size} grid`}
      >
        {grid.flatMap((row, y) =>
          row.map((v, x) => (
            <div
              key={`${y}-${x}`}
              style={{ background: heatColor(v / max), aspectRatio: "1 / 1" }}
              title={`(${x}, ${y}) · ${(v / max).toFixed(2)} of peak`}
            />
          )),
        )}
      </div>

      {/* Scale key — without it the colours are decoration, not data. */}
      <div className="mx-auto flex max-w-[280px] items-center gap-2">
        <span className="muoto text-caption text-slate-smoke">low</span>
        <div className="h-1.5 flex-1 rounded-full" style={{ background: gradient }} aria-hidden />
        <span className="muoto text-caption text-slate-smoke">peak</span>
      </div>

      {doc.frames.length > 1 && (
        <div className="flex items-center gap-2">
          <label htmlFor="heatmap-frame" className="sr-only">
            Attention map frame
          </label>
          <input
            id="heatmap-frame"
            type="range"
            min={0}
            max={doc.frames.length - 1}
            value={frame}
            onChange={(e) => setFrame(Number(e.target.value))}
            className="w-full"
            aria-valuetext={`frame ${current?.t ?? frame}`}
          />
        </div>
      )}

      <div className="flex items-center justify-between text-caption text-slate-smoke">
        <span className="muoto">
          frame {current?.t ?? 0}
          <span className="text-slate-smoke"> / {doc.frames.length - 1}</span>
        </span>
        {doc.simulated && (
          <span
            className="tag"
            title="No GPU worker is configured, so the map is synthesized. Run ml/notebooks/attention_rollout.ipynb for real rollout."
          >
            simulated
          </span>
        )}
      </div>
    </div>
  );
}
