"use client";

import { useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import { EpisodeSummary, SURFACE_HEX, surfaceVerdict } from "@/lib/api";
import { SurfaceLegend } from "@/components/ui";

/**
 * The fleet as a scatter plot — the page's hero, drawn straight onto the sage canvas
 * rather than inside a card, per the system's chart-as-page convention.
 *
 * x = when the episode was ingested. y = the classifier's confidence in its failure surface.
 *
 * Successes are deliberately *not* plotted against y: nothing was attributed, so a confidence
 * of zero would be a measurement that was never taken. They sit in their own hollow lane below
 * the axis, which keeps the main field honest — every mark in it is a real verdict.
 */

const W = 1200;
const H = 440;
const PAD = { top: 28, right: 28, bottom: 76, left: 56 };
const LANE_BAND_TOP = H - 66; // top of the fenced "not scored" band
const LANE_Y = H - 52; // where an unscored mark sits before the rug lays it out

type Plotted = {
  ep: EpisodeSummary;
  x: number;
  y: number;
  surface: string | null;
  scored: boolean;
};

export function FleetScatter({ episodes }: { episodes: EpisodeSummary[] }) {
  const router = useRouter();
  const [hover, setHover] = useState<Plotted | null>(null);

  const { points, counts, ticks } = useMemo(() => {
    const times = episodes.map((e) => new Date(e.started_at).getTime());
    const min = Math.min(...times);
    const max = Math.max(...times);
    const span = max - min || 1;

    const plotW = W - PAD.left - PAD.right;
    const plotH = H - PAD.top - PAD.bottom;

    const raw: Plotted[] = episodes.map((e, i) => {
      const v = surfaceVerdict(e.surface, e.classification_confidence, e.outcome);
      const scored = v.kind === "surface";
      const t = new Date(e.started_at).getTime();
      // When every episode shares a timestamp (a seeded fixture set), spread by index so
      // the marks stay readable instead of stacking into one column.
      const frac =
        span > 1 ? (t - min) / span : episodes.length > 1 ? i / (episodes.length - 1) : 0.5;
      const conf = e.classification_confidence ?? 0;
      return {
        ep: e,
        x: PAD.left + frac * plotW,
        y: scored ? PAD.top + (1 - conf) * plotH : LANE_Y,
        surface: scored ? e.surface : null,
        scored,
      };
    });

    // Fan out coincident marks.
    //
    // Episodes arrive in bulk uploads, so dozens share a timestamp to the millisecond — and the
    // discrete heuristic confidences (0.8, 0.75, …) collide on y as well. Plotted raw, forty
    // marks stack into one and the chart understates the fleet by an order of magnitude.
    const scoredPts = raw.filter((p) => p.scored);
    const unscoredPts = raw.filter((p) => !p.scored);

    // Scored marks keep their real reading and fan out only around it, so no mark moves more
    // than a few pixels from the confidence it actually has.
    const groups = new Map<string, Plotted[]>();
    for (const p of scoredPts) {
      const key = `${Math.round(p.x / 12)}:${Math.round(p.y / 12)}`;
      const g = groups.get(key);
      g ? g.push(p) : groups.set(key, [p]);
    }
    for (const g of groups.values()) {
      if (g.length < 2) continue;
      const perRow = Math.min(g.length, 7);
      const step = 13;
      g.forEach((p, k) => {
        const col = k % perRow;
        const row = Math.floor(k / perRow);
        p.x = Math.max(
          PAD.left + 6,
          Math.min(W - PAD.right - 6, p.x + (col - (perRow - 1) / 2) * step),
        );
        p.y = Math.max(PAD.top + 6, p.y - row * step);
      });
    }

    // The unscored lane is a rug, not a scatter: these episodes have no y reading at all, and
    // fanning them around a shared timestamp would either stack them into one blob or push them
    // up into the measured field and imply a confidence they do not have. Laying them out in
    // time order across a fenced band keeps every one visible and countable while claiming only
    // what is true — the order they arrived in, and that none of them was scored.
    const laneRows = 3;
    const laneStep = 11;
    const perLaneRow = Math.max(1, Math.ceil(unscoredPts.length / laneRows));
    const laneW = W - PAD.left - PAD.right - 12;
    unscoredPts
      .sort((a, b) => +new Date(a.ep.started_at) - +new Date(b.ep.started_at))
      .forEach((p, k) => {
        const col = k % perLaneRow;
        const row = Math.floor(k / perLaneRow);
        const spacing = Math.min(laneStep, laneW / Math.max(perLaneRow - 1, 1));
        p.x = PAD.left + 6 + col * spacing;
        p.y = LANE_BAND_TOP + 6 + row * laneStep;
      });

    const pts = [...scoredPts, ...unscoredPts];

    const c: Record<string, number> = { perception: 0, grounding: 0, motor: 0 };
    for (const p of pts) if (p.surface && p.surface in c) c[p.surface] += 1;

    return {
      points: pts,
      counts: c,
      ticks: [0, 0.25, 0.5, 0.75, 1],
    };
  }, [episodes]);

  const scored = points.filter((p) => p.scored);
  const unscored = points.filter((p) => !p.scored);

  return (
    <figure className="relative m-0">
      <div className="mb-3 flex items-end justify-between gap-4">
        <span className="muoto text-caption text-slate-smoke">[ inconclusive ]</span>
        <span className="muoto text-caption text-slate-smoke">[ attributed ]</span>
      </div>

      <div className="relative">
        <svg
          viewBox={`0 0 ${W} ${H}`}
          className="w-full"
          role="img"
          aria-label={`Fleet scatter: ${scored.length} attributed failures plotted by classification confidence over time, and ${unscored.length} episodes with no attributed surface.`}
        >
          {/* Dotted ambient grid — the graph-paper texture the canvas is meant to carry. */}
          <defs>
            <pattern id="fs-dots" width="16" height="16" patternUnits="userSpaceOnUse">
              <circle cx="1" cy="1" r="0.5" fill="#cad3d2" />
            </pattern>
          </defs>
          <rect
            x={PAD.left}
            y={PAD.top}
            width={W - PAD.left - PAD.right}
            height={H - PAD.top - PAD.bottom}
            fill="url(#fs-dots)"
          />

          {/* y gridlines + ticks */}
          {ticks.map((t) => {
            const y = PAD.top + (1 - t) * (H - PAD.top - PAD.bottom);
            return (
              <g key={t}>
                <line
                  x1={PAD.left}
                  y1={y}
                  x2={W - PAD.right}
                  y2={y}
                  stroke="#cad3d2"
                  strokeWidth="1"
                  strokeDasharray={t === 0 ? undefined : "2 4"}
                />
                <text
                  x={PAD.left - 10}
                  y={y + 3.5}
                  textAnchor="end"
                  className="fill-slate-smoke font-muoto"
                  fontSize="11"
                >
                  {Math.round(t * 100)}%
                </text>
              </g>
            );
          })}

          {/* The "not scored" lane, fenced off from the measured field. */}
          <line
            x1={PAD.left}
            y1={LANE_BAND_TOP}
            x2={W - PAD.right}
            y2={LANE_BAND_TOP}
            stroke="#cad3d2"
            strokeWidth="1"
            strokeDasharray="1 5"
          />
          <text
            x={PAD.left - 10}
            y={LANE_BAND_TOP + 22}
            textAnchor="end"
            className="fill-slate-smoke font-muoto"
            fontSize="11"
          >
            n/s
          </text>

          {/* Axis titles */}
          <text x={PAD.left} y={H - 8} className="fill-slate-smoke font-muoto" fontSize="11">
            earliest ingest
          </text>
          <text
            x={W / 2}
            y={H - 8}
            textAnchor="middle"
            className="fill-slate-smoke font-muoto"
            fontSize="11"
          >
            n/s band · not scored, in ingest order
          </text>
          <text
            x={W - PAD.right}
            y={H - 8}
            textAnchor="end"
            className="fill-slate-smoke font-muoto"
            fontSize="11"
          >
            latest ingest →
          </text>
          <text
            x={PAD.left - 44}
            y={PAD.top - 10}
            className="fill-slate-smoke font-muoto"
            fontSize="11"
          >
            confidence
          </text>

          {/* Unscored first, so a real verdict is never hidden behind a hollow mark. */}
          {unscored.map((p, i) => (
            <circle
              key={`u-${p.ep.id}-${i}`}
              cx={p.x}
              cy={p.y}
              r="4"
              fill="#ffffff"
              stroke="#09352e"
              strokeWidth="1"
              opacity="0.55"
              className="cursor-pointer"
              onMouseEnter={() => setHover(p)}
              onMouseLeave={() => setHover(null)}
              onClick={() => router.push(`/episodes/${p.ep.id}`)}
            />
          ))}

          {scored.map((p, i) => (
            <g
              key={`s-${p.ep.id}-${i}`}
              className="cursor-pointer"
              onMouseEnter={() => setHover(p)}
              onMouseLeave={() => setHover(null)}
              onClick={() => router.push(`/episodes/${p.ep.id}`)}
            >
              {/* 2px surface ring keeps overlapping marks separable. */}
              <circle cx={p.x} cy={p.y} r="7.5" fill="#e0e0e0" />
              <circle
                cx={p.x}
                cy={p.y}
                r="5.5"
                fill={p.surface ? SURFACE_HEX[p.surface] : "#ffffff"}
                stroke="#09352e"
                strokeWidth="1"
              />
              {hover?.ep.id === p.ep.id && (
                <circle
                  cx={p.x}
                  cy={p.y}
                  r="11"
                  fill="none"
                  stroke="#09352e"
                  strokeWidth="1"
                  strokeDasharray="2 3"
                />
              )}
            </g>
          ))}
        </svg>

        {/* Hover tooltip, positioned in the SVG's own coordinate space. */}
        {hover && (
          <div
            className="pointer-events-none absolute z-10 w-60 -translate-x-1/2 -translate-y-full rounded-card bg-bone-white p-3 shadow-hairline"
            style={{
              left: `${(hover.x / W) * 100}%`,
              top: `${(hover.y / H) * 100}%`,
              marginTop: -14,
            }}
          >
            <div className="muoto truncate text-caption text-slate-smoke">
              {hover.ep.id.slice(0, 8)}
            </div>
            <div className="mt-0.5 truncate text-body text-forest-ink">
              {hover.ep.instruction ?? "—"}
            </div>
            <div className="mt-2 flex items-center justify-between gap-2">
              <span className="muoto text-caption text-forest-ink">
                {hover.scored ? hover.surface : "not scored"}
              </span>
              <span className="muoto text-caption tabular-nums text-slate-smoke">
                {hover.scored
                  ? `${Math.round((hover.ep.classification_confidence ?? 0) * 100)}%`
                  : hover.ep.outcome}
              </span>
            </div>
          </div>
        )}

        {/* Floating chip over the plot. Sits in the empty mid-left of the field rather than
            the lower-left corner the system suggests — down there it covered the x-axis label
            and the top row of the n/s rug. */}
        <div className="absolute left-[6%] top-[42%] hidden sm:block">
          <span className="chip">
            {scored.length} of {points.length} explained
          </span>
        </div>
      </div>

      <figcaption className="mt-4 flex flex-wrap items-center justify-between gap-4">
        <SurfaceLegend counts={counts} />
        <span className="muoto flex items-center gap-1.5 text-caption text-slate-smoke">
          <svg width="10" height="10" aria-hidden>
            <circle cx="5" cy="5" r="3.5" fill="#ffffff" stroke="#09352e" strokeWidth="1" />
          </svg>
          no attributed surface
        </span>
      </figcaption>
    </figure>
  );
}
