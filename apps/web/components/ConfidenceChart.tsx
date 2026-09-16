"use client";

import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

type Point = { t: number; action_confidence: number | null; contact_force: number | null };

const INK = "#09352e";
const HAIRLINE = "#cad3d2";
const SMOKE = "#6c7a79";

/**
 * Action confidence and contact force over the episode.
 *
 * **Two stacked panels, not two y-axes on one plot.** Confidence lives in [0,1] while contact
 * force is unbounded newtons, and a dual-axis chart invents a relationship between them: where
 * the lines cross is an artefact of two arbitrary scales, not an event. It is also the exact
 * moment the comparison matters most — a motor-failure force spike would rescale a shared axis
 * and flatten the confidence line into the baseline.
 *
 * Small multiples sharing one frame axis fix both problems. `syncId` ties the two crosshairs
 * together, so hovering a frame reads both measures at once — which is the only thing the
 * single-plot version was actually buying.
 */
export function ConfidenceChart({ trace }: { trace: Point[] }) {
  const data = trace.map((p) => ({
    t: p.t,
    confidence: p.action_confidence,
    force: p.contact_force,
  }));

  const hasConfidence = data.some((d) => d.confidence != null);
  const hasForce = data.some((d) => d.force != null);

  if (!data.length) {
    return (
      <div className="flex h-[220px] items-center justify-center text-body text-slate-smoke">
        No per-frame signals on this episode.
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {hasConfidence && (
        <Panel
          title="action confidence"
          unit="0–1"
          axisLabel={!hasForce}
          domain={[0, 1]}
          ticks={[0, 0.5, 1]}
          dataKey="confidence"
          data={data}
          dashed={false}
          format={(n) => n.toFixed(2)}
        />
      )}
      {hasForce && (
        <Panel
          title="contact force"
          unit="newtons"
          axisLabel
          dataKey="force"
          data={data}
          dashed
          format={(n) => `${n.toFixed(2)} N`}
        />
      )}
    </div>
  );
}

function Panel({
  title,
  unit,
  axisLabel,
  domain,
  ticks,
  dataKey,
  data,
  dashed,
  format,
}: {
  title: string;
  unit: string;
  axisLabel: boolean;
  domain?: [number, number];
  ticks?: number[];
  dataKey: string;
  data: Record<string, unknown>[];
  dashed: boolean;
  format: (n: number) => string;
}) {
  return (
    <div className="space-y-1">
      {/* One series per panel, so the title is the legend. */}
      <div className="flex items-baseline gap-2">
        <svg width="16" height="8" aria-hidden>
          <line
            x1="0"
            y1="4"
            x2="16"
            y2="4"
            stroke={INK}
            strokeWidth="2"
            strokeDasharray={dashed ? "4 2" : undefined}
          />
        </svg>
        <span className="muoto text-caption text-forest-ink">{title}</span>
        <span className="muoto text-caption text-slate-smoke">{unit}</span>
      </div>

      <ResponsiveContainer width="100%" height={axisLabel ? 128 : 116}>
        <LineChart
          data={data}
          syncId="episode-trace"
          margin={{ top: 6, right: 8, left: -12, bottom: axisLabel ? 4 : 0 }}
        >
          <CartesianGrid stroke={HAIRLINE} strokeDasharray="2 4" vertical={false} />
          <XAxis
            dataKey="t"
            stroke={HAIRLINE}
            tick={{ fill: SMOKE, fontSize: 11 }}
            tickLine={false}
            axisLine={{ stroke: HAIRLINE }}
            hide={!axisLabel}
            label={
              axisLabel
                ? { value: "frame", position: "insideBottomRight", offset: -2, fill: SMOKE, fontSize: 11 }
                : undefined
            }
          />
          <YAxis
            domain={domain ?? ["auto", "auto"]}
            ticks={ticks}
            stroke={HAIRLINE}
            tick={{ fill: SMOKE, fontSize: 11 }}
            tickLine={false}
            axisLine={false}
            width={46}
          />
          <Tooltip
            cursor={{ stroke: INK, strokeWidth: 1, strokeDasharray: "2 3" }}
            contentStyle={{
              background: "#ffffff",
              border: "none",
              borderRadius: 12,
              boxShadow: "inset 0 0 0 0.5px #cad3d2",
              fontSize: 12,
              color: INK,
            }}
            labelStyle={{ color: SMOKE }}
            labelFormatter={(t) => `frame ${t}`}
            formatter={(value: number | string) => {
              const n = typeof value === "number" ? value : Number(value);
              return [format(n), title];
            }}
          />
          <Line
            type="monotone"
            dataKey={dataKey}
            stroke={INK}
            strokeWidth={2}
            strokeDasharray={dashed ? "4 2" : undefined}
            dot={false}
            isAnimationActive={false}
            connectNulls
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
