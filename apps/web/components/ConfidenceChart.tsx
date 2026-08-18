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

const CONFIDENCE = "#41a1cf";
const FORCE = "#cf3f52";

/** Action-confidence and contact-force over time.
 *
 * The two series must never share an axis: confidence lives in [0, 1] while contact force is
 * unbounded newtons, so a single force spike (the motor-failure signature) rescales the axis
 * and flattens the confidence line into the baseline — precisely when comparing the two
 * matters most. Confidence is pinned left on a fixed [0, 1] domain; force gets its own right
 * axis that is free to autoscale.
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
      <div className="h-[220px] flex items-center justify-center text-body-sm text-ash">
        No per-frame signals on this episode.
      </div>
    );
  }

  return (
    <div className="space-y-2">
      <ResponsiveContainer width="100%" height={220}>
        <LineChart data={data} margin={{ top: 8, right: 8, left: -8, bottom: 4 }}>
          <CartesianGrid stroke="#dee2de" strokeDasharray="3 3" vertical={false} />
          <XAxis
            dataKey="t"
            stroke="#646464"
            fontSize={11}
            tickLine={false}
            axisLine={{ stroke: "#dee2de" }}
            label={{
              value: "frame",
              position: "insideBottomRight",
              offset: -2,
              fill: "#b4b8b4",
              fontSize: 10,
            }}
          />
          <YAxis
            yAxisId="confidence"
            domain={[0, 1]}
            ticks={[0, 0.25, 0.5, 0.75, 1]}
            stroke={CONFIDENCE}
            fontSize={11}
            tickLine={false}
            axisLine={false}
            width={44}
          />
          {hasForce && (
            <YAxis
              yAxisId="force"
              orientation="right"
              stroke={FORCE}
              fontSize={11}
              tickLine={false}
              axisLine={false}
              width={44}
            />
          )}
          <Tooltip
            contentStyle={{
              background: "#ffffff",
              border: "1px solid #dee2de",
              borderRadius: 8,
              fontSize: 12,
              color: "#2c2c2c",
            }}
            labelStyle={{ color: "#646464" }}
            labelFormatter={(t) => `frame ${t}`}
            formatter={(value: number | string, name: string) => {
              const n = typeof value === "number" ? value : Number(value);
              return name === "confidence"
                ? [n.toFixed(2), "action confidence"]
                : [`${n.toFixed(2)} N`, "contact force"];
            }}
          />
          {hasConfidence && (
            <Line
              yAxisId="confidence"
              type="monotone"
              dataKey="confidence"
              stroke={CONFIDENCE}
              strokeWidth={2}
              dot={false}
              isAnimationActive={false}
              connectNulls
            />
          )}
          {hasForce && (
            <Line
              yAxisId="force"
              type="monotone"
              dataKey="force"
              stroke={FORCE}
              strokeWidth={1.5}
              strokeDasharray="4 2"
              dot={false}
              isAnimationActive={false}
              connectNulls
            />
          )}
        </LineChart>
      </ResponsiveContainer>

      <div className="flex flex-wrap gap-4 text-caption text-ash">
        <LegendKey color={CONFIDENCE} label="action confidence" axis="left axis · 0–1" />
        {hasForce && <LegendKey color={FORCE} label="contact force" axis="right axis · newtons" dashed />}
      </div>
    </div>
  );
}

function LegendKey({
  color,
  label,
  axis,
  dashed,
}: {
  color: string;
  label: string;
  axis: string;
  dashed?: boolean;
}) {
  return (
    <span className="flex items-center gap-1.5">
      <svg width="16" height="8" aria-hidden>
        <line
          x1="0"
          y1="4"
          x2="16"
          y2="4"
          stroke={color}
          strokeWidth="2"
          strokeDasharray={dashed ? "4 2" : undefined}
        />
      </svg>
      <span className="text-charcoal">{label}</span>
      <span className="text-fog">{axis}</span>
    </span>
  );
}
