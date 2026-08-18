"use client";

import {
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  CartesianGrid,
} from "recharts";

type Point = { t: number; action_confidence: number | null; contact_force: number | null };

export function ConfidenceChart({ trace }: { trace: Point[] }) {
  const data = trace.map((p) => ({
    t: p.t,
    confidence: p.action_confidence,
    force: p.contact_force,
  }));
  return (
    <ResponsiveContainer width="100%" height={220}>
      <LineChart data={data} margin={{ top: 8, right: 12, left: -12, bottom: 0 }}>
        <CartesianGrid stroke="#dee2de" strokeDasharray="3 3" />
        <XAxis dataKey="t" stroke="#646464" fontSize={11} tickLine={false} />
        <YAxis stroke="#646464" fontSize={11} tickLine={false} width={40} />
        <Tooltip
          contentStyle={{ background: "#ffffff", border: "1px solid #dee2de", borderRadius: 8, fontSize: 12, color: "#2c2c2c" }}
          labelStyle={{ color: "#646464" }}
        />
        <Line
          type="monotone"
          dataKey="confidence"
          stroke="#41a1cf"
          strokeWidth={2}
          dot={false}
          isAnimationActive={false}
          connectNulls
        />
        <Line
          type="monotone"
          dataKey="force"
          stroke="#cf3f52"
          strokeWidth={1.5}
          dot={false}
          isAnimationActive={false}
          connectNulls
        />
      </LineChart>
    </ResponsiveContainer>
  );
}
