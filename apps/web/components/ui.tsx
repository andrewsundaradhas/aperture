"use client";

import { surfaceColor } from "@/lib/api";

export function SurfacePill({ surface }: { surface: string | null }) {
  return (
    <span className={`pill ${surfaceColor(surface)}`}>{surface ?? "unclassified"}</span>
  );
}

export function OutcomePill({ outcome }: { outcome: string }) {
  const ok = outcome === "success";
  return (
    <span className={`pill ${ok ? "bg-ok/10 text-ok" : "bg-motor/10 text-motor"}`}>
      {outcome}
    </span>
  );
}

export function Confidence({ value }: { value: number | null }) {
  if (value == null) return <span className="text-ash">—</span>;
  const pct = Math.round(value * 100);
  const tone = value >= 0.5 ? "bg-signal" : value >= 0.25 ? "bg-perception" : "bg-fog";
  return (
    <div className="flex items-center gap-2">
      <div className="w-16 h-1.5 rounded-full bg-mist overflow-hidden">
        <div className={`h-full ${tone}`} style={{ width: `${pct}%` }} />
      </div>
      <span className="text-caption font-mono text-ash tabular-nums">{pct}%</span>
    </div>
  );
}

export function Spinner() {
  return <span className="inline-block w-4 h-4 border-2 border-mist border-t-signal rounded-full animate-spin" />;
}

export function ErrorNote({ error }: { error: unknown }) {
  if (!error) return null;
  return (
    <div className="card border-motor/40 bg-motor/5 p-4 text-body-sm text-motor">
      {String((error as Error)?.message ?? error)}
    </div>
  );
}

export function Empty({ children }: { children: React.ReactNode }) {
  return <div className="card p-8 text-center text-ash text-body-sm">{children}</div>;
}
