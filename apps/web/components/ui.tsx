"use client";

import { SURFACE_HEX, surfaceVerdict } from "@/lib/api";

/* ──────────────────────────────────────────────────────────────────────────
 * Data marks
 *
 * The failure surfaces are three steps of one green ramp, which is as far apart as a
 * mono-green system goes. So colour never carries identity on its own: every mark is
 * paired with its label, and the marks themselves differ in fill as well as hue
 * (hollow = no finding, filled = an attributed surface).
 * ────────────────────────────────────────────────────────────────────────── */

/** A single plotted circle, in the system's hollow/filled data-point language. */
export function Mark({
  surface,
  filled = true,
  size = 10,
  className = "",
}: {
  surface?: string | null;
  filled?: boolean;
  size?: number;
  className?: string;
}) {
  const hex = surface ? SURFACE_HEX[surface] : undefined;
  const r = size / 2 - 0.75;
  return (
    <svg width={size} height={size} className={className} aria-hidden>
      <circle
        cx={size / 2}
        cy={size / 2}
        r={r}
        fill={filled && hex ? hex : "#ffffff"}
        stroke="#09352e"
        strokeWidth="1"
      />
    </svg>
  );
}

/** Failure surface, shown honestly: successes and no-signal verdicts get their own label
 *  rather than the classifier's tie-broken bucket. */
export function SurfacePill({
  surface,
  confidence,
  outcome,
}: {
  surface: string | null;
  confidence?: number | null;
  outcome?: string;
}) {
  const v = surfaceVerdict(surface, confidence ?? null, outcome);
  const isSurface = v.kind === "surface";
  return (
    <span
      className="pill border border-lichen bg-bone-white text-forest-ink"
      title={v.hint}
    >
      <Mark surface={isSurface ? surface : null} filled={isSurface} size={8} />
      {v.label}
    </span>
  );
}

export function OutcomePill({ outcome }: { outcome: string }) {
  const ok = outcome === "success";
  return (
    <span
      className="pill border border-lichen bg-bone-white text-forest-ink"
      title={ok ? "The episode met its task goal." : "The episode did not meet its task goal."}
    >
      <Mark filled={!ok} surface={!ok ? "motor" : null} size={8} />
      {outcome}
    </span>
  );
}

/** Legend for the three failure surfaces. Always rendered wherever surfaces are plotted —
 *  with a sub-3:1 lightest step, the label is what makes the mark legible. */
export function SurfaceLegend({
  counts,
  className = "",
}: {
  counts?: Record<string, number>;
  className?: string;
}) {
  return (
    <ul className={`flex flex-wrap items-center gap-x-5 gap-y-2 ${className}`}>
      {(["perception", "grounding", "motor"] as const).map((s) => (
        <li key={s} className="flex items-center gap-1.5">
          <Mark surface={s} size={10} />
          <span className="muoto text-caption text-forest-ink">{s}</span>
          {counts && (
            <span className="muoto text-caption tabular-nums text-slate-smoke">
              {counts[s] ?? 0}
            </span>
          )}
        </li>
      ))}
    </ul>
  );
}

export function Confidence({ value }: { value: number | null }) {
  if (value == null) return <span className="muoto text-slate-smoke">—</span>;
  const pct = Math.round(value * 100);
  return (
    <div
      className="flex items-center gap-2"
      role="meter"
      aria-valuenow={pct}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-label="classification confidence"
    >
      <div className="h-1 w-16 overflow-hidden rounded-full bg-lichen">
        <div
          className="h-full rounded-full bg-motor"
          style={{ width: `${Math.max(pct, value > 0 ? 3 : 0)}%` }}
        />
      </div>
      <span className="muoto text-caption tabular-nums text-slate-smoke">{pct}%</span>
    </div>
  );
}

export function Spinner({ label }: { label?: string }) {
  return (
    <span
      className="inline-block h-3.5 w-3.5 animate-spin rounded-full border-2 border-lichen border-t-forest-ink"
      role="status"
      aria-label={label ?? "loading"}
    />
  );
}

/** Action button that keeps its label while working — a bare spinner loses the affordance
 *  and shifts the layout as the button collapses to icon width. */
export function ActionButton({
  onClick,
  busy,
  disabled,
  variant = "btn",
  children,
}: {
  onClick: () => void;
  busy?: boolean;
  disabled?: boolean;
  variant?: "btn" | "btn-filled" | "btn-ghost";
  children: React.ReactNode;
}) {
  return (
    <button
      onClick={onClick}
      disabled={disabled || busy}
      aria-busy={busy || undefined}
      className={variant}
    >
      {busy && <Spinner />}
      {children}
    </button>
  );
}

/** A failed request is a different problem from an empty result — say which, and say what
 *  to do about it. The most common cause locally is the API simply not running. */
export function ErrorNote({ error }: { error: unknown }) {
  if (!error) return null;
  const msg = String((error as Error)?.message ?? error);
  const unreachable = /failed to fetch|networkerror|load failed/i.test(msg);
  return (
    <div role="alert" className="card space-y-1.5 p-5">
      <div className="flex items-center gap-2">
        <span className="h-2 w-2 rounded-full bg-alarm" aria-hidden />
        <span className="text-body font-medium text-alarm">
          {unreachable ? "Can’t reach the Aperture API" : "Request failed"}
        </span>
      </div>
      <div className="muoto break-words text-caption text-slate-smoke">{msg}</div>
      {unreachable && (
        <div className="pt-1 text-caption text-slate-smoke">
          Start it with{" "}
          <span className="muoto text-forest-ink">
            cd apps/api &amp;&amp; uvicorn aperture.main:app --reload
          </span>
        </div>
      )}
    </div>
  );
}

export function Empty({ children }: { children: React.ReactNode }) {
  return (
    <div className="card p-10 text-center text-body text-slate-smoke">{children}</div>
  );
}

export function Skeleton({ className = "" }: { className?: string }) {
  return <div className={`skeleton ${className}`} aria-hidden />;
}

/** Table-shaped placeholder so the page doesn't jump when rows arrive. */
export function TableSkeleton({ rows = 5, cols = 6 }: { rows?: number; cols?: number }) {
  return (
    <div className="card space-y-3 p-4" role="status" aria-label="loading episodes">
      {Array.from({ length: rows }).map((_, r) => (
        <div key={r} className="flex gap-3">
          {Array.from({ length: cols }).map((_, c) => (
            <Skeleton key={c} className={`h-4 ${c === 1 ? "flex-[3]" : "flex-1"}`} />
          ))}
        </div>
      ))}
    </div>
  );
}

export function CardSkeleton({ count = 4 }: { count?: number }) {
  return (
    <div className="grid grid-cols-1 gap-4 md:grid-cols-2" role="status" aria-label="loading">
      {Array.from({ length: count }).map((_, i) => (
        <div key={i} className="card space-y-3 p-5">
          <Skeleton className="h-4 w-24" />
          <Skeleton className="h-4 w-40" />
          <Skeleton className="h-3 w-52" />
        </div>
      ))}
    </div>
  );
}

/* ──────────────────────────────────────────────────────────────────────────
 * Page-as-chart furniture
 * ────────────────────────────────────────────────────────────────────────── */

/** Section label — the small uppercase eyebrow used above every panel. */
export function Eyebrow({ children }: { children: React.ReactNode }) {
  return <div className="cinetype text-[11px] text-slate-smoke">{children}</div>;
}

/** Edge-of-canvas bracket label that frames a band as if it were a chart axis. */
export function AxisLabel({
  children,
  align = "left",
}: {
  children: React.ReactNode;
  align?: "left" | "right";
}) {
  return (
    <span
      className={`muoto text-caption text-slate-smoke ${align === "right" ? "text-right" : ""}`}
    >
      [ {children} ]
    </span>
  );
}

/** A section header carrying its two axis labels, so every band reads as a plot region. */
export function SectionHead({
  left,
  right,
  title,
  action,
}: {
  left: string;
  right?: string;
  title: string;
  action?: React.ReactNode;
}) {
  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between gap-4">
        <AxisLabel>{left}</AxisLabel>
        {right && <AxisLabel align="right">{right}</AxisLabel>}
      </div>
      <div className="h-px w-full bg-lichen" aria-hidden />
      <div className="flex flex-wrap items-end justify-between gap-3 pt-1">
        <h2 className="text-heading-sm font-medium text-forest-ink">{title}</h2>
        {action}
      </div>
    </div>
  );
}

export function PageHeader({
  title,
  subtitle,
  children,
}: {
  title: string;
  subtitle?: string;
  children?: React.ReactNode;
}) {
  return (
    <div className="flex flex-wrap items-end justify-between gap-4">
      <div className="space-y-1">
        <h1 className="text-heading font-medium text-forest-ink">{title}</h1>
        {subtitle && <p className="max-w-xl text-body text-slate-smoke">{subtitle}</p>}
      </div>
      {children && <div className="flex flex-wrap items-center gap-2">{children}</div>}
    </div>
  );
}

/** A single measured quantity. The number is the mark — no plot, so no hover layer. */
export function StatTile({
  label,
  value,
  note,
}: {
  label: string;
  value: React.ReactNode;
  note?: string;
}) {
  return (
    <div className="card p-5">
      <div className="cinetype text-[11px] text-slate-smoke">{label}</div>
      <div className="mt-2 text-heading font-medium tabular-nums text-forest-ink">{value}</div>
      {note && <div className="muoto mt-1 text-caption text-slate-smoke">{note}</div>}
    </div>
  );
}
