"use client";

import { surfaceVerdict } from "@/lib/api";

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
  return (
    <span className={`pill ${v.tone}`} title={v.hint}>
      {v.label}
    </span>
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
    <div
      className="flex items-center gap-2"
      role="meter"
      aria-valuenow={pct}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-label="classification confidence"
    >
      <div className="w-16 h-1.5 rounded-full bg-mist overflow-hidden">
        <div className={`h-full ${tone}`} style={{ width: `${pct}%` }} />
      </div>
      <span className="text-caption font-mono text-ash tabular-nums">{pct}%</span>
    </div>
  );
}

export function Spinner({ label }: { label?: string }) {
  return (
    <span
      className="inline-block w-4 h-4 border-2 border-mist border-t-signal rounded-full animate-spin"
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
  variant?: "btn" | "btn-accent" | "btn-filled";
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
    <div role="alert" className="card border-motor/40 bg-motor/5 p-4 space-y-1">
      <div className="text-body-sm font-medium text-motor">
        {unreachable ? "Can’t reach the Aperture API" : "Request failed"}
      </div>
      <div className="text-caption font-mono text-motor/80 break-words">{msg}</div>
      {unreachable && (
        <div className="text-caption text-ash pt-1">
          Start it with{" "}
          <span className="font-mono text-charcoal">
            cd apps/api &amp;&amp; uvicorn aperture.main:app --reload
          </span>
        </div>
      )}
    </div>
  );
}

export function Empty({ children }: { children: React.ReactNode }) {
  return <div className="card p-8 text-center text-ash text-body-sm">{children}</div>;
}

export function Skeleton({ className = "" }: { className?: string }) {
  return <div className={`skeleton ${className}`} aria-hidden />;
}

/** Table-shaped placeholder so the page doesn't jump when rows arrive. */
export function TableSkeleton({ rows = 5, cols = 6 }: { rows?: number; cols?: number }) {
  return (
    <div className="card p-3 space-y-3" role="status" aria-label="loading episodes">
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
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4" role="status" aria-label="loading">
      {Array.from({ length: count }).map((_, i) => (
        <div key={i} className="card p-6 space-y-3">
          <Skeleton className="h-5 w-24" />
          <Skeleton className="h-4 w-40" />
          <Skeleton className="h-3 w-52" />
        </div>
      ))}
    </div>
  );
}

/** Section label — the small uppercase eyebrow used above every panel. */
export function Eyebrow({ children }: { children: React.ReactNode }) {
  return (
    <div className="text-caption uppercase tracking-wide text-ash">{children}</div>
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
      <div>
        <h1 className="serif text-heading text-graphite">{title}</h1>
        {subtitle && <p className="text-body-sm text-ash mt-1">{subtitle}</p>}
      </div>
      {children && <div className="flex flex-wrap items-center gap-2">{children}</div>}
    </div>
  );
}
