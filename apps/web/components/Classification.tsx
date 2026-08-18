"use client";

import { AMBIGUOUS_MARGIN, Classification, Frame, SURFACES, SURFACE_BLURB } from "@/lib/api";

const BAR_TONE: Record<string, string> = {
  perception: "bg-perception",
  grounding: "bg-grounding",
  motor: "bg-motor",
};
const TEXT_TONE: Record<string, string> = {
  perception: "text-perception",
  grounding: "text-grounding",
  motor: "text-motor",
};

function pct(v: number): string {
  return `${Math.round(v * 100)}%`;
}

/** Big z-scores are routine here (a near-zero MAD makes them explode), so keep them readable. */
function num(v: number): string {
  if (!Number.isFinite(v)) return "∞";
  if (Math.abs(v) >= 1000) return v.toExponential(1);
  return String(Math.round(v * 100) / 100);
}

/** Turn a heuristic's raw evidence dict into the sentence a reviewer actually needs. */
function evidenceLine(surface: string, ev: Record<string, any>, present: boolean): string {
  if (!present) return "signal not present in this episode's logs";
  switch (surface) {
    case "perception": {
      const n = ev.collapsed_frames ?? 0;
      if (!n) return "confidence held against its rolling baseline";
      return `confidence collapsed on ${n} frame${n === 1 ? "" : "s"} (${pct(ev.fraction ?? 0)} of the trace)`;
    }
    case "grounding": {
      const blocks = ev.blocks ?? 0;
      const re = ev.reissues ?? 0;
      if (blocks < 2) return `only ${blocks} sub-goal block — too few to judge replanning`;
      if (!re) return `${blocks} sub-goal blocks, none re-issued`;
      return `${blocks} sub-goal blocks, ${re} re-issued`;
    }
    case "motor": {
      if (ev.flatline) return "force flatlined near zero — contact appears lost";
      const z = ev.max_z;
      return z == null ? "no force anomaly" : `peak force deviation ${num(z)}σ from median`;
    }
    default:
      return "";
  }
}

/** The evaluation verdict with its reasoning.
 *
 * The classifier scores all three heuristics and records the margin to the runner-up
 * specifically so this can be shown; the bucket name alone is the least useful part of it.
 */
export function HeuristicBreakdown({ classification }: { classification: Classification | null }) {
  if (!classification) {
    return (
      <p className="text-body-sm text-ash">
        Not classified yet. Run <span className="text-charcoal">Classify</span> to score the
        three failure heuristics against this episode.
      </p>
    );
  }

  const { surface, confidence, details, method } = classification;
  const margin = typeof details?._margin === "number" ? details._margin : null;
  const fired = confidence >= 0.005;
  const ambiguous = fired && margin != null && margin < AMBIGUOUS_MARGIN;

  return (
    <div className="space-y-4">
      <div className="space-y-1">
        {fired ? (
          <p className="text-body-sm text-charcoal">
            Classified as{" "}
            <span className={`font-medium ${TEXT_TONE[surface] ?? "text-charcoal"}`}>{surface}</span>{" "}
            at {pct(confidence)} confidence.{" "}
            <span className="text-ash">{SURFACE_BLURB[surface]}</span>
          </p>
        ) : (
          <p className="text-body-sm text-charcoal">
            <span className="font-medium">No heuristic fired.</span>{" "}
            <span className="text-ash">
              Every signal scored zero, so the surface below is the classifier&rsquo;s tie-break
              rather than a finding.
            </span>
          </p>
        )}
        {ambiguous && (
          <p className="text-caption text-perception">
            Contested call — the runner-up is within {pct(margin!)}. Treat the surface as a
            hypothesis, not a conclusion.
          </p>
        )}
      </div>

      <ul className="space-y-2.5">
        {SURFACES.map((s) => {
          const d = details?.[s] ?? {};
          const c: number = d.confidence ?? 0;
          const present: boolean = d.signal_present ?? false;
          const won = fired && s === surface;
          return (
            <li key={s} className="space-y-1">
              <div className="flex items-center gap-2">
                <span
                  className={`w-1.5 h-1.5 rounded-full ${present ? BAR_TONE[s] : "bg-fog"}`}
                  aria-hidden
                />
                <span
                  className={`text-body-sm ${won ? `font-medium ${TEXT_TONE[s]}` : "text-charcoal"}`}
                >
                  {s}
                </span>
                {won && <span className="pill bg-linen text-ash border border-mist">winner</span>}
                <span className="ml-auto font-mono text-caption tabular-nums text-ash">
                  {pct(c)}
                </span>
              </div>
              <div className="h-1.5 w-full overflow-hidden rounded-full bg-mist/60">
                <div
                  className={`h-full ${present ? BAR_TONE[s] : "bg-fog"} ${won ? "" : "opacity-40"}`}
                  style={{ width: `${Math.max(c * 100, c > 0 ? 2 : 0)}%` }}
                />
              </div>
              <div className={`text-caption ${present ? "text-ash" : "text-fog italic"}`}>
                {evidenceLine(s, d.evidence ?? {}, present)}
              </div>
            </li>
          );
        })}
      </ul>

      <div className="text-caption text-fog">
        method: <span className="font-mono">{method}</span>
        {margin != null && (
          <>
            {" · "}margin over runner-up:{" "}
            <span className="font-mono tabular-nums">{pct(margin)}</span>
          </>
        )}
      </div>
    </div>
  );
}

/** The sub-goal sequence, collapsed into contiguous blocks.
 *
 * This is the grounding heuristic made visible: a block repeating a sub-goal seen earlier is
 * a re-issue, and a run of them is the policy failing to keep the instruction grounded.
 */
export function SubgoalTrack({ frames }: { frames: Frame[] }) {
  const seq = frames.map((f) => f.subgoal).filter((s): s is string => !!s);
  if (!seq.length) {
    return <p className="text-caption text-fog italic">No sub-goal tokens in this episode.</p>;
  }

  const blocks: { subgoal: string; count: number; reissue: boolean }[] = [];
  const seen = new Set<string>();
  for (const s of seq) {
    const last = blocks[blocks.length - 1];
    if (last && last.subgoal === s) {
      last.count += 1;
      continue;
    }
    blocks.push({ subgoal: s, count: 1, reissue: seen.has(s) });
    seen.add(s);
  }
  const reissues = blocks.filter((b) => b.reissue).length;

  return (
    <div className="space-y-2">
      <div className="flex flex-wrap items-center gap-1.5">
        {blocks.map((b, i) => (
          <span key={i} className="flex items-center gap-1.5">
            {i > 0 && (
              <span className="text-fog" aria-hidden>
                &rsaquo;
              </span>
            )}
            <span
              className={`pill ${
                b.reissue
                  ? "bg-grounding/10 text-grounding"
                  : "bg-linen text-charcoal border border-mist"
              }`}
              title={
                b.reissue
                  ? `"${b.subgoal}" re-issued after the policy had already moved on`
                  : `${b.count} frame${b.count === 1 ? "" : "s"}`
              }
            >
              {b.subgoal}
              <span className="ml-1 font-mono text-fog">&times;{b.count}</span>
            </span>
          </span>
        ))}
      </div>
      <p className="text-caption text-ash">
        {blocks.length} block{blocks.length === 1 ? "" : "s"}
        {reissues > 0 ? (
          <>
            {" · "}
            <span className="text-grounding">{reissues} re-issued</span> — the replanning signal
            behind a grounding verdict
          </>
        ) : (
          " · no re-issues"
        )}
      </p>
    </div>
  );
}
