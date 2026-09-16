"use client";

import { AMBIGUOUS_MARGIN, Classification, Frame, SURFACES, SURFACE_BLURB } from "@/lib/api";
import { Mark } from "@/components/ui";

const BAR_TONE: Record<string, string> = {
  perception: "bg-perception",
  grounding: "bg-grounding",
  motor: "bg-motor",
};
// Text stays in ink tokens and never wears the series colour — the mark beside it carries
// identity. Two of the three surface greens would also fail text contrast on this canvas.
const TEXT_TONE: Record<string, string> = {
  perception: "text-forest-ink",
  grounding: "text-forest-ink",
  motor: "text-forest-ink",
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
      const peak = ev.peak_confidence;
      if (!n) {
        return peak != null
          ? `confidence held near its episode peak of ${Number(peak).toFixed(2)}`
          : "confidence held near its episode peak";
      }
      return (
        `confidence collapsed on ${n} frame${n === 1 ? "" : "s"} ` +
        `(${pct(ev.fraction ?? 0)} of the trace)` +
        (peak != null ? ` from a peak of ${Number(peak).toFixed(2)}` : "")
      );
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
      <p className="text-body text-slate-smoke">
        Not classified yet. Run <span className="text-forest-ink">Classify</span> to score the
        three failure heuristics against this episode.
      </p>
    );
  }

  const { surface, confidence, details, method } = classification;

  // The learned path produces a probability distribution, not three heuristic scores. Rendering
  // the heuristic bars for it showed three empty rows reading "signal not present" next to a
  // blurb about contact force — a rationale the model never gave. For an interpretability tool
  // that is worse than showing nothing, so the two verdicts get the explanation they actually
  // have.
  if (method === "learned") {
    return <LearnedBreakdown surface={surface} confidence={confidence} details={details} />;
  }

  const margin = typeof details?._margin === "number" ? details._margin : null;
  const fired = confidence >= 0.005;
  const ambiguous = fired && margin != null && margin < AMBIGUOUS_MARGIN;
  const inferred = details?._inferred_by_elimination;

  return (
    <div className="space-y-4">
      <div className="space-y-1">
        {fired ? (
          <p className="text-body text-forest-ink">
            Classified as{" "}
            <span className="font-medium text-forest-ink">{surface}</span>{" "}
            at {pct(confidence)} confidence.{" "}
            <span className="text-slate-smoke">{SURFACE_BLURB[surface]}</span>
          </p>
        ) : (
          <p className="text-body text-forest-ink">
            <span className="font-medium">No heuristic fired.</span>{" "}
            <span className="text-slate-smoke">
              Every signal scored zero, so the surface below is the classifier&rsquo;s tie-break
              rather than a finding.
            </span>
          </p>
        )}
        {ambiguous && (
          <p className="text-caption text-caution">
            Contested call — the runner-up is within {pct(margin!)}. Treat the surface as a
            hypothesis, not a conclusion.
          </p>
        )}
        {inferred && (
          <p className="text-caption text-caution">
            Inferred by elimination — {inferred.reason} That is an argument from a missing
            signal, not positive evidence.
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
<Mark surface={s} filled={present} size={9} />
                <span
                  className={`text-body ${won ? "font-medium text-forest-ink" : "text-forest-ink"}`}
                >
                  {s}
                </span>
                {won && <span className="tag">winner</span>}
                <span className="muoto ml-auto text-caption tabular-nums text-slate-smoke">
                  {pct(c)}
                </span>
              </div>
              <div className="h-1.5 w-full overflow-hidden rounded-full bg-lichen">
                <div
                  className={`h-full rounded-full ${present ? BAR_TONE[s] : "bg-slate-smoke"} ${won ? "" : "opacity-40"}`}
                  style={{ width: `${Math.max(c * 100, c > 0 ? 2 : 0)}%` }}
                />
              </div>
              <div className={`text-caption ${present ? "text-slate-smoke" : "italic text-slate-smoke/70"}`}>
                {evidenceLine(s, d.evidence ?? {}, present)}
              </div>
            </li>
          );
        })}
      </ul>

      <div className="muoto text-caption text-slate-smoke">
        method: <span className="text-forest-ink">{method}</span>
        {margin != null && (
          <>
            {" · "}margin over runner-up:{" "}
            <span className="tabular-nums text-forest-ink">{pct(margin)}</span>
          </>
        )}
      </div>
    </div>
  );
}

/** What the trained `FailureHead` actually predicted.
 *
 * The model outputs one probability per surface. There are no per-heuristic signals behind it,
 * so this shows the distribution and says plainly where it came from — including that a single
 * frame is all the model saw.
 */
function LearnedBreakdown({
  surface,
  confidence,
  details,
}: {
  surface: string;
  confidence: number;
  details: Record<string, any>;
}) {
  const probs: Record<string, number> = details?.probs ?? {};
  const ranked = SURFACES.map((s) => ({ s, p: probs[s] ?? 0 })).sort((a, b) => b.p - a.p);
  const margin = ranked.length > 1 ? ranked[0].p - ranked[1].p : null;

  return (
    <div className="space-y-4">
      <p className="text-body text-forest-ink">
        Predicted{" "}
        <span className="font-medium text-forest-ink">{surface}</span>{" "}
        at {pct(confidence)} confidence.{" "}
        <span className="text-slate-smoke">
          From the trained failure head, over this episode&rsquo;s first frame image.
        </span>
      </p>
      {margin != null && margin < AMBIGUOUS_MARGIN && (
        <p className="text-caption text-caution">
          Close call — the runner-up is within {pct(margin)}.
        </p>
      )}

      <ul className="space-y-2.5">
        {ranked.map(({ s, p }) => (
          <li key={s} className="space-y-1">
            <div className="flex items-center gap-2">
              <Mark surface={s} size={9} />
              <span
                className={`text-body ${s === surface ? "font-medium" : ""} text-forest-ink`}
              >
                {s}
              </span>
              {s === surface && (
                <span className="tag">predicted</span>
              )}
              <span className="muoto ml-auto text-caption tabular-nums text-slate-smoke">{pct(p)}</span>
            </div>
            <div className="h-1.5 w-full overflow-hidden rounded-full bg-lichen">
              <div
                className={`h-full rounded-full ${BAR_TONE[s]} ${s === surface ? "" : "opacity-40"}`}
                style={{ width: `${Math.max(p * 100, p > 0 ? 2 : 0)}%` }}
              />
            </div>
          </li>
        ))}
      </ul>

      <p className="muoto text-caption text-slate-smoke">
        method: <span className="text-forest-ink">learned</span> · one frame, no heuristic signals —
        read the attention map below for where it looked.
      </p>
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
    return <p className="text-caption italic text-slate-smoke">No sub-goal tokens in this episode.</p>;
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
              <span className="text-slate-smoke" aria-hidden>
                &rsaquo;
              </span>
            )}
            <span
              className="pill border border-lichen bg-bone-white text-forest-ink"
              title={
                b.reissue
                  ? `"${b.subgoal}" re-issued after the policy had already moved on`
                  : `${b.count} frame${b.count === 1 ? "" : "s"}`
              }
            >
              {b.reissue && <Mark surface="grounding" size={8} />}
              {b.subgoal}
              <span className="ml-1 text-slate-smoke">&times;{b.count}</span>
            </span>
          </span>
        ))}
      </div>
      <p className="text-caption text-slate-smoke">
        {blocks.length} block{blocks.length === 1 ? "" : "s"}
        {reissues > 0 ? (
          <>
            {" · "}
            <span className="text-forest-ink">{reissues} re-issued</span> — the replanning signal
            behind a grounding verdict
          </>
        ) : (
          " · no re-issues"
        )}
      </p>
    </div>
  );
}
