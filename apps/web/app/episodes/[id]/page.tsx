"use client";

import { useCallback, useEffect, useState } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import { api, Attribution, Episode, Robot, robotName } from "@/lib/api";
import { ConfidenceChart } from "@/components/ConfidenceChart";
import { Heatmap } from "@/components/Heatmap";
import { HeuristicBreakdown, SubgoalTrack } from "@/components/Classification";
import {
  ActionButton,
  Eyebrow,
  ErrorNote,
  Mark,
  OutcomePill,
  Skeleton,
  SurfacePill,
} from "@/components/ui";

export default function EpisodeDetail() {
  const { id } = useParams<{ id: string }>();
  const [episode, setEpisode] = useState<Episode | null>(null);
  const [attr, setAttr] = useState<Attribution | null>(null);
  const [robot, setRobot] = useState<Robot | undefined>(undefined);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<unknown>(null);

  const load = useCallback(async () => {
    try {
      const [ep, at] = await Promise.all([
        api.getEpisode(id),
        api.getAttribution(id).catch(() => null),
      ]);
      setEpisode(ep);
      setAttr(at);
      setError(null);
      api
        .listRobots()
        .then((rs) => setRobot(rs.find((r) => r.id === ep.robot_id)))
        .catch(() => {});
    } catch (e) {
      setError(e);
    }
  }, [id]);

  useEffect(() => {
    load();
  }, [load]);

  async function run(kind: "classify" | "attribute") {
    setBusy(kind);
    try {
      if (kind === "classify") await api.classify(id);
      else setAttr(await api.runAttribution(id));
      await load();
    } catch (e) {
      setError(e);
    } finally {
      setBusy(null);
    }
  }

  // An error with an episode already loaded is a failed action, not a dead page.
  if (error && !episode) return <ErrorNote error={error} />;
  if (!episode) return <DetailSkeleton />;

  const cls = episode.classification;
  const failed = episode.outcome !== "success";
  const cf = attr?.counterfactual_result;
  const hasSignals = episode.frames.some((f) => f.action_confidence != null || f.contact_force != null);
  const hasSubgoals = episode.frames.some((f) => f.subgoal);

  return (
    <div className="space-y-6">
      {/* Its own layer: the condensed headline's tall glyph box sits over it and swallowed clicks. */}
      <div className="relative z-10">
        <Link href="/episodes" className="link inline-block py-1 text-body">
          &larr; Episodes
        </Link>
      </div>

      <div className="flex flex-wrap items-start gap-4">
        <div className="min-w-0 flex-1 space-y-2">
          <h1 className="text-[48px] leading-[0.92] text-lavender">
            {episode.instruction ?? "Untitled episode"}
          </h1>
          <div className="flex flex-wrap items-center gap-2">
            <OutcomePill outcome={episode.outcome} />
            {failed && (
              <SurfacePill
                surface={cls?.surface ?? null}
                confidence={cls?.confidence ?? null}
                outcome={episode.outcome}
              />
            )}
          </div>
        </div>
        {failed && (
          <div className="flex gap-2">
            <ActionButton onClick={() => run("classify")} busy={busy === "classify"} disabled={!!busy}>
              Re-analyse
            </ActionButton>
            <ActionButton
              onClick={() => run("attribute")}
              busy={busy === "attribute"}
              disabled={!!busy}
            >
              Generate attention map
            </ActionButton>
          </div>
        )}
      </div>

      <ErrorNote error={error} />

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
        <div className="card space-y-3 p-5">
          <Eyebrow>Episode</Eyebrow>
          <dl className="grid grid-cols-2 gap-x-3 gap-y-3">
            <Meta label="Robot" value={robotName(robot)} />
            <Meta label="Frames" value={String(episode.frames.length)} />
            <Meta label="Recorded" value={new Date(episode.started_at).toLocaleString()} />
            <Meta label="Format" value={episode.source_format === "rlds" ? "RLDS" : "LeRobot"} />
          </dl>
        </div>

        <div className="card space-y-4 p-5 lg:col-span-2">
          <Eyebrow>Why it failed</Eyebrow>
          {failed ? (
            <HeuristicBreakdown classification={cls} />
          ) : (
            <p className="text-body text-slate-smoke">
              This episode succeeded — there is no failure to explain.
            </p>
          )}
        </div>
      </div>

      {hasSignals && (
        <div className="card p-5">
          <Eyebrow>Policy confidence and contact force</Eyebrow>
          <div className="mt-2">
            <ConfidenceChart trace={attr?.confidence_trace ?? episode.frames} />
          </div>
        </div>
      )}

      {(hasSubgoals || cf) && (
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
          {hasSubgoals && (
            <div className="card space-y-3 p-5">
              <Eyebrow>Sub-goal sequence</Eyebrow>
              <SubgoalTrack frames={episode.frames} />
            </div>
          )}
          {cf && (
            <div className="card space-y-3 p-5">
              <Eyebrow>Instruction sensitivity</Eyebrow>
              <Counterfactual cf={cf} />
            </div>
          )}
        </div>
      )}

      {attr?.attention_map_uri && (
        <div className="card p-5">
          <Eyebrow>Where the policy looked</Eyebrow>
          <div className="mt-3">
            <Heatmap uri={attr.attention_map_uri} />
          </div>
        </div>
      )}
    </div>
  );
}

function Counterfactual({ cf }: { cf: any }) {
  if (cf.applicable === false)
    return <p className="text-body text-slate-smoke">Not applicable: {cf.reason}</p>;

  return (
    <div className="space-y-2">
      <div className="flex flex-wrap items-center gap-2">
        <span className="pill border border-lichen bg-bone-white text-forest-ink">
          <Mark
            surface={cf.verdict === "grounding-sensitive" ? "grounding" : null}
            filled={cf.verdict === "grounding-sensitive"}
            size={8}
          />
          {cf.verdict}
        </span>
        <span className="text-body text-slate-smoke">
          {Math.round(cf.sensitivity * 100)}% of rephrasings changed the target
        </span>
      </div>
      <div className="muoto text-caption text-slate-smoke">
        original &ldquo;{cf.base_instruction}&rdquo; &rarr; {cf.base_target}
      </div>
      <ul className="text-body">
        {cf.trials?.map((t: any, i: number) => (
          <li key={i} className="flex justify-between gap-2 border-b border-lichen py-1.5 last:border-b-0">
            <span className="truncate text-forest-ink">&ldquo;{t.instruction}&rdquo;</span>
            <span
              className={`muoto text-caption tabular-nums ${t.changed ? "text-deep-fern" : "text-slate-smoke"}`}
              title={t.changed ? "resolved to a different object" : "unchanged"}
            >
              {t.target}
              {t.changed ? " ↯" : ""}
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}

function Meta({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt className="muoto text-caption text-slate-smoke">{label}</dt>
      <dd className="mt-0.5 text-body text-forest-ink">{value}</dd>
    </div>
  );
}

function DetailSkeleton() {
  return (
    <div className="space-y-6" role="status" aria-label="loading episode">
      <Skeleton className="h-4 w-24" />
      <Skeleton className="h-7 w-80" />
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
        <Skeleton className="h-40 w-full rounded-card" />
        <Skeleton className="h-40 w-full rounded-card lg:col-span-2" />
      </div>
    </div>
  );
}
