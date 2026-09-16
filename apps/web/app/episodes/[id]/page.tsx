"use client";

import { useCallback, useEffect, useState } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import { api, Attribution, Episode } from "@/lib/api";
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
    } catch (e) {
      setError(e);
    }
  }, [id]);

  useEffect(() => {
    load();
  }, [load]);

  async function classify() {
    setBusy("classify");
    try {
      await api.classify(id);
      await load();
    } catch (e) {
      setError(e);
    } finally {
      setBusy(null);
    }
  }

  async function attribute() {
    setBusy("attribute");
    try {
      setAttr(await api.runAttribution(id));
      // Attribution can be the first thing to reveal a classification-dependent result,
      // so refresh the episode alongside it.
      await load();
    } catch (e) {
      setError(e);
    } finally {
      setBusy(null);
    }
  }

  // An error with an episode already loaded is a failed action, not a dead page — keep the
  // page up and show the error above it.
  if (error && !episode) return <ErrorNote error={error} />;
  if (!episode) return <DetailSkeleton />;

  const cf = attr?.counterfactual_result;
  const cls = episode.classification;

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center gap-3">
        <Link href="/episodes" className="link">
          &larr; Episodes
        </Link>
        <span className="muoto text-caption text-slate-smoke">{episode.id.slice(0, 12)}</span>
        <OutcomePill outcome={episode.outcome} />
        <SurfacePill
          surface={cls?.surface ?? null}
          confidence={cls?.confidence ?? null}
          outcome={episode.outcome}
        />
        <div className="ml-auto flex gap-2">
          <ActionButton onClick={classify} busy={busy === "classify"} disabled={!!busy}>
            Classify
          </ActionButton>
          <ActionButton
            onClick={attribute}
            busy={busy === "attribute"}
            disabled={!!busy}
            variant="btn-filled"
          >
            Run attribution
          </ActionButton>
        </div>
      </div>

      <ErrorNote error={error} />

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
        <div className="card space-y-3 p-4">
          <Eyebrow>Instruction</Eyebrow>
          <div className="text-body-lg text-forest-ink">{episode.instruction ?? "—"}</div>
          <div className="grid grid-cols-2 gap-3 pt-1">
            <Meta label="Format" value={episode.source_format.toUpperCase()} />
            <Meta label="Frames" value={String(episode.frames.length)} />
            <Meta label="Robot" value={episode.robot_id.slice(0, 8)} />
            <Meta label="Started" value={new Date(episode.started_at).toLocaleString()} />
          </div>
        </div>

        <div className="card p-4 lg:col-span-2">
          <Eyebrow>Action-confidence &amp; contact-force trace</Eyebrow>
          <div className="mt-2">
            <ConfidenceChart trace={attr?.confidence_trace ?? episode.frames} />
          </div>
        </div>
      </div>

      {/* Why this episode was bucketed the way it was — the three heuristics, scored. */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <div className="card space-y-4 p-4">
          <Eyebrow>Why this classification</Eyebrow>
          <HeuristicBreakdown classification={cls} />
        </div>

        <div className="card space-y-4 p-4">
          <div className="space-y-3">
            <Eyebrow>Sub-goal sequence</Eyebrow>
            <SubgoalTrack frames={episode.frames} />
          </div>
          <div className="space-y-3 border-t border-lichen pt-4">
            <Eyebrow>Instruction sensitivity (counterfactual)</Eyebrow>
            <Counterfactual cf={cf} />
          </div>
        </div>
      </div>

      <div className="card p-4">
        <Eyebrow>Attention rollout</Eyebrow>
        <div className="mt-3">
          <Heatmap uri={attr?.attention_map_uri ?? null} />
        </div>
      </div>
    </div>
  );
}

function Counterfactual({ cf }: { cf: any }) {
  if (!cf)
    return (
      <p className="text-body text-slate-smoke">
        Runs automatically for grounding failures when you attribute. Motor and perception
        failures skip it.
      </p>
    );
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
          {Math.round(cf.sensitivity * 100)}% of paraphrases changed the target
        </span>
      </div>
      <div className="muoto text-caption text-slate-smoke">
        base &ldquo;{cf.base_instruction}&rdquo; &rarr; {cf.base_target}
      </div>
      <ul className="text-body">
        {cf.trials?.map((t: any, i: number) => (
          <li key={i} className="flex justify-between gap-2 border-b border-lichen py-1.5 last:border-b-0">
            <span className="truncate text-forest-ink">&ldquo;{t.instruction}&rdquo;</span>
            <span
              className={`muoto text-caption tabular-nums ${
                t.changed ? "text-deep-fern" : "text-slate-smoke"
              }`}
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
      <div className="muoto text-caption text-slate-smoke">{label}</div>
      <div className="mt-0.5 text-body text-forest-ink">{value}</div>
    </div>
  );
}

function DetailSkeleton() {
  return (
    <div className="space-y-6" role="status" aria-label="loading episode">
      <div className="flex items-center gap-3">
        <Skeleton className="h-4 w-24" />
        <Skeleton className="h-4 w-28" />
        <Skeleton className="h-5 w-16 rounded-full" />
      </div>
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
        <div className="card space-y-3 p-4">
          <Skeleton className="h-3 w-20" />
          <Skeleton className="h-4 w-44" />
          <Skeleton className="h-10 w-full" />
        </div>
        <div className="card p-4 lg:col-span-2">
          <Skeleton className="h-3 w-52" />
          <Skeleton className="mt-3 h-[220px] w-full" />
        </div>
      </div>
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <Skeleton className="h-56 w-full rounded-card" />
        <Skeleton className="h-56 w-full rounded-card" />
      </div>
    </div>
  );
}
