"use client";

import { useCallback, useEffect, useState } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import { api, Attribution, Episode } from "@/lib/api";
import { ConfidenceChart } from "@/components/ConfidenceChart";
import { Heatmap } from "@/components/Heatmap";
import { ErrorNote, OutcomePill, Spinner, SurfacePill } from "@/components/ui";

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
    } catch (e) {
      setError(e);
    } finally {
      setBusy(null);
    }
  }

  if (error) return <ErrorNote error={error} />;
  if (!episode) return <div className="card p-8 flex justify-center"><Spinner /></div>;

  const cf = attr?.counterfactual_result;

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3">
        <Link href="/episodes" className="text-ash hover:text-graphite transition">← Episodes</Link>
        <span className="font-mono text-body-sm text-ash">{episode.id.slice(0, 12)}</span>
        <OutcomePill outcome={episode.outcome} />
        <div className="ml-auto flex gap-2">
          <button onClick={classify} disabled={!!busy} className="btn">
            {busy === "classify" ? <Spinner /> : "Classify"}
          </button>
          <button onClick={attribute} disabled={!!busy} className="btn-accent">
            {busy === "attribute" ? <Spinner /> : "Run attribution"}
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <div className="card p-4 space-y-2">
          <div className="text-caption uppercase tracking-wide text-ash">Instruction</div>
          <div className="text-graphite">{episode.instruction ?? "—"}</div>
          <div className="grid grid-cols-2 gap-2 pt-2 text-sm">
            <Meta label="Format" value={episode.source_format.toUpperCase()} />
            <Meta label="Frames" value={String(episode.frames.length)} />
            <Meta label="Robot" value={episode.robot_id.slice(0, 8)} />
            <Meta label="Started" value={new Date(episode.started_at).toLocaleString()} />
          </div>
        </div>

        <div className="card p-4 lg:col-span-2">
          <div className="text-xs uppercase tracking-wide text-muted mb-2">
            Action-confidence &amp; contact-force trace
          </div>
          <ConfidenceChart trace={attr?.confidence_trace ?? episode.frames} />
          <div className="flex gap-4 text-xs text-muted mt-1">
            <span className="flex items-center gap-1"><i className="w-2 h-2 rounded-full bg-accent inline-block" /> confidence</span>
            <span className="flex items-center gap-1"><i className="w-2 h-2 rounded-full bg-motor inline-block" /> contact force</span>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className="card p-4">
          <div className="text-xs uppercase tracking-wide text-muted mb-3">Attention rollout</div>
          <Heatmap uri={attr?.attention_map_uri ?? null} />
        </div>

        <div className="card p-4 space-y-3">
          <div className="text-xs uppercase tracking-wide text-muted">Instruction sensitivity (counterfactual)</div>
          {!cf ? (
            <p className="text-sm text-muted">
              Runs automatically for grounding failures when you attribute. Motor/perception
              failures skip it.
            </p>
          ) : cf.applicable === false ? (
            <p className="text-sm text-muted">Not applicable: {cf.reason}</p>
          ) : (
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <span className={`pill ${cf.verdict === "grounding-sensitive" ? "bg-grounding/10 text-grounding" : "bg-ok/10 text-ok"}`}>
                  {cf.verdict}
                </span>
                <span className="text-body-sm text-ash">sensitivity {Math.round(cf.sensitivity * 100)}%</span>
              </div>
              <div className="text-caption font-mono text-ash">base → {cf.base_target}</div>
              <ul className="text-body-sm space-y-1">
                {cf.trials?.map((t: any, i: number) => (
                  <li key={i} className="flex justify-between gap-2 border-b border-mist py-1.5">
                    <span className="text-charcoal truncate">“{t.instruction}”</span>
                    <span className={`font-mono text-xs ${t.changed ? "text-motor" : "text-muted"}`}>
                      {t.target}{t.changed ? " ↯" : ""}
                    </span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function Meta({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div className="text-ash text-caption">{label}</div>
      <div className="font-mono text-graphite">{value}</div>
    </div>
  );
}
