"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import {
  api,
  Cluster,
  EpisodeSummary,
  SURFACES,
  SURFACE_BLURB,
  surfaceVerdict,
} from "@/lib/api";
import { Empty, ErrorNote, Skeleton, SurfacePill } from "@/components/ui";

const SURFACE_BAR: Record<string, string> = {
  perception: "bg-perception",
  grounding: "bg-grounding",
  motor: "bg-motor",
};

export default function Home() {
  const [episodes, setEpisodes] = useState<EpisodeSummary[] | null>(null);
  const [clusters, setClusters] = useState<Cluster[] | null>(null);
  const [error, setError] = useState<unknown>(null);

  useEffect(() => {
    Promise.all([api.listEpisodes(), api.listClusters()])
      .then(([e, c]) => {
        setEpisodes(e);
        setClusters(c);
      })
      .catch(setError);
  }, []);

  const stats = useMemo(() => {
    if (!episodes) return null;
    const failures = episodes.filter((e) => e.outcome !== "success");
    // Only count verdicts that are actual findings — a tie-broken 0% surface is not one.
    const attributed = failures.filter(
      (e) =>
        surfaceVerdict(e.surface, e.classification_confidence, e.outcome).kind === "surface",
    );
    const bySurface = Object.fromEntries(
      SURFACES.map((s) => [s, attributed.filter((e) => e.surface === s).length]),
    ) as Record<string, number>;
    return {
      total: episodes.length,
      failures: failures.length,
      attributed: attributed.length,
      unattributed: failures.length - attributed.length,
      bySurface,
      robots: new Set(episodes.map((e) => e.robot_id)).size,
    };
  }, [episodes]);

  const topClusters = useMemo(() => (clusters ?? []).slice(0, 3), [clusters]);

  return (
    <div className="space-y-16">
      <section className="max-w-3xl space-y-6 pt-4">
        <span className="pill border border-mist bg-linen text-ash">
          Ingestion → Evaluation → Interpretability → Loop closure
        </span>
        <h1 className="serif text-heading-lg text-graphite">
          The evaluation &amp; interpretability layer for VLA robot policies
        </h1>
        <p className="max-w-2xl text-subheading font-normal leading-relaxed text-ash">
          Aperture classifies <em className="not-italic text-charcoal">why</em> a policy failed,
          attributes it with interpretability techniques, clusters similar failures across a
          fleet, exports a scoped fine-tune dataset, and verifies the fix worked after
          retraining.
        </p>
        <div className="flex items-center gap-3 pt-1">
          <Link href="/episodes" className="btn-accent">
            Browse episodes
            <span className="inline-flex h-4 w-4 items-center justify-center rounded-full border border-signal text-[10px] leading-none">
              →
            </span>
          </Link>
          <Link href="/clusters" className="btn">
            View clusters
          </Link>
        </div>
      </section>

      <ErrorNote error={error} />

      {!error && (
        <section className="space-y-4">
          <div className="flex items-baseline justify-between gap-4">
            <h2 className="serif text-heading-sm text-graphite">Fleet at a glance</h2>
            <Link href="/episodes" className="text-body-sm text-signal hover:underline">
              All episodes →
            </Link>
          </div>

          {!stats ? (
            <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
              {Array.from({ length: 4 }).map((_, i) => (
                <div key={i} className="card space-y-3 p-5">
                  <Skeleton className="h-3 w-20" />
                  <Skeleton className="h-8 w-14" />
                </div>
              ))}
            </div>
          ) : stats.total === 0 ? (
            <Empty>
              No episodes ingested yet. Seed the demo with{" "}
              <span className="font-mono text-charcoal">
                cd apps/api &amp;&amp; python -m aperture.seed
              </span>
              .
            </Empty>
          ) : (
            <>
              <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
                <Stat label="Episodes" value={stats.total} note={`${stats.robots} robots`} />
                <Stat
                  label="Failures"
                  value={stats.failures}
                  note={`${Math.round((stats.failures / stats.total) * 100)}% of episodes`}
                />
                <Stat
                  label="Attributed"
                  value={stats.attributed}
                  note={
                    stats.unattributed > 0
                      ? `${stats.unattributed} inconclusive`
                      : "every failure has a surface"
                  }
                />
                <Stat
                  label="Clusters"
                  value={clusters?.length ?? 0}
                  note="recurring failure modes"
                />
              </div>

              <div className="card space-y-4 p-6">
                <div className="text-caption uppercase tracking-wide text-ash">
                  Where failures land
                </div>
                {stats.attributed === 0 ? (
                  <p className="text-body-sm text-ash">
                    No failure has a confident surface yet. Open an episode and run{" "}
                    <span className="text-charcoal">Classify</span>.
                  </p>
                ) : (
                  <>
                    {/* One bar, three segments — the fleet's failure mix at a glance. */}
                    <div className="flex h-2.5 w-full overflow-hidden rounded-full bg-mist/60">
                      {SURFACES.map((s) =>
                        stats.bySurface[s] ? (
                          <div
                            key={s}
                            className={SURFACE_BAR[s]}
                            style={{ width: `${(stats.bySurface[s] / stats.attributed) * 100}%` }}
                            title={`${s}: ${stats.bySurface[s]}`}
                          />
                        ) : null,
                      )}
                    </div>
                    <ul className="grid grid-cols-1 gap-3 sm:grid-cols-3">
                      {SURFACES.map((s) => (
                        <li key={s} className="space-y-1">
                          <div className="flex items-center gap-2">
                            <SurfacePill surface={s} confidence={1} />
                            <span className="ml-auto font-mono text-body-sm tabular-nums text-graphite">
                              {stats.bySurface[s]}
                            </span>
                          </div>
                          <p className="text-caption leading-relaxed text-ash">
                            {SURFACE_BLURB[s]}
                          </p>
                        </li>
                      ))}
                    </ul>
                  </>
                )}
              </div>
            </>
          )}
        </section>
      )}

      {!error && topClusters.length > 0 && (
        <section className="space-y-4">
          <div className="flex items-baseline justify-between gap-4">
            <h2 className="serif text-heading-sm text-graphite">Largest failure modes</h2>
            <Link href="/clusters" className="text-body-sm text-signal hover:underline">
              All clusters →
            </Link>
          </div>
          <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
            {topClusters.map((c) => (
              <Link key={c.id} href={`/clusters/${c.id}`} className="card-link group block p-5">
                <div className="flex items-center justify-between">
                  <SurfacePill surface={c.dominant_surface} />
                  <span className="serif text-heading-sm tabular-nums text-graphite">
                    {c.episode_count}
                  </span>
                </div>
                <div className="mt-3 font-mono text-body-sm text-charcoal transition group-hover:text-cerulean">
                  {c.label}
                </div>
                <p className="mt-1 text-caption text-ash">
                  Export a scoped dataset and verify the retrain →
                </p>
              </Link>
            ))}
          </div>
        </section>
      )}

      <section className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <Link href="/episodes" className="card-link group block p-6">
          <div className="mb-2 text-caption uppercase tracking-wide text-ash">
            Ingestion → Evaluation → Interpretability
          </div>
          <div className="serif text-heading-sm text-graphite transition group-hover:text-signal">
            Episodes
          </div>
          <p className="mt-2 text-body-sm leading-relaxed text-ash">
            Every ingested episode with its failure surface, confidence trace, attention map,
            and instruction-sensitivity probe.
          </p>
        </Link>
        <Link href="/clusters" className="card-link group block p-6">
          <div className="mb-2 text-caption uppercase tracking-wide text-ash">
            Clustering → Export → Loop closure
          </div>
          <div className="serif text-heading-sm text-graphite transition group-hover:text-signal">
            Failure clusters
          </div>
          <p className="mt-2 text-body-sm leading-relaxed text-ash">
            Fleet-level failure patterns. Export a scoped dataset and verify the before/after
            success-rate delta after a retrain.
          </p>
        </Link>
      </section>
    </div>
  );
}

function Stat({ label, value, note }: { label: string; value: number; note: string }) {
  return (
    <div className="card p-5">
      <div className="text-caption uppercase tracking-wide text-ash">{label}</div>
      <div className="serif mt-1 text-heading tabular-nums text-graphite">{value}</div>
      <div className="mt-1 text-caption text-fog">{note}</div>
    </div>
  );
}
