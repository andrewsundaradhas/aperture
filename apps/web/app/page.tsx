"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import {
  api,
  Cluster,
  EpisodeSummary,
  SURFACES,
  fetchAllEpisodes,
  surfaceVerdict,
} from "@/lib/api";
import {
  Empty,
  ErrorNote,
  SectionHead,
  Skeleton,
  StatTile,
} from "@/components/ui";
import { FleetScatter } from "@/components/FleetScatter";
import { SurfaceMix } from "@/components/SurfaceMix";
import { ClusterBars } from "@/components/ClusterBars";
import { PipelineTrack } from "@/components/PipelineTrack";

export default function Home() {
  const [episodes, setEpisodes] = useState<EpisodeSummary[] | null>(null);
  const [clusters, setClusters] = useState<Cluster[] | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [truncated, setTruncated] = useState(false);

  useEffect(() => {
    // The whole fleet, not page one — every reading on this page is a fleet-wide total.
    Promise.all([fetchAllEpisodes(), api.listClusters()])
      .then(([e, c]) => {
        setEpisodes(e.episodes);
        setTruncated(!e.complete);
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

  const clustered = useMemo(
    () => (clusters ?? []).reduce((n, c) => n + c.episode_count, 0),
    [clusters],
  );

  return (
    <div className="space-y-section">
      {/* ── Hero ─────────────────────────────────────────────────────────── */}
      <section className="space-y-10 pt-2">
        <div className="grid gap-10 lg:grid-cols-[1.3fr_1fr] lg:items-start">
          <div className="space-y-5">
            <span className="tag tag-ink">Evaluation &amp; interpretability</span>
            <h1 className="max-w-2xl text-display font-medium text-forest-ink">
              Why the policy failed, not just that it did.
            </h1>
            <p className="max-w-xl text-body-lg text-slate-smoke">
              Aperture classifies the failure surface behind every episode, attributes it with
              interpretability techniques, clusters recurring modes across a fleet, exports a
              scoped fine-tune set, and verifies the retrain actually worked.
            </p>
            <div className="flex flex-wrap items-center gap-3 pt-1">
              <Link href="/episodes" className="btn-filled">
                Browse episodes
              </Link>
              <Link href="/clusters" className="btn">
                View clusters
              </Link>
            </div>
          </div>

          {/* The floating info card, upper-right, as the system specifies. */}
          <aside className="card gridded-fine p-5">
            <div className="cinetype text-[11px] text-slate-smoke">Reading this page</div>
            <div className="mt-3 h-px w-full bg-lichen" aria-hidden />
            <dl className="mt-3 space-y-3 text-body">
              <div>
                <dt className="muoto text-caption text-slate-smoke">the plot</dt>
                <dd className="text-forest-ink">
                  Every episode is a mark. Height is how confident the classifier is in its
                  verdict.
                </dd>
              </div>
              <div>
                <dt className="muoto text-caption text-slate-smoke">filled vs hollow</dt>
                <dd className="text-forest-ink">
                  A filled mark carries an attributed failure surface. A hollow one does not —
                  it succeeded, or nothing fired.
                </dd>
              </div>
              <div>
                <dt className="muoto text-caption text-slate-smoke">the point</dt>
                <dd className="text-forest-ink">
                  Marks drifting to the top are explained failures. Everything in the lane below
                  is still unexplained.
                </dd>
              </div>
            </dl>
          </aside>
        </div>
      </section>

      <ErrorNote error={error} />

      {/* ── The fleet, plotted ───────────────────────────────────────────── */}
      {!error && (
        <section className="space-y-5">
          <SectionHead
            left="fleet scatter"
            right="confidence ↑"
            title="Every episode, plotted"
            action={
              <Link href="/episodes" className="link text-body">
                [ all episodes ]
              </Link>
            }
          />

          {!episodes ? (
            <Skeleton className="h-[420px] w-full" />
          ) : episodes.length === 0 ? (
            <Empty>
              No episodes ingested yet. Seed the demo with{" "}
              <span className="muoto text-forest-ink">
                cd apps/api &amp;&amp; python -m aperture.seed
              </span>
              .
            </Empty>
          ) : (
            <>
              <FleetScatter episodes={episodes} />
              {truncated && (
                <p className="muoto text-caption text-slate-smoke">
                  Showing the most recent {episodes.length} episodes — the fleet has more.
                </p>
              )}
            </>
          )}
        </section>
      )}

      {/* ── Fleet at a glance ────────────────────────────────────────────── */}
      {!error && (
        <section className="space-y-5">
          <SectionHead left="fleet totals" right="measured" title="At a glance" />

          {!stats ? (
            <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
              {Array.from({ length: 4 }).map((_, i) => (
                <div key={i} className="card space-y-3 p-5">
                  <Skeleton className="h-3 w-20" />
                  <Skeleton className="h-8 w-14" />
                </div>
              ))}
            </div>
          ) : (
            <>
              <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
                <StatTile
                  label="Episodes"
                  value={stats.total}
                  note={`${stats.robots} robots reporting`}
                />
                <StatTile
                  label="Failures"
                  value={stats.failures}
                  note={
                    stats.total
                      ? `${Math.round((stats.failures / stats.total) * 100)}% of episodes`
                      : "—"
                  }
                />
                <StatTile
                  label="Attributed"
                  value={stats.attributed}
                  note={
                    stats.unattributed > 0
                      ? `${stats.unattributed} still inconclusive`
                      : "every failure has a surface"
                  }
                />
                <StatTile
                  label="Clusters"
                  value={clusters?.length ?? 0}
                  note="recurring failure modes"
                />
              </div>

              <div className="card p-6">
                <div className="cinetype text-[11px] text-slate-smoke">Where failures land</div>
                <div className="mt-4">
                  <SurfaceMix counts={stats.bySurface} total={stats.attributed} />
                </div>
              </div>
            </>
          )}
        </section>
      )}

      {/* ── Failure modes ────────────────────────────────────────────────── */}
      {!error && (clusters?.length ?? 0) > 0 && (
        <section className="space-y-5">
          <SectionHead
            left="cluster size"
            right="largest first"
            title="Recurring failure modes"
            action={
              <Link href="/clusters" className="link text-body">
                [ all clusters ]
              </Link>
            }
          />
          <ClusterBars clusters={clusters!.slice(0, 6)} />
        </section>
      )}

      {/* ── The loop ─────────────────────────────────────────────────────── */}
      {!error && stats && (
        <section className="space-y-6">
          <SectionHead left="chaos" right="clarity" title="Closing the loop" />
          <div className="card p-8">
            <PipelineTrack
              stages={[
                {
                  label: "Ingestion",
                  value: String(stats.total),
                  note: "RLDS + LeRobot v3 episodes",
                },
                {
                  label: "Evaluation",
                  value: String(stats.attributed),
                  note: "failure surfaces attributed",
                },
                {
                  label: "Clustering",
                  value: String(clustered),
                  note: "episodes in a failure mode",
                },
                {
                  label: "Loop closure",
                  value: String(clusters?.length ?? 0),
                  note: "modes ready to verify",
                },
              ]}
            />
          </div>
        </section>
      )}

      {/* ── Entry points ─────────────────────────────────────────────────── */}
      <section className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <Link href="/episodes" className="card-link group block p-6">
          <div className="cinetype text-[11px] text-slate-smoke">
            Ingestion → Evaluation → Interpretability
          </div>
          <div className="mt-3 text-heading-sm font-medium text-forest-ink group-hover:text-deep-fern">
            Episodes
          </div>
          <p className="mt-2 max-w-md text-body leading-relaxed text-slate-smoke">
            Every ingested episode with its failure surface, confidence trace, attention map,
            and instruction-sensitivity probe.
          </p>
        </Link>
        <Link href="/clusters" className="card-link group block p-6">
          <div className="cinetype text-[11px] text-slate-smoke">
            Clustering → Export → Loop closure
          </div>
          <div className="mt-3 text-heading-sm font-medium text-forest-ink group-hover:text-deep-fern">
            Failure clusters
          </div>
          <p className="mt-2 max-w-md text-body leading-relaxed text-slate-smoke">
            Fleet-level failure patterns. Export a scoped dataset and verify the before/after
            success-rate delta after a retrain.
          </p>
        </Link>
      </section>
    </div>
  );
}
