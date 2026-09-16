"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { api, Cluster, SURFACE_BLURB, recomputeClusters } from "@/lib/api";
import {
  ActionButton,
  CardSkeleton,
  Empty,
  ErrorNote,
  PageHeader,
  SurfacePill,
} from "@/components/ui";

export default function ClustersPage() {
  const [clusters, setClusters] = useState<Cluster[]>([]);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>(null);

  function load() {
    setLoading(true);
    api
      .listClusters()
      .then((c) => {
        setClusters(c);
        setError(null);
      })
      .catch(setError)
      .finally(() => setLoading(false));
  }
  useEffect(load, []);

  async function recompute() {
    setBusy(true);
    try {
      setClusters(await recomputeClusters());
      setError(null);
    } catch (e) {
      setError(e);
    } finally {
      setBusy(false);
    }
  }

  const total = clusters.reduce((n, c) => n + c.episode_count, 0);
  const largest = Math.max(...clusters.map((c) => c.episode_count), 1);

  return (
    <div className="space-y-5">
      <PageHeader
        title="Failure clusters"
        subtitle="Recurring failure modes across the fleet, largest first."
      >
        <ActionButton onClick={recompute} busy={busy} variant="btn-accent">
          Recompute clusters
        </ActionButton>
      </PageHeader>

      <ErrorNote error={error} />

      {loading ? (
        <CardSkeleton />
      ) : error ? null : clusters.length === 0 ? (
        <Empty>
          No clusters yet. Classify some failed episodes, then{" "}
          <span className="text-charcoal">Recompute</span>.
        </Empty>
      ) : (
        <>
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
            {clusters.map((c) => (
              <Link key={c.id} href={`/clusters/${c.id}`} className="card-link group block p-6">
                <div className="flex items-start justify-between gap-3">
                  <SurfacePill surface={c.dominant_surface} />
                  <div className="text-right">
                    <div className="serif text-heading-sm tabular-nums text-graphite">
                      {c.episode_count}
                    </div>
                    <div className="text-caption text-fog">
                      {Math.round((c.episode_count / total) * 100)}% of clustered
                    </div>
                  </div>
                </div>

                {/* Size relative to the largest cluster — scannable across cards. */}
                <div className="mt-3 h-1 w-full overflow-hidden rounded-full bg-mist/60">
                  <div
                    className="h-full bg-signal/60"
                    style={{ width: `${(c.episode_count / largest) * 100}%` }}
                  />
                </div>

                <div className="mt-3 font-mono text-body-sm text-charcoal group-hover:text-cerulean transition">
                  {c.label}
                </div>
                {c.dominant_surface && (
                  <p className="mt-1 text-caption leading-relaxed text-ash">
                    {SURFACE_BLURB[c.dominant_surface]}
                  </p>
                )}
                {c.representative_episode_id && (
                  <div className="mt-2 text-caption text-fog">
                    representative{" "}
                    <span className="font-mono">
                      {c.representative_episode_id.slice(0, 8)}
                    </span>
                  </div>
                )}
              </Link>
            ))}
          </div>
          <p className="text-caption text-ash">
            {clusters.length} cluster{clusters.length === 1 ? "" : "s"} covering {total} episode
            {total === 1 ? "" : "s"}.
          </p>
        </>
      )}
    </div>
  );
}
