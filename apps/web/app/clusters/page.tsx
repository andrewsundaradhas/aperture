"use client";

import { useEffect, useMemo, useState } from "react";
import { api, Cluster, SURFACES, recomputeClusters } from "@/lib/api";
import {
  ActionButton,
  CardSkeleton,
  Empty,
  ErrorNote,
  PageHeader,
  SectionHead,
  StatTile,
} from "@/components/ui";
import { ClusterBars } from "@/components/ClusterBars";
import { SurfaceMix } from "@/components/SurfaceMix";

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

  // Clusters per surface — how the fleet's recurring modes distribute, not the episodes'.
  const mix = useMemo(() => {
    const counts = Object.fromEntries(
      SURFACES.map((s) => [
        s,
        clusters.filter((c) => c.dominant_surface === s).reduce((n, c) => n + c.episode_count, 0),
      ]),
    ) as Record<string, number>;
    return { counts, total: Object.values(counts).reduce((a, b) => a + b, 0) };
  }, [clusters]);

  const biggest = clusters.length
    ? clusters.reduce((a, b) => (b.episode_count > a.episode_count ? b : a))
    : null;

  return (
    <div className="space-y-8">
      <PageHeader
        title="Failure clusters"
        subtitle="Recurring failure modes across the fleet, largest first."
      >
        <ActionButton onClick={recompute} busy={busy} variant="btn-filled">
          Recompute clusters
        </ActionButton>
      </PageHeader>

      <ErrorNote error={error} />

      {loading ? (
        <CardSkeleton />
      ) : error ? null : clusters.length === 0 ? (
        <Empty>
          No clusters yet. Classify some failed episodes, then{" "}
          <span className="text-forest-ink">Recompute</span>.
        </Empty>
      ) : (
        <>
          <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
            <StatTile label="Clusters" value={clusters.length} note="recurring modes" />
            <StatTile label="Episodes clustered" value={total} note="across all modes" />
            <StatTile
              label="Largest mode"
              value={biggest?.episode_count ?? 0}
              note={biggest?.label ?? "—"}
            />
            <StatTile
              label="Mean size"
              value={clusters.length ? Math.round((total / clusters.length) * 10) / 10 : 0}
              note="episodes per mode"
            />
          </div>

          <section className="space-y-5">
            <SectionHead
              left="cluster size"
              right="largest first"
              title="Every failure mode"
            />
            <ClusterBars clusters={clusters} />
          </section>

          {mix.total > 0 && (
            <section className="space-y-5">
              <SectionHead
                left="surface"
                right="share of clustered"
                title="What the modes are made of"
              />
              <div className="card p-6">
                <SurfaceMix counts={mix.counts} total={mix.total} />
              </div>
            </section>
          )}
        </>
      )}
    </div>
  );
}
