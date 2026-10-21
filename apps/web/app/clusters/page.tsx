"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { api, Cluster, recomputeClusters } from "@/lib/api";
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

  const biggest = clusters.length
    ? clusters.reduce((a, b) => (b.episode_count > a.episode_count ? b : a))
    : null;

  return (
    <div className="space-y-8">
      <PageHeader
        title="Failure modes"
        subtitle="Groups of failures that look alike. Fixing one mode fixes every episode in it."
      >
        <ActionButton onClick={recompute} busy={busy} variant="btn-filled">
          Find failure modes
        </ActionButton>
      </PageHeader>

      <ErrorNote error={error} />

      {loading ? (
        <CardSkeleton />
      ) : error ? null : clusters.length === 0 ? (
        <Empty>
          No failure modes yet.{" "}
          <Link href="/upload" className="link">
            Upload failed episodes
          </Link>
          , then press <span className="text-forest-ink">Find failure modes</span>.
        </Empty>
      ) : (
        <>
          <div className="grid grid-cols-2 gap-4 lg:grid-cols-3">
            <StatTile label="Failure modes" value={clusters.length} note="recurring patterns" />
            <StatTile label="Failures covered" value={total} note="episodes in a mode" />
            <StatTile
              label="Largest mode"
              value={biggest?.episode_count ?? 0}
              note="episodes"
            />
          </div>

          <section className="space-y-5">
            <SectionHead left="episodes per mode" right="largest first" title="All failure modes" />
            <div className="card p-3">
              <ClusterBars clusters={clusters} />
            </div>
          </section>

        </>
      )}
    </div>
  );
}
