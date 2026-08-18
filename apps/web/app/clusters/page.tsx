"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { api, Cluster } from "@/lib/api";
import { Empty, ErrorNote, SurfacePill, Spinner } from "@/components/ui";

export default function ClustersPage() {
  const [clusters, setClusters] = useState<Cluster[]>([]);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>(null);

  function load() {
    setLoading(true);
    api.listClusters().then(setClusters).catch(setError).finally(() => setLoading(false));
  }
  useEffect(load, []);

  async function recompute() {
    setBusy(true);
    try {
      setClusters(await api.recomputeClusters());
    } catch (e) {
      setError(e);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="space-y-5">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="serif text-heading text-graphite">Failure clusters</h1>
          <p className="text-body-sm text-ash mt-1">Recurring failure modes across the fleet, largest first.</p>
        </div>
        <button onClick={recompute} disabled={busy} className="btn-accent">
          {busy ? <Spinner /> : "Recompute clusters"}
        </button>
      </div>

      <ErrorNote error={error} />

      {loading ? (
        <div className="card p-8 flex justify-center"><Spinner /></div>
      ) : clusters.length === 0 ? (
        <Empty>No clusters yet. Classify some failed episodes, then Recompute.</Empty>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {clusters.map((c) => (
            <Link key={c.id} href={`/clusters/${c.id}`} className="card p-6 hover:border-signal/50 transition">
              <div className="flex items-center justify-between">
                <SurfacePill surface={c.dominant_surface} />
                <span className="serif text-heading-sm text-graphite tabular-nums">{c.episode_count}</span>
              </div>
              <div className="mt-3 font-mono text-body-sm text-charcoal">{c.label}</div>
              <div className="text-caption text-ash mt-1">
                {c.episode_count} episodes in this failure mode
                {c.representative_episode_id && (
                  <> · representative <span className="font-mono">{c.representative_episode_id.slice(0, 8)}</span></>
                )}
              </div>
            </Link>
          ))}
        </div>
      )}
    </div>
  );
}
