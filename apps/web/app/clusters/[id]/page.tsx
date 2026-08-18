"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import { api, API_BASE, ClusterDetail, VerifyResult } from "@/lib/api";
import { Empty, ErrorNote, OutcomePill, Spinner, SurfacePill } from "@/components/ui";

export default function ClusterDetailPage() {
  const { id } = useParams<{ id: string }>();
  const [cluster, setCluster] = useState<ClusterDetail | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [exportUrl, setExportUrl] = useState<string | null>(null);
  const [format, setFormat] = useState("lerobot");
  const [verify, setVerify] = useState<VerifyResult | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  const load = useCallback(() => {
    api.getCluster(id).then(setCluster).catch(setError);
  }, [id]);
  useEffect(load, [load]);

  async function doExport() {
    setBusy("export");
    setExportUrl(null);
    try {
      const res = await api.exportDataset(id, format);
      const url = res.download_url.startsWith("http") ? res.download_url : `${API_BASE}${res.download_url}`;
      setExportUrl(url);
    } catch (e) {
      setError(e);
    } finally {
      setBusy(null);
    }
  }

  async function doVerify() {
    const files = Array.from(fileRef.current?.files ?? []);
    if (!files.length) return;
    setBusy("verify");
    try {
      setVerify(await api.verify(id, files));
    } catch (e) {
      setError(e);
    } finally {
      setBusy(null);
    }
  }

  if (error) return <ErrorNote error={error} />;
  if (!cluster) return <div className="card p-8 flex justify-center"><Spinner /></div>;

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3">
        <Link href="/clusters" className="text-ash hover:text-graphite transition">← Clusters</Link>
        <SurfacePill surface={cluster.dominant_surface} />
        <span className="font-mono text-body-sm text-charcoal">{cluster.label}</span>
        <span className="ml-auto text-body-sm text-ash">{cluster.episode_count} episodes</span>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Scoped dataset export */}
        <div className="card p-4 space-y-3">
          <div className="text-caption uppercase tracking-wide text-ash">Scoped fine-tune export</div>
          <p className="text-body-sm text-ash">
            Exports only this cluster’s {cluster.episode_count} episodes — not the whole fleet.
          </p>
          <div className="flex gap-2 items-center">
            <select value={format} onChange={(e) => setFormat(e.target.value)} className="field">
              <option value="lerobot">LeRobot</option>
              <option value="rlds">RLDS</option>
            </select>
            <button onClick={doExport} disabled={!!busy} className="btn-accent">
              {busy === "export" ? <Spinner /> : "Export dataset"}
            </button>
          </div>
          {exportUrl && (
            <a href={exportUrl} target="_blank" rel="noreferrer" className="text-signal text-body-sm hover:underline break-all">
              ↓ Download scoped dataset
            </a>
          )}
        </div>

        {/* Loop-closure verify */}
        <div className="card p-4 space-y-3">
          <div className="text-caption uppercase tracking-wide text-ash">Verify fix (loop closure)</div>
          <p className="text-body-sm text-ash">
            Upload a post-retrain batch of the same tasks; Aperture re-measures the same failure
            signature and computes the before/after delta.
          </p>
          <input ref={fileRef} type="file" multiple accept=".json" className="text-caption text-ash file:mr-2 file:btn" />
          <button onClick={doVerify} disabled={!!busy} className="btn">
            {busy === "verify" ? <Spinner /> : "Run verification"}
          </button>
          {verify && <BeforeAfter v={verify} />}
        </div>
      </div>

      <div className="card overflow-hidden">
        <div className="px-4 py-3 text-caption uppercase tracking-wide text-ash border-b border-mist">
          Episodes in this cluster
        </div>
        {cluster.episodes.length === 0 ? (
          <Empty>Empty cluster.</Empty>
        ) : (
          <table className="data">
            <thead>
              <tr><th>Episode</th><th>Instruction</th><th>Outcome</th><th>Surface</th></tr>
            </thead>
            <tbody>
              {cluster.episodes.map((e) => (
                <tr key={e.id}>
                  <td><Link href={`/episodes/${e.id}`} className="font-mono text-signal hover:underline">{e.id.slice(0, 8)}</Link></td>
                  <td className="text-charcoal">{e.instruction ?? "—"}</td>
                  <td><OutcomePill outcome={e.outcome} /></td>
                  <td><SurfacePill surface={e.surface} /></td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}

function BeforeAfter({ v }: { v: VerifyResult }) {
  const positive = v.delta > 0.001;
  return (
    <div className="space-y-2 pt-1">
      <div className="flex items-end gap-3">
        <Bar label="before" value={v.pre_success_rate} tone="bg-motor" />
        <Bar label="after" value={v.post_success_rate} tone="bg-ok" />
        <div className="ml-auto text-right">
          <div className="text-caption text-ash">success-rate Δ</div>
          <div className={`serif text-heading-sm tabular-nums ${positive ? "text-ok" : "text-ash"}`}>
            {positive ? "+" : ""}{Math.round(v.delta * 100)}%
          </div>
        </div>
      </div>
      <div className="text-caption text-ash">n={v.pre_n} before · n={v.post_n} matched after</div>
    </div>
  );
}

function Bar({ label, value, tone }: { label: string; value: number; tone: string }) {
  return (
    <div className="flex flex-col items-center gap-1">
      <div className="w-10 h-24 bg-linen border border-mist rounded-md flex items-end overflow-hidden">
        <div className={`w-full ${tone}`} style={{ height: `${Math.round(value * 100)}%` }} />
      </div>
      <span className="text-caption text-ash">{label}</span>
      <span className="text-caption font-mono text-charcoal tabular-nums">{Math.round(value * 100)}%</span>
    </div>
  );
}
