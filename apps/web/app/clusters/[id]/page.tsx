"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import Link from "next/link";
import { api, blobUrl, ClusterDetail, SURFACE_BLURB, VerifyResult } from "@/lib/api";
import {
  ActionButton,
  Empty,
  ErrorNote,
  Eyebrow,
  OutcomePill,
  Skeleton,
  SurfacePill,
} from "@/components/ui";

export default function ClusterDetailPage() {
  const { id } = useParams<{ id: string }>();
  const router = useRouter();
  const [cluster, setCluster] = useState<ClusterDetail | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [exportUrl, setExportUrl] = useState<string | null>(null);
  const [exportCount, setExportCount] = useState<number | null>(null);
  const [format, setFormat] = useState("lerobot");
  const [verify, setVerify] = useState<VerifyResult | null>(null);
  const [fileCount, setFileCount] = useState(0);
  const fileRef = useRef<HTMLInputElement>(null);

  const load = useCallback(() => {
    api
      .getCluster(id)
      .then((c) => {
        setCluster(c);
        setError(null);
      })
      .catch(setError);
  }, [id]);
  useEffect(load, [load]);

  async function doExport() {
    setBusy("export");
    setExportUrl(null);
    try {
      const res = await api.exportDataset(id, format);
      setExportUrl(blobUrl(res.download_url));
      setExportCount(res.episode_count);
      setError(null);
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
      setError(null);
    } catch (e) {
      setError(e);
    } finally {
      setBusy(null);
    }
  }

  if (error && !cluster) return <ErrorNote error={error} />;
  if (!cluster) return <ClusterSkeleton />;

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center gap-3">
        <Link href="/clusters" className="link">
          &larr; Clusters
        </Link>
        <SurfacePill surface={cluster.dominant_surface} />
        <span className="muoto text-body text-forest-ink">{cluster.label}</span>
        <span className="ml-auto text-body text-slate-smoke">
          {cluster.episode_count} episode{cluster.episode_count === 1 ? "" : "s"}
        </span>
      </div>

      {cluster.dominant_surface && (
        <p className="max-w-2xl text-body leading-relaxed text-slate-smoke">
          {SURFACE_BLURB[cluster.dominant_surface]}
        </p>
      )}

      <ErrorNote error={error} />

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        {/* Scoped dataset export */}
        <div className="card space-y-3 p-4">
          <Eyebrow>Scoped fine-tune export</Eyebrow>
          <p className="text-body text-slate-smoke">
            Exports only this cluster&rsquo;s {cluster.episode_count} episodes — not the whole
            fleet.
          </p>
          <div className="flex flex-wrap items-center gap-2">
            <label htmlFor="export-format" className="sr-only">
              Export format
            </label>
            <select
              id="export-format"
              value={format}
              onChange={(e) => setFormat(e.target.value)}
              className="field w-auto"
            >
              <option value="lerobot">LeRobot</option>
              <option value="rlds">RLDS</option>
            </select>
            <ActionButton
              onClick={doExport}
              busy={busy === "export"}
              disabled={!!busy}
              variant="btn-filled"
            >
              Export dataset
            </ActionButton>
          </div>
          {exportUrl && (
            <a
              href={exportUrl}
              target="_blank"
              rel="noreferrer"
              className="link inline-flex items-center gap-1.5 text-body"
            >
              &darr; Download {exportCount ?? cluster.episode_count} episodes ·{" "}
              {format.toUpperCase()}
            </a>
          )}
        </div>

        {/* Loop-closure verify */}
        <div className="card space-y-3 p-4">
          <Eyebrow>Verify fix (loop closure)</Eyebrow>
          <p className="text-body text-slate-smoke">
            Upload a post-retrain batch of the same tasks; Aperture re-measures the same failure
            signature and computes the before/after delta.
          </p>
          <div className="flex flex-wrap items-center gap-2">
            <label htmlFor="verify-files" className="btn cursor-pointer">
              Choose files
            </label>
            <input
              id="verify-files"
              ref={fileRef}
              type="file"
              multiple
              // The verify endpoint takes everything ingestion takes — RLDS .tfrecord
              // streams and LeRobot v3 archives, not just the portable JSON fixtures.
              accept=".json,.tfrecord,.tfrecords,.zip,.tar,.tar.gz,.tgz"
              onChange={(e) => setFileCount(e.target.files?.length ?? 0)}
              className="sr-only"
            />
            <span className="muoto text-caption text-slate-smoke">
              {fileCount ? `${fileCount} file${fileCount === 1 ? "" : "s"} selected` : "no files selected"}
            </span>
            <ActionButton
              onClick={doVerify}
              busy={busy === "verify"}
              disabled={!!busy || fileCount === 0}
            >
              Run verification
            </ActionButton>
          </div>
          {verify && <BeforeAfter v={verify} />}
        </div>
      </div>

      <div className="card overflow-hidden">
        <div className="border-b border-lichen px-4 py-3">
          <Eyebrow>Episodes in this cluster</Eyebrow>
        </div>
        {cluster.episodes.length === 0 ? (
          <Empty>Empty cluster.</Empty>
        ) : (
          <div className="table-scroll">
            <table className="data">
              <thead>
                <tr>
                  <th>Episode</th>
                  <th>Instruction</th>
                  <th>Outcome</th>
                  <th>Surface</th>
                </tr>
              </thead>
              <tbody>
                {cluster.episodes.map((e) => (
                  <tr
                    key={e.id}
                    className="row-link cursor-pointer"
                    onClick={() => router.push(`/episodes/${e.id}`)}
                  >
                    <td>
                      <Link
                        href={`/episodes/${e.id}`}
                        onClick={(ev) => ev.stopPropagation()}
                        className="row-link-label muoto text-deep-fern"
                      >
                        {e.id.slice(0, 8)}
                      </Link>
                    </td>
                    <td className="text-forest-ink">{e.instruction ?? "—"}</td>
                    <td>
                      <OutcomePill outcome={e.outcome} />
                    </td>
                    <td>
                      <SurfacePill surface={e.surface} outcome={e.outcome} />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}

function BeforeAfter({ v }: { v: VerifyResult }) {
  const positive = v.delta > 0.001;
  const regressed = v.delta < -0.001;
  return (
    <div className="space-y-2 border-t border-lichen pt-3">
      <div className="flex items-end gap-3">
        <Bar label="before" value={v.pre_success_rate} tone="bg-lichen" />
        <Bar label="after" value={v.post_success_rate} tone={positive ? "bg-moss" : "bg-lichen"} />
        <div className="ml-auto text-right">
          <div className="muoto text-caption text-slate-smoke">success-rate &Delta;</div>
          <div
            className={`text-heading font-medium tabular-nums ${
              positive ? "text-deep-fern" : regressed ? "text-alarm" : "text-slate-smoke"
            }`}
          >
            {positive ? "+" : ""}
            {Math.round(v.delta * 100)}%
          </div>
        </div>
      </div>
      <div className="muoto text-caption text-slate-smoke">
        n={v.pre_n} before · n={v.post_n} matched after
      </div>
      {!positive && !regressed && (
        <p className="text-caption text-caution">
          No measurable movement — the retrain did not shift this failure signature.
        </p>
      )}
    </div>
  );
}

function Bar({ label, value, tone }: { label: string; value: number; tone: string }) {
  const pct = Math.round(value * 100);
  return (
    <div className="flex flex-col items-center gap-1">
      <div
        className="flex h-24 w-10 items-end overflow-hidden rounded-tag bg-ash-gray shadow-hairline"
        role="meter"
        aria-valuenow={pct}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-label={`${label} success rate`}
      >
        <div className={`w-full rounded-[4px] ${tone}`} style={{ height: `${pct}%` }} />
      </div>
      <span className="muoto text-caption text-slate-smoke">{label}</span>
      <span className="muoto text-caption tabular-nums text-forest-ink">{pct}%</span>
    </div>
  );
}

function ClusterSkeleton() {
  return (
    <div className="space-y-6" role="status" aria-label="loading cluster">
      <div className="flex items-center gap-3">
        <Skeleton className="h-4 w-24" />
        <Skeleton className="h-5 w-20 rounded-full" />
        <Skeleton className="h-4 w-36" />
      </div>
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <Skeleton className="h-40 w-full rounded-card" />
        <Skeleton className="h-40 w-full rounded-card" />
      </div>
      <Skeleton className="h-64 w-full rounded-card" />
    </div>
  );
}
