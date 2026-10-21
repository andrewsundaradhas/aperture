"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import Link from "next/link";
import { api, blobUrl, ClusterDetail, SURFACE_BLURB, VerifyResult, clusterTitle } from "@/lib/api";
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
  // Large modes hold thousands of episodes; render a page at a time.
  const [shown, setShown] = useState(50);
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
      {/* Its own layer: the condensed headline's tall glyph box sits over it and swallowed clicks. */}
      <div className="relative z-10">
        <Link href="/clusters" className="link inline-block py-1 text-body">
          &larr; Failure modes
        </Link>
      </div>

      <div className="space-y-2">
        <h1 className="text-heading text-lavender">{clusterTitle(cluster)}</h1>
        <p className="max-w-2xl text-body leading-relaxed text-slate-smoke">
          {cluster.episode_count} failed episode{cluster.episode_count === 1 ? "" : "s"} that look
          alike.{" "}
          {cluster.dominant_surface ? SURFACE_BLURB[cluster.dominant_surface] : ""}
        </p>
      </div>

      <ErrorNote error={error} />

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        {/* Scoped dataset export */}
        <div className="card space-y-3 p-4">
          <Eyebrow>Export as training data</Eyebrow>
          <p className="text-body text-slate-smoke">
            Download just these {cluster.episode_count} episodes to fine-tune your policy on this
            failure mode.
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
              <option value="lerobot">LeRobot format</option>
              <option value="rlds">RLDS format</option>
            </select>
            <ActionButton
              onClick={doExport}
              busy={busy === "export"}
              disabled={!!busy}
              variant="btn-filled"
            >
              Export
            </ActionButton>
          </div>
          {exportUrl && (
            <a
              href={exportUrl}
              // Save the file rather than open it in a tab: it is a dataset, not a page.
              download={`${clusterTitle(cluster).replace(/[^a-z0-9]+/gi, "-").toLowerCase()}-${format}.json`}
              className="link inline-flex items-center gap-1.5 text-body"
            >
              &darr; Download {exportCount ?? cluster.episode_count} episodes ·{" "}
              {format.toUpperCase()}
            </a>
          )}
        </div>

        {/* Loop-closure verify */}
        <div className="card space-y-3 p-4">
          <Eyebrow>Check a retrained policy</Eyebrow>
          <p className="text-body text-slate-smoke">
            After retraining, upload new episodes of the same tasks. Aperture compares the success
            rate before and after.
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
              Compare
            </ActionButton>
          </div>
          {verify && <BeforeAfter v={verify} />}
        </div>
      </div>

      <div className="card overflow-hidden">
        <div className="border-b border-lichen px-4 py-3">
          <Eyebrow>Episodes in this failure mode</Eyebrow>
        </div>
        {cluster.episodes.length === 0 ? (
          <Empty>No episodes in this failure mode.</Empty>
        ) : (
          <div className="table-scroll">
            <table className="data">
              <thead>
                <tr>
                  <th>Task</th>
                  <th>Result</th>
                  <th>Failure cause</th>
                </tr>
              </thead>
              <tbody>
                {cluster.episodes.slice(0, shown).map((e) => (
                  <tr
                    key={e.id}
                    className="row-link cursor-pointer"
                    onClick={() => router.push(`/episodes/${e.id}`)}
                  >
                    <td className="max-w-md truncate">
                      <Link
                        href={`/episodes/${e.id}`}
                        onClick={(ev) => ev.stopPropagation()}
                        className="row-link-label text-forest-ink"
                      >
                        {e.instruction ?? "Untitled episode"}
                      </Link>
                    </td>
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
            {cluster.episodes.length > shown && (
              <div className="flex items-center justify-between gap-3 px-4 py-3 text-caption text-slate-smoke">
                <span>
                  Showing {shown} of {cluster.episodes.length} episodes
                </span>
                <button className="btn py-1.5" onClick={() => setShown((n) => n + 100)}>
                  Show more
                </button>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

function BeforeAfter({ v }: { v: VerifyResult }) {
  // No uploaded episode resembles this failure mode (different task or failure signature):
  // there is nothing to compare, which is not the same as the success rate dropping to zero.
  if (v.post_n === 0) {
    return (
      <p className="border-t border-lichen pt-3 text-body text-caution">
        None of the uploaded episodes match this failure mode, so there&rsquo;s nothing to
        compare yet. Upload new runs of the same tasks after retraining.
      </p>
    );
  }
  const positive = v.delta > 0.001;
  const regressed = v.delta < -0.001;
  return (
    <div className="space-y-2 border-t border-lichen pt-3">
      <div className="flex items-end gap-3">
        <Bar label="before" value={v.pre_success_rate} tone="bg-lichen" />
        <Bar label="after" value={v.post_success_rate} tone={positive ? "bg-moss" : "bg-lichen"} />
        <div className="ml-auto text-right">
          <div className="muoto text-caption text-slate-smoke">change in success rate</div>
          <div
            className={`font-display text-heading-sm tabular-nums ${
              positive ? "text-deep-fern" : regressed ? "text-alarm" : "text-slate-smoke"
            }`}
          >
            {positive ? "+" : ""}
            {Math.round(v.delta * 100)}%
          </div>
        </div>
      </div>
      <div className="muoto text-caption text-slate-smoke">
        {v.pre_n} episodes before · {v.post_n} matching episodes after
      </div>
      {!positive && !regressed && (
        <p className="text-caption text-caution">
          No measurable change — the retrained policy still fails this way.
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
        <div className={`w-full rounded-tag ${tone}`} style={{ height: `${pct}%` }} />
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
