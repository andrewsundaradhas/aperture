"use client";

import Link from "next/link";
import { useRef, useState } from "react";

import { api, putUpload, recomputeClusters, waitForJob } from "@/lib/api";
import { ActionButton, ErrorNote, PageHeader, SectionHead, Spinner } from "@/components/ui";

type Stage = "waiting" | "uploading" | "ingesting" | "classifying" | "done" | "failed";

type Item = {
  file: File;
  stage: Stage;
  progress: number;
  episodes?: number;
  classified?: number;
  surfaces?: Record<string, number>;
  error?: string;
};

const ACCEPT = ".zip,.tar,.tar.gz,.tgz,.tfrecord,.json";

// Large real datasets can take a while to parse and classify; the default 2-minute poll is for
// single jobs on small inputs.
const LONG_JOB = { timeoutMs: 60 * 60 * 1000, intervalMs: 1500 };

function bytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 ** 2) return `${(n / 1024).toFixed(1)} KB`;
  if (n < 1024 ** 3) return `${(n / 1024 ** 2).toFixed(1)} MB`;
  return `${(n / 1024 ** 3).toFixed(2)} GB`;
}

const STAGE_LABEL: Record<Stage, string> = {
  waiting: "waiting",
  uploading: "uploading",
  ingesting: "parsing episodes",
  classifying: "classifying failures",
  done: "done",
  failed: "failed",
};

export default function UploadPage() {
  const input = useRef<HTMLInputElement>(null);
  const [items, setItems] = useState<Item[]>([]);
  const [running, setRunning] = useState(false);
  const [clustering, setClustering] = useState(false);
  const [clusterNote, setClusterNote] = useState<string | null>(null);
  const [error, setError] = useState<unknown>(null);

  function update(i: number, patch: Partial<Item>) {
    setItems((prev) => prev.map((it, k) => (k === i ? { ...it, ...patch } : it)));
  }

  function choose(files: FileList | null) {
    if (!files?.length) return;
    setItems(Array.from(files).map((file) => ({ file, stage: "waiting", progress: 0 })));
    setClusterNote(null);
  }

  async function runOne(i: number, file: File) {
    try {
      update(i, { stage: "uploading", progress: 0 });
      const p = await api.presign(file.name);
      await putUpload(p.upload_url, file, (f) => update(i, { progress: f }));

      update(i, { stage: "ingesting", progress: 1 });
      const accepted = await api.ingest(p.key, file.name);
      const ingest = await waitForJob(accepted.job_id, LONG_JOB);
      const episodes = ingest.result?.count ?? 0;
      update(i, { episodes });

      const classifyJob = ingest.result?.classify_job_id;
      if (classifyJob) {
        update(i, { stage: "classifying" });
        const c = await waitForJob(classifyJob, LONG_JOB);
        update(i, { classified: c.result?.classified ?? 0, surfaces: c.result?.surfaces ?? {} });
      }
      update(i, { stage: "done" });
    } catch (e) {
      update(i, { stage: "failed", error: String((e as Error)?.message ?? e) });
    }
  }

  async function start() {
    setRunning(true);
    setError(null);
    // One at a time: a fleet export is usually a few large archives, and parallel uploads
    // just split the same bandwidth while multiplying the worker's memory peak.
    for (let i = 0; i < items.length; i++) {
      if (items[i].stage === "waiting" || items[i].stage === "failed") await runOne(i, items[i].file);
    }
    setRunning(false);
  }

  async function cluster() {
    setClustering(true);
    setError(null);
    try {
      const clusters = await recomputeClusters();
      const n = clusters.reduce((a, c) => a + c.episode_count, 0);
      setClusterNote(`${clusters.length} cluster${clusters.length === 1 ? "" : "s"} covering ${n} failed episodes.`);
    } catch (e) {
      setError(e);
    } finally {
      setClustering(false);
    }
  }

  const finished = items.length > 0 && items.every((it) => it.stage === "done" || it.stage === "failed");
  const anyDone = items.some((it) => it.stage === "done");

  return (
    <div className="space-y-8">
      <PageHeader
        title="Upload episodes"
        subtitle="LeRobot v3 dataset archives, RLDS / Open X-Embodiment .tfrecord shards, or episode JSON. Failed episodes are classified automatically once parsed."
      />

      <ErrorNote error={error} />

      <section className="space-y-5">
        <SectionHead left="files" right="sent straight to storage" title="Choose files" />
        <div
          className="card flex flex-col items-center gap-3 p-10 text-center"
          onDragOver={(e) => e.preventDefault()}
          onDrop={(e) => {
            e.preventDefault();
            if (!running) choose(e.dataTransfer.files);
          }}
        >
          <p className="text-body text-slate-smoke">Drop files here, or</p>
          <input
            ref={input}
            type="file"
            multiple
            accept={ACCEPT}
            className="sr-only"
            id="upload-input"
            onChange={(e) => choose(e.target.files)}
            disabled={running}
          />
          <label htmlFor="upload-input" className="btn cursor-pointer">
            Browse…
          </label>
          <p className="muoto text-caption text-slate-smoke">{ACCEPT.split(",").join("  ")}</p>
        </div>
      </section>

      {items.length > 0 && (
        <section className="space-y-5">
          <SectionHead
            left="progress"
            right={`${items.length} file${items.length === 1 ? "" : "s"}`}
            title="Queue"
            action={
              <ActionButton onClick={start} busy={running} disabled={running || finished} variant="btn-filled">
                {running ? "Working" : "Upload & ingest"}
              </ActionButton>
            }
          />
          <ul className="space-y-3">
            {items.map((it, i) => (
              <li key={i} className="card space-y-2 p-5">
                <div className="flex flex-wrap items-center gap-3">
                  <span className="text-body font-medium text-forest-ink">{it.file.name}</span>
                  <span className="muoto text-caption text-slate-smoke">{bytes(it.file.size)}</span>
                  <span className="ml-auto flex items-center gap-2">
                    {["uploading", "ingesting", "classifying"].includes(it.stage) && <Spinner />}
                    <span className={`tag ${it.stage === "failed" ? "text-alarm" : ""}`}>{STAGE_LABEL[it.stage]}</span>
                  </span>
                </div>
                {it.stage === "uploading" && (
                  <div className="h-1.5 w-full overflow-hidden rounded-full bg-lichen">
                    <div className="h-full rounded-full bg-forest-ink" style={{ width: `${Math.round(it.progress * 100)}%` }} />
                  </div>
                )}
                {it.episodes != null && (
                  <p className="text-caption text-slate-smoke">
                    {it.episodes} episode{it.episodes === 1 ? "" : "s"} ingested
                    {it.classified != null && (
                      <>
                        {" · "}
                        {it.classified} failure{it.classified === 1 ? "" : "s"} classified
                        {it.surfaces && Object.keys(it.surfaces).length > 0 && (
                          <>
                            {" ("}
                            {Object.entries(it.surfaces)
                              .map(([s, n]) => `${n} ${s}`)
                              .join(", ")}
                            {")"}
                          </>
                        )}
                      </>
                    )}
                  </p>
                )}
                {it.error && <p className="muoto wrap-break-word text-caption text-alarm">{it.error}</p>}
              </li>
            ))}
          </ul>
        </section>
      )}

      {finished && anyDone && (
        <section className="card flex flex-wrap items-center gap-4 p-6">
          <p className="text-body text-forest-ink">
            Next: group the new failures into recurring modes.
            {clusterNote && <span className="ml-2 text-slate-smoke">{clusterNote}</span>}
          </p>
          <div className="ml-auto flex gap-2">
            <ActionButton onClick={cluster} busy={clustering}>
              Recompute clusters
            </ActionButton>
            <Link href="/episodes" className="btn">
              View episodes
            </Link>
            {clusterNote && (
              <Link href="/clusters" className="btn">
                View clusters
              </Link>
            )}
          </div>
        </section>
      )}
    </div>
  );
}
