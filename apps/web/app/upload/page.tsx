"use client";

import Link from "next/link";
import { useEffect, useRef, useState } from "react";

import { Outcome, SURFACE_LABEL, api, putUpload, recomputeClusters, waitForJob } from "@/lib/api";
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

// Shown as guidance only — not used as the file picker's `accept` filter: Open X-Embodiment
// shards are named `…tfrecord-00000-of-00128`, which no `accept` pattern can match, so a filter
// would hide the very files people need to upload. The server validates the format.
const FORMATS = "LeRobot v3 (.zip / .tar.gz)  ·  RLDS / Open X-Embodiment (.tfrecord)  ·  episode JSON";

// Public sample datasets, served from /public/samples (see scripts/build_samples.py).
const SAMPLES = [
  {
    file: "xarm-lift-40-episodes.zip",
    title: "xArm lift — 40 policy rollouts",
    detail:
      "LeRobot v3 · 0.9 MB · 32 failures, 8 successes. Rollouts of a pick-and-lift policy from the public lerobot/xarm_lift_medium dataset (recorded in simulation). Public datasets like this one don't log policy confidence or contact force, so its failures come back Undetermined — your own logs with those signals get specific causes.",
  },
  {
    file: "signal-logged-demo.tfrecord",
    title: "Signal-logged demo — every failure cause",
    detail:
      "RLDS · 13 KB · synthetic. 15 episodes carrying the per-step signals a deployed policy can log — action confidence, contact force, sub-goals — so Aperture names each cause: perception, instruction grounding and motor.",
  },
  {
    file: "nyu-rot-xarm-real-robot.tfrecord",
    title: "NYU ROT — 14 real-robot demonstrations",
    detail:
      "RLDS / Open X-Embodiment · 5.6 MB · real xArm hardware, tasks like “erase the board”. Demonstrations, so every episode succeeds.",
  },
];

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

  const queueRef = useRef<HTMLElement>(null);
  const [outcome, setOutcome] = useState<Outcome>("auto");
  const [dragging, setDragging] = useState(false);

  // A file dropped just outside the drop zone would otherwise make the browser navigate away
  // to display it, losing the page.
  useEffect(() => {
    const block = (e: DragEvent) => e.preventDefault();
    window.addEventListener("dragover", block);
    window.addEventListener("drop", block);
    return () => {
      window.removeEventListener("dragover", block);
      window.removeEventListener("drop", block);
    };
  }, []);

  /** Picking files starts the upload straight away — there is no second step to miss. */
  function choose(files: FileList | File[] | null) {
    if (!files?.length || running) return;
    const list: Item[] = Array.from(files).map((file) => ({ file, stage: "waiting", progress: 0 }));
    setItems(list);
    setClusterNote(null);
    // Clear the picker so choosing the same file again still fires a change event.
    if (input.current) input.current.value = "";
    requestAnimationFrame(() =>
      queueRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }),
    );
    void start(list);
  }

  async function trySample(name: string) {
    setError(null);
    try {
      const res = await fetch(`/samples/${name}`);
      if (!res.ok) throw new Error(`Couldn’t load the sample (${res.status}).`);
      choose([new File([await res.blob()], name)]);
    } catch (e) {
      setError(e);
    }
  }

  async function runOne(i: number, file: File) {
    try {
      update(i, { stage: "uploading", progress: 0 });
      const p = await api.presign(file.name);
      await putUpload(p.upload_url, file, (f) => update(i, { progress: f }));

      update(i, { stage: "ingesting", progress: 1 });
      const accepted = await api.ingest(p.key, file.name, outcome);
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

  async function start(list: Item[] = items) {
    setRunning(true);
    setError(null);
    // One at a time: a fleet export is usually a few large archives, and parallel uploads
    // just split the same bandwidth while multiplying the worker's memory peak.
    for (let i = 0; i < list.length; i++) {
      if (list[i].stage === "waiting" || list[i].stage === "failed") await runOne(i, list[i].file);
    }
    setRunning(false);
  }

  async function cluster() {
    setClustering(true);
    setError(null);
    try {
      const clusters = await recomputeClusters();
      const n = clusters.reduce((a, c) => a + c.episode_count, 0);
      setClusterNote(
        clusters.length
          ? `${clusters.length} failure mode${clusters.length === 1 ? "" : "s"} covering ${n} failed episodes.`
          : "No recurring failure modes yet — upload more failed episodes.",
      );
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
        title="Upload data"
        subtitle="Add robot episodes. Failed episodes are analysed automatically once uploaded."
      />

      <ErrorNote error={error} />

      <section className="space-y-5">
        <SectionHead left="your data" title="Upload your own files" />
        <label className="flex flex-wrap items-center gap-3 text-body text-slate-smoke">
          Results in these files
          <select
            value={outcome}
            onChange={(e) => setOutcome(e.target.value as Outcome)}
            disabled={running}
            className="field w-auto"
            aria-label="How to label the result of each uploaded episode"
          >
            <option value="auto">Detect from the data (recommended)</option>
            <option value="fail">All are failed runs</option>
            <option value="success">All are successful runs</option>
          </select>
          <span className="text-caption">
            Use this when your logs don&rsquo;t record whether each run succeeded.
          </span>
        </label>
        <div
          className={`card flex flex-col items-center gap-3 p-10 text-center transition ${
            dragging ? "bg-bone-white ring-2 ring-deep-fern" : ""
          }`}
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragging(false);
            choose(e.dataTransfer.files);
          }}
        >
          <p className="text-body text-slate-smoke">
            {running ? "Uploading — you can add more files when this finishes." : "Drop files here, or"}
          </p>
          <input
            ref={input}
            type="file"
            multiple
            className="sr-only"
            id="upload-input"
            onChange={(e) => choose(e.target.files)}
            disabled={running}
          />
          <label
            htmlFor="upload-input"
            className={`btn cursor-pointer ${running ? "pointer-events-none opacity-50" : ""}`}
          >
            Choose files…
          </label>
          <p className="muoto text-caption text-slate-smoke">{FORMATS}</p>
        </div>
      </section>

      {items.length > 0 && (
        <section ref={queueRef} className="scroll-mt-24 space-y-5">
          <SectionHead
            left="progress"
            right={`${items.length} file${items.length === 1 ? "" : "s"}`}
            title="Uploads"
            action={
              !running && items.some((it) => it.stage === "failed") ? (
                <ActionButton onClick={() => start()}>Retry failed</ActionButton>
              ) : undefined
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
                    {it.classified === 0 && <> · no failures to analyse</>}
                    {it.classified != null && it.classified > 0 && (
                      <>
                        {" · "}
                        {it.classified} failure{it.classified === 1 ? "" : "s"} analysed
                        {it.surfaces && Object.keys(it.surfaces).length > 0 && (
                          <>
                            {" ("}
                            {Object.entries(it.surfaces)
                              .map(([s, n]) => `${n} ${(SURFACE_LABEL[s] ?? s).toLowerCase()}`)
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
            Next: group the new failures into recurring failure modes.
            {clusterNote && <span className="ml-2 text-slate-smoke">{clusterNote}</span>}
          </p>
          <div className="ml-auto flex gap-2">
            <ActionButton onClick={cluster} busy={clustering}>
              Find failure modes
            </ActionButton>
            <Link href="/episodes" className="btn">
              View episodes
            </Link>
            {clusterNote && (
              <Link href="/clusters" className="btn">
                View failure modes
              </Link>
            )}


          </div>
        </section>
      )}

      <section id="samples" className="scroll-mt-24 space-y-5">
        <SectionHead left="sample data" right="real public datasets" title="Or try it with sample data" />
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-3">
          {SAMPLES.map((sm) => (
            <div key={sm.file} className="card flex flex-col gap-3 p-5">
              <div className="text-body-lg font-medium text-forest-ink">{sm.title}</div>
              <p className="flex-1 text-body text-slate-smoke">{sm.detail}</p>
              <div className="flex flex-wrap gap-2">
                <ActionButton onClick={() => trySample(sm.file)} disabled={running} variant="btn-filled">
                  Upload this sample
                </ActionButton>
                <a href={`/samples/${sm.file}`} download className="btn">
                  Download
                </a>
              </div>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
