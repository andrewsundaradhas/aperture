// Thin typed client for the Aperture API.
//
// Every call goes to this app's own `/api/aperture/*` route, which attaches the API key
// server-side (see `app/api/aperture/[...path]/route.ts`). The browser never sees a credential,
// so there is deliberately no key in this file and no `NEXT_PUBLIC_*` variable holding one.
// Point the proxy at a different backend with the server-side `APERTURE_API_BASE_URL`.

const BASE = "/api/aperture";

export const API_BASE = BASE;

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    ...init,
    headers: { ...(init?.headers || {}) },
    cache: "no-store",
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`${res.status} ${res.statusText}: ${text}`);
  }
  return res.json() as Promise<T>;
}

/** Resolve a storage URI into something the browser can actually fetch.
 *
 * The storage layer hands back backend-native URIs — `local://<key>` from the filesystem
 * backend, `r2://<bucket>/<key>` from Cloudflare R2 — neither of which is a URL. Both are
 * served by the API's `/v1/blobs/<key>` route, reached here through the same server-side proxy
 * as every other call, so the blob request carries no credential either.
 * Only an http(s) URI (a real R2 presigned URL) is used verbatim.
 */
export function blobUrl(uri: string): string {
  if (/^https?:\/\//.test(uri)) return uri;
  const key = uri
    .replace(/^local:\/\//, "")
    .replace(/^r2:\/\/[^/]+\//, "")
    .replace(/^\/?v1\/blobs\//, "")
    .replace(/^\//, "");
  return `${BASE}/v1/blobs/${key}`;
}

export type EpisodeSummary = {
  id: string;
  robot_id: string;
  started_at: string;
  outcome: string;
  source_format: string;
  instruction: string | null;
  surface: string | null;
  classification_confidence: number | null;
};

export type Frame = {
  t: number;
  action_confidence: number | null;
  contact_force: number | null;
  subgoal: string | null;
};

/** Per-heuristic evidence behind a verdict. `_margin` is the gap to the runner-up. */
export type Classification = {
  surface: string;
  confidence: number;
  method: string;
  details: Record<string, any>;
};

export type Episode = {
  id: string;
  robot_id: string;
  org_id: string;
  started_at: string;
  outcome: string;
  source_format: string;
  instruction: string | null;
  rlds_uri: string | null;
  frames: Frame[];
  classification: Classification | null;
};

export type Attribution = {
  episode_id: string;
  job_status: string;
  attention_map_uri: string | null;
  confidence_trace: { t: number; action_confidence: number | null; contact_force: number | null }[];
  counterfactual_result: any | null;
};

export type Cluster = {
  id: string;
  label: string;
  dominant_surface: string | null;
  episode_count: number;
  representative_episode_id?: string | null;
};

export type ClusterDetail = Cluster & {
  episodes: { id: string; robot_id: string; outcome: string; surface: string | null; instruction: string | null }[];
};

/** A queued unit of background work. Clustering and dataset ingestion are too slow to hold a
 *  request open, so both answer with one of these and the client polls. */
export type Job = {
  id: string;
  kind: string;
  status: "queued" | "running" | "done" | "failed";
  result: Record<string, any> | null;
  error: string | null;
  attempts: number;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
};

export type JobAccepted = { job_id: string; status: string; poll: string };

/** How to label the success/failure of uploaded episodes: from the data, or all one way. */
export type Outcome = "auto" | "success" | "fail";

export type Org = { org_id: string; slug: string; name: string; tier: string };
export type ApiKey = {
  id: string;
  prefix: string;
  name: string;
  created_at: string;
  last_used_at: string | null;
  revoked_at: string | null;
};
export type Robot = { id: string; embodiment_type: string; policy_name: string };

/** PUT a file's bytes to a presigned upload URL.
 *
 * On R2/S3 that URL is the bucket itself (the API never sees the bytes; the bucket needs a CORS
 * rule allowing PUT from the dashboard origin). On the local filesystem backend it is the API's
 * own `/v1/blobs/...` route, which is reached through this app's proxy like every other call.
 * XHR rather than fetch so the caller gets upload progress.
 */
function uploadErrorText(status: number, body: string): string {
  if (status === 413) return "the file is larger than this workspace allows.";
  if (status === 401 || status === 403) return "you are not signed in, or this upload link has expired.";
  try {
    const detail = JSON.parse(body)?.detail;
    if (typeof detail === "string") return detail;
  } catch {
    /* not JSON */
  }
  return `the server answered ${status}.`;
}

export function putUpload(uploadUrl: string, file: File, onProgress?: (fraction: number) => void): Promise<void> {
  const url = /^https?:\/\//.test(uploadUrl) ? uploadUrl : `${BASE}${uploadUrl.startsWith("/") ? "" : "/"}${uploadUrl}`;
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("PUT", url);
    xhr.upload.onprogress = (e) => e.lengthComputable && onProgress?.(e.loaded / e.total);
    xhr.onload = () =>
      xhr.status >= 200 && xhr.status < 300
        ? resolve()
        : reject(new Error(`Upload failed: ${uploadErrorText(xhr.status, xhr.responseText)}`));
    xhr.onerror = () =>
      reject(new Error("Upload failed: network error (on R2/S3, check the bucket's CORS allows PUT from this origin)."));
    xhr.send(file);
  });
}

export class JobFailedError extends Error {
  constructor(public readonly job: Job) {
    super(job.error || `Job ${job.kind} failed`);
    this.name = "JobFailedError";
  }
}

export type VerifyResult = {
  cluster_id: string;
  verification_run_id: string;
  pre_success_rate: number;
  post_success_rate: number;
  delta: number;
  pre_n: number;
  post_n: number;
};

export const api = {
  listEpisodes: (
    q: {
      robot_id?: string;
      surface?: string;
      outcome?: string;
      limit?: number;
      offset?: number;
    } = {},
  ) => {
    const p = new URLSearchParams(
      Object.entries(q)
        .filter(([, v]) => v !== undefined && v !== "" && v !== null)
        .map(([k, v]) => [k, String(v)]),
    );
    return req<EpisodeSummary[]>(`/v1/episodes${p.toString() ? `?${p}` : ""}`);
  },
  getEpisode: (id: string) => req<Episode>(`/v1/episodes/${id}`),
  classify: (id: string) => req(`/v1/episodes/${id}/classify`, { method: "POST" }),
  getAttribution: (id: string) => req<Attribution>(`/v1/episodes/${id}/attribution`),
  runAttribution: (id: string) => req<Attribution>(`/v1/episodes/${id}/attribution`, { method: "POST" }),
  listClusters: () => req<Cluster[]>(`/v1/clusters`),
  startRecompute: () => req<JobAccepted>(`/v1/clusters/recompute`, { method: "POST" }),
  getJob: (id: string) => req<Job>(`/v1/jobs/${id}`),
  getCluster: (id: string) => req<ClusterDetail>(`/v1/clusters/${id}`),
  exportDataset: (id: string, format: string) =>
    req<{ download_url: string; episode_count: number; format: string }>(
      `/v1/clusters/${id}/dataset-export`,
      { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ format }) }
    ),
  classifyPending: (reclassify = false) =>
    req<JobAccepted>(`/v1/episodes/classify-pending${reclassify ? "?reclassify=true" : ""}`, { method: "POST" }),

  // ── uploads: presign → PUT bytes → queue ingest ──
  presign: (filename: string) =>
    req<{ key: string; upload_url: string; ingest_url: string }>(`/v1/uploads/presign`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ filename }),
    }),
  ingest: (key: string, filename?: string, outcome: Outcome = "auto") =>
    req<JobAccepted>(`/v1/uploads/ingest`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ key, filename, outcome }),
    }),

  // ── organization settings ──
  me: () => req<Org>(`/v1/onboarding/me`),
  listKeys: () => req<ApiKey[]>(`/v1/onboarding/keys`),
  createKey: (name: string) =>
    req<{ api_key: string }>(`/v1/onboarding/keys`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    }),
  revokeKey: async (id: string) => {
    const res = await fetch(`${BASE}/v1/onboarding/keys/${id}`, { method: "DELETE", cache: "no-store" });
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}: ${await res.text().catch(() => "")}`);
  },
  listRobots: () => req<Robot[]>(`/v1/onboarding/robots?limit=500`),
  registerRobot: (embodiment_type: string, policy_name: string) =>
    req<Robot>(`/v1/onboarding/robots`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ embodiment_type, policy_name }),
    }),

  verify: (id: string, files: File[]) => {
    const fd = new FormData();
    files.forEach((f) => fd.append("files", f));
    return req<VerifyResult>(`/v1/clusters/${id}/verify`, { method: "POST", body: fd });
  },
};

/** Poll a job until it finishes.
 *
 * Throws `JobFailedError` when the job fails, so a caller's existing catch renders the real
 * reason rather than a generic failure. Gives up after `timeoutMs` so a worker that is down
 * surfaces as an error instead of a spinner that never stops.
 */
export async function waitForJob(
  jobId: string,
  { intervalMs = 700, timeoutMs = 120_000 }: { intervalMs?: number; timeoutMs?: number } = {},
): Promise<Job> {
  const deadline = Date.now() + timeoutMs;
  for (;;) {
    const job = await api.getJob(jobId);
    if (job.status === "done") return job;
    if (job.status === "failed") throw new JobFailedError(job);
    if (Date.now() > deadline) {
      throw new Error(
        `Still ${job.status} after ${Math.round(timeoutMs / 1000)}s. Processing may be ` +
          `delayed — check back shortly, or contact your administrator if it persists.`,
      );
    }
    await new Promise((resolve) => setTimeout(resolve, intervalMs));
  }
}

/** Every episode matching a filter, not just the first page.
 *
 * `GET /v1/episodes` is paginated and defaults to 50 rows. Fleet-wide readings — the totals on
 * the overview, the scatter plot, the surface mix — describe the whole fleet, so reading one
 * page silently under-reports every one of them: a fleet of 162 episodes rendered as "50".
 *
 * Pages at the API's maximum and stops on the first short page. `cap` bounds the work for a
 * genuinely large fleet; `complete` reports whether the cap truncated the answer, so a caller
 * can say so rather than quietly presenting a partial fleet as the whole one.
 */
const PAGE_SIZE = 500;

export async function fetchAllEpisodes(
  q: { robot_id?: string; surface?: string; outcome?: string } = {},
  cap = 5000,
): Promise<{ episodes: EpisodeSummary[]; complete: boolean }> {
  const out: EpisodeSummary[] = [];
  for (let offset = 0; offset < cap; offset += PAGE_SIZE) {
    const page = await api.listEpisodes({ ...q, limit: PAGE_SIZE, offset });
    out.push(...page);
    if (page.length < PAGE_SIZE) return { episodes: out, complete: true };
  }
  return { episodes: out, complete: false };
}

/** Recluster the fleet and resolve once the new clusters are ready. */
export async function recomputeClusters(): Promise<Cluster[]> {
  const accepted = await api.startRecompute();
  await waitForJob(accepted.job_id);
  return api.listClusters();
}

// ── Failure-surface presentation ──────────────────────────────────────────────

export const SURFACES = ["perception", "grounding", "motor"] as const;

/** What each failure cause means, in words an operator can act on. */
export const SURFACE_LABEL: Record<string, string> = {
  perception: "Perception",
  grounding: "Instruction grounding",
  motor: "Motor / actuation",
  unknown: "Undetermined",
};

export const SURFACE_BLURB: Record<string, string> = {
  perception: "The policy lost confidence in what it was seeing — camera, lighting or occlusion.",
  grounding: "The policy kept re-planning — it never settled on what the instruction referred to.",
  motor: "Actuation went wrong — contact force or the commanded motion spiked or dropped out.",
  unknown: "The logs don't carry enough signal to tell which part of the system failed.",
};

export function surfaceLabel(surface: string | null | undefined): string {
  return surface ? SURFACE_LABEL[surface] ?? surface : "Mixed";
}

/** A robot as an operator knows it: its model, and the policy running on it when known. */
export function robotName(r: Robot | undefined): string {
  if (!r) return "—";
  return r.policy_name && r.policy_name !== "unknown"
    ? `${r.embodiment_type} · ${r.policy_name}`
    : r.embodiment_type;
}

/** A failure mode's display name: its cause plus a stable number, e.g. "Perception · mode 2". */
export function clusterTitle(c: Pick<Cluster, "label" | "dominant_surface">): string {
  const n = c.label.match(/cluster-(\d+)$/);
  return `${surfaceLabel(c.dominant_surface)} · mode ${n ? Number(n[1]) + 1 : "—"}`;
}

/** The plotted hex for each failure surface.
 *
 * Three steps of one green ramp — as far apart as a mono-green system allows. Validated
 * against the sage canvas for lightness, chroma, and both CVD and normal-vision separation;
 * the lightest step sits under 3:1, so every mark that uses these is paired with a label.
 * Components import these for SVG/canvas marks, where a Tailwind class cannot reach.
 */
export const SURFACE_HEX: Record<string, string> = {
  perception: "#e894ff",
  grounding: "#93ffe4",
  motor: "#ffac47",
  unknown: "#7d6a85",
};

export function surfaceColor(surface: string | null): string {
  switch (surface) {
    case "perception":
      return "border-lichen bg-bone-white text-forest-ink";
    case "grounding":
      return "border-lichen bg-bone-white text-forest-ink";
    case "motor":
      return "border-lichen bg-bone-white text-forest-ink";
    default:
      return "border border-lichen bg-bone-white text-slate-smoke";
  }
}

export type Verdict = {
  kind: "surface" | "none" | "inconclusive" | "unclassified";
  label: string;
  tone: string;
  hint: string;
};

/** What to actually show for an episode's failure surface.
 *
 * The classifier always names a surface, even when nothing fired — ties break toward motor
 * by design. Rendering that as a confident bucket is false certainty, so two cases get their
 * own label instead: a successful episode has no failure to attribute, and a verdict that
 * rounds to 0% is a tie-break rather than a finding.
 */
export function surfaceVerdict(
  surface: string | null,
  confidence: number | null,
  outcome?: string,
): Verdict {
  if (outcome === "success") {
    return {
      kind: "none",
      label: "No failure",
      tone: "border border-lichen bg-bone-white text-forest-ink",
      hint: "Episode succeeded — there is no failure surface to attribute.",
    };
  }
  if (surface === "unknown") {
    return {
      kind: "inconclusive",
      label: "Undetermined",
      tone: "border border-lichen bg-bone-white text-slate-smoke",
      hint: "Not enough signal in this episode's logs to attribute the failure to any surface.",
    };
  }
  if (!surface) {
    return {
      kind: "unclassified",
      label: "Not analysed",
      tone: "border border-lichen bg-bone-white text-slate-smoke",
      hint: "Not analysed yet. Analysis runs automatically after upload.",
    };
  }
  if (confidence != null && confidence < 0.005) {
    return {
      kind: "inconclusive",
      label: "Undetermined",
      tone: "border border-lichen bg-bone-white text-slate-smoke",
      hint: "No heuristic fired. The surface shown by the classifier is a tie-break, not a finding.",
    };
  }
  return {
    kind: "surface",
    label: surfaceLabel(surface),
    tone: surfaceColor(surface),
    hint: SURFACE_BLURB[surface] ?? "",
  };
}

/** Two heuristics within this margin means the verdict is a coin-flip worth flagging. */
export const AMBIGUOUS_MARGIN = 0.1;

export function isAmbiguous(details: Record<string, any> | undefined, confidence: number): boolean {
  if (!details || confidence < 0.005) return false;
  const margin = details["_margin"];
  return typeof margin === "number" && margin < AMBIGUOUS_MARGIN;
}
