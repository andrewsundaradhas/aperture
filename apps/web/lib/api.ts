// Thin typed client for the Aperture API. Reads base URL + API key from public env vars so
// the dashboard works locally (defaults) and against Render/Supabase in production.

const BASE = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";
const KEY = process.env.NEXT_PUBLIC_API_KEY || "demo-key";

export const API_BASE = BASE;

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    ...init,
    headers: { "X-API-Key": KEY, ...(init?.headers || {}) },
    cache: "no-store",
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`${res.status} ${res.statusText}: ${text}`);
  }
  return res.json() as Promise<T>;
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
  listEpisodes: (q: { robot_id?: string; surface?: string; outcome?: string } = {}) => {
    const p = new URLSearchParams(Object.entries(q).filter(([, v]) => v) as [string, string][]);
    return req<EpisodeSummary[]>(`/v1/episodes${p.toString() ? `?${p}` : ""}`);
  },
  getEpisode: (id: string) => req<Episode>(`/v1/episodes/${id}`),
  classify: (id: string) => req(`/v1/episodes/${id}/classify`, { method: "POST" }),
  getAttribution: (id: string) => req<Attribution>(`/v1/episodes/${id}/attribution`),
  runAttribution: (id: string) => req<Attribution>(`/v1/episodes/${id}/attribution`, { method: "POST" }),
  listClusters: () => req<Cluster[]>(`/v1/clusters`),
  recomputeClusters: () => req<Cluster[]>(`/v1/clusters/recompute`, { method: "POST" }),
  getCluster: (id: string) => req<ClusterDetail>(`/v1/clusters/${id}`),
  exportDataset: (id: string, format: string) =>
    req<{ download_url: string; episode_count: number; format: string }>(
      `/v1/clusters/${id}/dataset-export`,
      { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ format }) }
    ),
  verify: (id: string, files: File[]) => {
    const fd = new FormData();
    files.forEach((f) => fd.append("files", f));
    return req<VerifyResult>(`/v1/clusters/${id}/verify`, { method: "POST", body: fd });
  },
};

export function surfaceColor(surface: string | null): string {
  switch (surface) {
    case "perception":
      return "bg-perception/10 text-perception";
    case "grounding":
      return "bg-grounding/10 text-grounding";
    case "motor":
      return "bg-motor/10 text-motor";
    default:
      return "bg-linen text-ash border border-mist";
  }
}
