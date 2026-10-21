"use client";

import { Suspense, useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { EpisodeSummary, Robot, SURFACES, SURFACE_LABEL, api, fetchAllEpisodes, robotName } from "@/lib/api";
import {
  Confidence,
  Empty,
  ErrorNote,
  OutcomePill,
  PageHeader,
  SurfacePill,
  TableSkeleton,
} from "@/components/ui";

const CAUSES = [...SURFACES, "unknown"];

function EpisodesView() {
  const router = useRouter();
  const params = useSearchParams();
  const [episodes, setEpisodes] = useState<EpisodeSummary[]>([]);
  const [total, setTotal] = useState(0);
  const [robots, setRobots] = useState<Robot[]>([]);
  const [surface, setSurface] = useState(params.get("cause") ?? "");
  const [robot, setRobot] = useState(params.get("robot") ?? "");
  const [outcome, setOutcome] = useState(params.get("outcome") ?? "");
  const [query, setQuery] = useState("");
  // Fleets run to tens of thousands of episodes; render a page at a time.
  const [shown, setShown] = useState(100);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<unknown>(null);

  useEffect(() => {
    api.listRobots().then(setRobots).catch(() => {});
    fetchAllEpisodes()
      .then((r) => setTotal(r.episodes.length))
      .catch(() => {});
  }, []);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    fetchAllEpisodes({
      surface: surface || undefined,
      robot_id: robot || undefined,
      outcome: outcome || undefined,
    })
      .then(({ episodes: rows }) => {
        if (cancelled) return;
        setEpisodes(rows);
        setError(null);
      })
      .catch((e) => !cancelled && setError(e))
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
  }, [surface, robot, outcome]);

  useEffect(() => setShown(100), [surface, robot, outcome, query]);

  const robotsById = useMemo(() => new Map(robots.map((r) => [r.id, r])), [robots]);

  const visible = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return episodes;
    return episodes.filter((e) => (e.instruction ?? "").toLowerCase().includes(q));
  }, [episodes, query]);

  const filtered = !!(surface || robot || outcome || query);

  function clearAll() {
    setSurface("");
    setRobot("");
    setOutcome("");
    setQuery("");
  }

  return (
    <div className="space-y-6">
      <PageHeader title="Episodes" subtitle="Every uploaded episode, its result and its failure cause.">
        <Link href="/upload" className="btn">
          Upload data
        </Link>
      </PageHeader>

      <div className="flex flex-wrap items-center gap-2">
        <input
          type="search"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search tasks"
          aria-label="Search episodes by task"
          className="field w-56"
        />
        <select
          value={outcome}
          onChange={(e) => setOutcome(e.target.value)}
          aria-label="Filter by result"
          className="field w-auto"
        >
          <option value="">All results</option>
          <option value="fail">Failed</option>
          <option value="success">Succeeded</option>
        </select>
        <select
          value={surface}
          onChange={(e) => setSurface(e.target.value)}
          aria-label="Filter by failure cause"
          className="field w-auto"
        >
          <option value="">All causes</option>
          {CAUSES.map((s) => (
            <option key={s} value={s}>
              {SURFACE_LABEL[s]}
            </option>
          ))}
        </select>
        {robots.length > 1 && (
          <select
            value={robot}
            onChange={(e) => setRobot(e.target.value)}
            aria-label="Filter by robot"
            className="field w-auto"
          >
            <option value="">All robots</option>
            {robots.map((r) => (
              <option key={r.id} value={r.id}>
                {robotName(r)}
              </option>
            ))}
          </select>
        )}
        {filtered && (
          <button onClick={clearAll} className="btn-ghost">
            Clear filters
          </button>
        )}
        <span className="muoto ml-auto text-caption tabular-nums text-slate-smoke">
          {visible.length} of {total} episodes
        </span>
      </div>

      <ErrorNote error={error} />

      {loading ? (
        <TableSkeleton />
      ) : error ? null : visible.length === 0 ? (
        <Empty>
          {filtered ? (
            <>
              No episodes match these filters.{" "}
              <button onClick={clearAll} className="link">
                Clear filters
              </button>
            </>
          ) : (
            <>
              No episodes yet.{" "}
              <Link href="/upload" className="link">
                Upload data
              </Link>{" "}
              or try a sample dataset to get started.
            </>
          )}
        </Empty>
      ) : (
        <div className="card overflow-hidden">
          <div className="table-scroll">
            <table className="data">
              <thead>
                <tr>
                  <th>Task</th>
                  <th>Robot</th>
                  <th>Result</th>
                  <th>Failure cause</th>
                  <th>Confidence</th>
                  <th>Recorded</th>
                </tr>
              </thead>
              <tbody>
                {visible.slice(0, shown).map((e) => (
                  <tr
                    key={e.id}
                    className="row-link cursor-pointer"
                    onClick={() => router.push(`/episodes/${e.id}`)}
                  >
                    <td className="max-w-xs truncate" title={e.instruction ?? ""}>
                      <Link
                        href={`/episodes/${e.id}`}
                        onClick={(ev) => ev.stopPropagation()}
                        className="row-link-label text-forest-ink"
                      >
                        {e.instruction ?? "Untitled episode"}
                      </Link>
                    </td>
                    <td className="whitespace-nowrap text-slate-smoke">
                      {robotName(robotsById.get(e.robot_id))}
                    </td>
                    <td>
                      <OutcomePill outcome={e.outcome} />
                    </td>
                    <td>
                      <SurfacePill
                        surface={e.surface}
                        confidence={e.classification_confidence}
                        outcome={e.outcome}
                      />
                    </td>
                    <td>
                      {e.outcome === "success" ? (
                        <span className="text-slate-smoke">—</span>
                      ) : (
                        <Confidence value={e.surface === "unknown" ? null : e.classification_confidence} />
                      )}
                    </td>
                    <td className="whitespace-nowrap text-slate-smoke">
                      {new Date(e.started_at).toLocaleDateString()}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {visible.length > shown && (
            <div className="flex items-center justify-between gap-3 px-4 py-3 text-caption text-slate-smoke">
              <span>
                Showing {shown} of {visible.length}
              </span>
              <button className="btn py-1.5" onClick={() => setShown((n) => n + 200)}>
                Show more
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default function EpisodesPage() {
  // useSearchParams needs a Suspense boundary for this route to render statically.
  return (
    <Suspense fallback={<TableSkeleton />}>
      <EpisodesView />
    </Suspense>
  );
}
