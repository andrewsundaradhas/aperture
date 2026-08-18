"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { api, EpisodeSummary, SURFACES } from "@/lib/api";
import {
  Confidence,
  Empty,
  ErrorNote,
  OutcomePill,
  PageHeader,
  SurfacePill,
  TableSkeleton,
} from "@/components/ui";

const OUTCOMES = ["", "fail", "success"];

export default function EpisodesPage() {
  const router = useRouter();
  const [episodes, setEpisodes] = useState<EpisodeSummary[]>([]);
  // Facet options come from the unfiltered set. Deriving them from the filtered results
  // makes the robot dropdown collapse to the one robot you just picked, stranding you.
  const [allEpisodes, setAllEpisodes] = useState<EpisodeSummary[]>([]);
  const [surface, setSurface] = useState("");
  const [robot, setRobot] = useState("");
  const [outcome, setOutcome] = useState("");
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<unknown>(null);

  useEffect(() => {
    api.listEpisodes().then(setAllEpisodes).catch(() => {});
  }, []);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    api
      .listEpisodes({
        surface: surface || undefined,
        robot_id: robot || undefined,
        outcome: outcome || undefined,
      })
      .then((rows) => {
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

  const robots = useMemo(
    () => Array.from(new Set(allEpisodes.map((e) => e.robot_id))).sort(),
    [allEpisodes],
  );

  const visible = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return episodes;
    return episodes.filter(
      (e) =>
        (e.instruction ?? "").toLowerCase().includes(q) || e.id.toLowerCase().includes(q),
    );
  }, [episodes, query]);

  const filtered = !!(surface || robot || outcome || query);

  return (
    <div className="space-y-5">
      <PageHeader
        title="Episodes"
        subtitle="Every ingested episode, its failure surface and confidence."
      >
        <input
          type="search"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search instruction or id"
          aria-label="Search episodes by instruction or id"
          className="field w-52"
        />
        <select
          value={surface}
          onChange={(e) => setSurface(e.target.value)}
          aria-label="Filter by failure surface"
          className="field"
        >
          <option value="">all surfaces</option>
          {SURFACES.map((s) => (
            <option key={s} value={s}>
              {s}
            </option>
          ))}
        </select>
        <select
          value={outcome}
          onChange={(e) => setOutcome(e.target.value)}
          aria-label="Filter by outcome"
          className="field"
        >
          {OUTCOMES.map((o) => (
            <option key={o} value={o}>
              {o || "all outcomes"}
            </option>
          ))}
        </select>
        <select
          value={robot}
          onChange={(e) => setRobot(e.target.value)}
          aria-label="Filter by robot"
          className="field"
        >
          <option value="">all robots</option>
          {robots.map((r) => (
            <option key={r} value={r}>
              {r.slice(0, 8)}
            </option>
          ))}
        </select>
      </PageHeader>

      <ErrorNote error={error} />

      {loading ? (
        <TableSkeleton />
      ) : error ? null : visible.length === 0 ? (
        // Distinguish "nothing matches your filters" from "there is no data at all" —
        // telling someone to seed a database that already has rows sends them the wrong way.
        <Empty>
          {filtered ? (
            <>
              No episodes match these filters.{" "}
              <button
                onClick={() => {
                  setSurface("");
                  setRobot("");
                  setOutcome("");
                  setQuery("");
                }}
                className="text-signal hover:underline"
              >
                Clear all
              </button>
            </>
          ) : (
            <>
              No episodes yet. Seed the demo with{" "}
              <span className="font-mono text-charcoal">python -m aperture.seed</span>.
            </>
          )}
        </Empty>
      ) : (
        <>
          <div className="card overflow-hidden">
            <div className="table-scroll">
              <table className="data">
                <thead>
                  <tr>
                    <th>Episode</th>
                    <th>Instruction</th>
                    <th>Format</th>
                    <th>Outcome</th>
                    <th>Surface</th>
                    <th>Confidence</th>
                  </tr>
                </thead>
                <tbody>
                  {visible.map((e) => (
                    <tr
                      key={e.id}
                      // The row carried a pointer cursor while only the id cell navigated.
                      // Make the whole row the target it already looked like.
                      className="row-link cursor-pointer"
                      onClick={() => router.push(`/episodes/${e.id}`)}
                    >
                      <td>
                        <Link
                          href={`/episodes/${e.id}`}
                          onClick={(ev) => ev.stopPropagation()}
                          className="row-link-label font-mono text-signal"
                        >
                          {e.id.slice(0, 8)}
                        </Link>
                      </td>
                      <td className="max-w-xs truncate text-charcoal" title={e.instruction ?? ""}>
                        {e.instruction ?? "—"}
                      </td>
                      <td className="font-mono text-caption uppercase text-ash">
                        {e.source_format}
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
                        <Confidence value={e.classification_confidence} />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
          <p className="text-caption text-ash">
            {visible.length} of {allEpisodes.length} episode
            {allEpisodes.length === 1 ? "" : "s"}
          </p>
        </>
      )}
    </div>
  );
}
