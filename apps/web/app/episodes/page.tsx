"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { api, EpisodeSummary } from "@/lib/api";
import { Confidence, Empty, ErrorNote, OutcomePill, SurfacePill, Spinner } from "@/components/ui";

const SURFACES = ["", "perception", "grounding", "motor"];

export default function EpisodesPage() {
  const [episodes, setEpisodes] = useState<EpisodeSummary[]>([]);
  const [surface, setSurface] = useState("");
  const [robot, setRobot] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<unknown>(null);

  useEffect(() => {
    setLoading(true);
    api
      .listEpisodes({ surface: surface || undefined, robot_id: robot || undefined })
      .then(setEpisodes)
      .catch(setError)
      .finally(() => setLoading(false));
  }, [surface, robot]);

  const robots = Array.from(new Set(episodes.map((e) => e.robot_id)));

  return (
    <div className="space-y-5">
      <div className="flex items-end justify-between">
        <div>
          <h1 className="serif text-heading text-graphite">Episodes</h1>
          <p className="text-body-sm text-ash mt-1">
            Every ingested episode, its failure surface and confidence.
          </p>
        </div>
        <div className="flex gap-2">
          <select
            value={surface}
            onChange={(e) => setSurface(e.target.value)}
            className="field"
          >
            {SURFACES.map((s) => (
              <option key={s} value={s}>{s || "all surfaces"}</option>
            ))}
          </select>
          <select
            value={robot}
            onChange={(e) => setRobot(e.target.value)}
            className="field"
          >
            <option value="">all robots</option>
            {robots.map((r) => (
              <option key={r} value={r}>{r.slice(0, 8)}</option>
            ))}
          </select>
        </div>
      </div>

      <ErrorNote error={error} />

      {loading ? (
        <div className="card p-8 flex justify-center"><Spinner /></div>
      ) : episodes.length === 0 ? (
        <Empty>No episodes. Seed the demo with <span className="font-mono text-charcoal">python -m aperture.seed</span>.</Empty>
      ) : (
        <div className="card overflow-hidden">
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
              {episodes.map((e) => (
                <tr key={e.id} className="cursor-pointer">
                  <td>
                    <Link href={`/episodes/${e.id}`} className="font-mono text-signal hover:underline">
                      {e.id.slice(0, 8)}
                    </Link>
                  </td>
                  <td className="max-w-xs truncate text-charcoal">{e.instruction ?? "—"}</td>
                  <td className="font-mono text-caption text-ash uppercase">{e.source_format}</td>
                  <td><OutcomePill outcome={e.outcome} /></td>
                  <td><SurfacePill surface={e.surface} /></td>
                  <td><Confidence value={e.classification_confidence} /></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
