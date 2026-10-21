"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import {
  api,
  Cluster,
  EpisodeSummary,
  SURFACES,
  fetchAllEpisodes,
  surfaceVerdict,
} from "@/lib/api";
import {
  ErrorNote,
  OutcomePill,
  PageHeader,
  SectionHead,
  Skeleton,
  StatTile,
  SurfacePill,
} from "@/components/ui";
import { SurfaceMix } from "@/components/SurfaceMix";
import { ClusterBars } from "@/components/ClusterBars";

export default function Overview() {
  const router = useRouter();
  const [episodes, setEpisodes] = useState<EpisodeSummary[] | null>(null);
  const [clusters, setClusters] = useState<Cluster[] | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [truncated, setTruncated] = useState(false);

  useEffect(() => {
    // The whole fleet, not page one — every number here is a fleet-wide total.
    Promise.all([fetchAllEpisodes(), api.listClusters()])
      .then(([e, c]) => {
        setEpisodes(e.episodes);
        setTruncated(!e.complete);
        setClusters(c);
      })
      .catch(setError);
  }, []);

  const stats = useMemo(() => {
    if (!episodes) return null;
    const failures = episodes.filter((e) => e.outcome !== "success");
    const explained = failures.filter(
      (e) => surfaceVerdict(e.surface, e.classification_confidence, e.outcome).kind === "surface",
    );
    const counts = Object.fromEntries(
      SURFACES.map((s) => [s, explained.filter((e) => e.surface === s).length]),
    ) as Record<string, number>;
    const analysed = failures.filter((e) => e.surface);
    counts.unknown = analysed.length - explained.length;
    const recent = [...failures]
      .sort((a, b) => +new Date(b.started_at) - +new Date(a.started_at))
      .slice(0, 6);
    return {
      total: episodes.length,
      failures: failures.length,
      explained: explained.length,
      analysed: analysed.length,
      pending: failures.length - analysed.length,
      counts,
      recent,
    };
  }, [episodes]);

  const loading = !error && (!episodes || !clusters);
  const empty = !loading && !error && stats?.total === 0;

  return (
    <div className="space-y-10">
      <PageHeader title="Overview" subtitle="Failures across your fleet, and what is causing them.">
        <Link href="/upload" className="btn-filled">
          Upload data
        </Link>
      </PageHeader>

      <ErrorNote error={error} />

      {empty && <GetStarted />}

      {loading && (
        <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
          {Array.from({ length: 4 }).map((_, i) => (
            <div key={i} className="card space-y-3 p-5">
              <Skeleton className="h-3 w-20" />
              <Skeleton className="h-8 w-14" />
            </div>
          ))}
        </div>
      )}

      {stats && stats.total > 0 && (
        <>
          <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
            <StatTile
              label="Episodes"
              value={stats.total}
              note={truncated ? "most recent shown" : "uploaded"}
            />
            <StatTile
              label="Failures"
              value={stats.failures}
              note={`${Math.round((stats.failures / stats.total) * 100)}% of episodes`}
            />
            <StatTile
              label="Cause identified"
              value={stats.explained}
              note={
                stats.pending > 0
                  ? `${stats.pending} still being analysed`
                  : `of ${stats.failures} failures`
              }
            />
            <StatTile
              label="Failure modes"
              value={clusters?.length ?? 0}
              note="recurring patterns"
            />
          </div>

          {stats.failures > 0 && (
            <section className="space-y-5">
              <SectionHead left="failure causes" title="What is causing failures" />
              <div className="card p-6">
                <SurfaceMix counts={stats.counts} total={stats.analysed} />
              </div>
            </section>
          )}

          <section className="space-y-5">
            <SectionHead
              left="failure modes"
              right="largest first"
              title="Recurring failure modes"
              action={
                <Link href="/clusters" className="link text-body">
                  View all
                </Link>
              }
            />
            {(clusters?.length ?? 0) > 0 ? (
              <div className="card p-3">
                <ClusterBars clusters={clusters!.slice(0, 5)} />
              </div>
            ) : (
              <div className="card p-6 text-body text-slate-smoke">
                No recurring patterns found yet. Failures are grouped once several of them look
                alike — open{" "}
                <Link href="/clusters" className="link">
                  Failure modes
                </Link>{" "}
                and press <span className="text-forest-ink">Find failure modes</span> after
                uploading more data.
              </div>
            )}
          </section>

          {stats.recent.length > 0 && (
            <section className="space-y-5">
              <SectionHead
                left="latest"
                title="Recent failures"
                action={
                  <Link href="/episodes?outcome=fail" className="link text-body">
                    View all
                  </Link>
                }
              />
              <div className="card overflow-hidden">
                <div className="table-scroll">
                  <table className="data">
                    <thead>
                      <tr>
                        <th>Task</th>
                        <th>Result</th>
                        <th>Failure cause</th>
                        <th>Recorded</th>
                      </tr>
                    </thead>
                    <tbody>
                      {stats.recent.map((e) => (
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
                            <SurfacePill
                              surface={e.surface}
                              confidence={e.classification_confidence}
                              outcome={e.outcome}
                            />
                          </td>
                          <td className="whitespace-nowrap text-slate-smoke">
                            {new Date(e.started_at).toLocaleString()}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </section>
          )}
        </>
      )}
    </div>
  );
}

function GetStarted() {
  const steps = [
    ["Upload", "Add robot episodes — LeRobot v3 datasets or RLDS / Open X-Embodiment files."],
    ["Analyse", "Every failed episode is checked automatically to find what caused it."],
    ["Act", "Review recurring failure modes, export them as a training set, verify the fix."],
  ];
  return (
    <section className="card space-y-6 p-8">
      <div className="space-y-2">
        <h2 className="text-heading text-lavender">Get started</h2>
        <p className="max-w-2xl text-body text-slate-smoke">
          No episodes yet. Upload your own data, or try Aperture in one click with a sample
          dataset of real robot episodes.
        </p>
      </div>
      <ol className="grid grid-cols-1 gap-4 md:grid-cols-3">
        {steps.map(([title, body], i) => (
          <li key={title} className="space-y-1.5 rounded-tag bg-bone-white p-4 shadow-hairline">
            <div className="muoto text-caption text-slate-smoke">step {i + 1}</div>
            <div className="text-body-lg font-medium text-forest-ink">{title}</div>
            <p className="text-body text-slate-smoke">{body}</p>
          </li>
        ))}
      </ol>
      <div className="flex flex-wrap gap-3">
        <Link href="/upload" className="btn-filled">
          Upload data
        </Link>
        <Link href="/upload#samples" className="btn">
          Try a sample dataset
        </Link>
      </div>
    </section>
  );
}
