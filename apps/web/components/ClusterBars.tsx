"use client";

import Link from "next/link";
import { Cluster, SURFACE_BLURB, SURFACE_HEX, clusterTitle } from "@/lib/api";
import { Mark } from "@/components/ui";

/**
 * Recurring failure modes, largest first.
 *
 * Horizontal bars rather than the cards this used to be: the question a fleet owner asks here
 * is "which mode is biggest", and that is a magnitude comparison. Bars sorted descending answer
 * it in one glance; three equal-sized cards actively hide it. Long cluster labels also read far
 * better on a horizontal axis.
 */
export function ClusterBars({ clusters }: { clusters: Cluster[] }) {
  const ranked = [...clusters].sort((a, b) => b.episode_count - a.episode_count);
  const max = Math.max(...ranked.map((c) => c.episode_count), 1);

  return (
    <ul className="space-y-1">
      {ranked.map((c) => {
        const hex = c.dominant_surface ? SURFACE_HEX[c.dominant_surface] : "#c3b0cb";
        return (
          <li key={c.id}>
            <Link
              href={`/clusters/${c.id}`}
              className="group grid grid-cols-[auto_1fr_auto] items-center gap-4 rounded-tag px-2 py-2.5 transition hover:bg-bone-white"
            >
              <span className="flex w-52 min-w-0 items-center gap-2">
                <Mark surface={c.dominant_surface} size={10} />
                <span className="truncate text-body text-forest-ink group-hover:text-deep-fern group-hover:underline">
                  {clusterTitle(c)}
                </span>
              </span>

              <span className="flex items-center gap-3">
                {/* 4px rounded data-end, anchored to the baseline at left. */}
                <span
                  className="h-2.5 rounded-tag"
                  style={{
                    width: `${Math.max((c.episode_count / max) * 100, 2)}%`,
                    background: hex,
                  }}
                />
                <span className="hidden truncate text-caption text-slate-smoke lg:block">
                  {c.dominant_surface ? SURFACE_BLURB[c.dominant_surface] : ""}
                </span>
              </span>

              <span className="text-body-lg font-medium tabular-nums text-forest-ink">
                {c.episode_count}
              </span>
            </Link>
          </li>
        );
      })}
    </ul>
  );
}
