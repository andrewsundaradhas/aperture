"use client";

import { SURFACES, SURFACE_BLURB, SURFACE_HEX, SURFACE_LABEL } from "@/lib/api";
import { Mark } from "@/components/ui";

/**
 * Where the fleet's attributed failures land — one bar, three segments.
 *
 * A stacked bar is the right form here because the parts are a whole: every attributed
 * failure is exactly one surface, and the question is the mix, not the trend. Segments are
 * separated by a 2px canvas-coloured gap so adjacent greens never touch, and each carries a
 * direct label underneath — the lightest step is under 3:1 on this canvas, so the label is
 * what makes it readable, not the hue.
 */
export function SurfaceMix({
  counts,
  total,
}: {
  counts: Record<string, number>;
  total: number;
}) {
  // "Undetermined" is shown when present: hiding it would make the explained share look larger
  // than it is.
  const keys = [...SURFACES, ...(counts.unknown ? ["unknown"] : [])];
  if (!total) {
    return <p className="text-body text-slate-smoke">No failures analysed yet.</p>;
  }

  return (
    <div className="space-y-5">
      <div className="flex h-3 w-full gap-[2px]" role="img" aria-label="Failure surface mix">
        {keys.map((s) =>
          counts[s] ? (
            <div
              key={s}
              className="h-full rounded-tag"
              style={{
                width: `${(counts[s] / total) * 100}%`,
                background: SURFACE_HEX[s],
              }}
              title={`${SURFACE_LABEL[s]}: ${counts[s]}`}
            />
          ) : null,
        )}
      </div>

      <ul className={`grid grid-cols-1 gap-x-8 gap-y-4 ${keys.length > 3 ? "sm:grid-cols-2 lg:grid-cols-4" : "sm:grid-cols-3"}`}>
        {keys.map((s) => (
          <li key={s} className="space-y-1.5">
            <div className="flex items-baseline gap-2">
              <Mark surface={s} size={10} className="translate-y-px" />
              <span className="text-body text-forest-ink">{SURFACE_LABEL[s]}</span>
              <span className="ml-auto text-body-lg font-medium tabular-nums text-forest-ink">
                {counts[s] ?? 0}
              </span>
              <span className="muoto text-caption tabular-nums text-slate-smoke">
                {Math.round(((counts[s] ?? 0) / total) * 100)}%
              </span>
            </div>
            <div className="h-px w-full bg-lichen" aria-hidden />
            <p className="text-caption leading-relaxed text-slate-smoke">{SURFACE_BLURB[s]}</p>
          </li>
        ))}
      </ul>
    </div>
  );
}
