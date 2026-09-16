"use client";

/**
 * The four layers as a plotted sequence rather than a process diagram — connected nodes on a
 * hairline path, which is the system's timeline convention. Each node carries the live count
 * of what has actually passed through that stage, so the diagram is a reading of the fleet
 * rather than an illustration of the architecture.
 */
export function PipelineTrack({
  stages,
}: {
  stages: { label: string; value: string; note: string }[];
}) {
  return (
    <ol className="relative grid grid-cols-2 gap-y-8 lg:grid-cols-4">
      {/* The connecting path, behind the nodes. */}
      <span
        className="absolute left-[12.5%] right-[12.5%] top-[7px] hidden h-px bg-forest-ink lg:block"
        aria-hidden
      />
      {stages.map((s, i) => (
        <li key={s.label} className="relative flex flex-col gap-2 lg:items-center lg:text-center">
          <span
            className="relative z-10 h-[15px] w-[15px] rounded-full bg-bone-white"
            style={{ boxShadow: "0 0 0 1px #09352e" }}
            aria-hidden
          >
            {i === 0 && (
              <span className="absolute inset-[4px] rounded-full bg-moss" aria-hidden />
            )}
          </span>
          <span className="cinetype text-[11px] text-slate-smoke">{s.label}</span>
          <span className="text-heading-sm font-medium tabular-nums text-forest-ink">
            {s.value}
          </span>
          <span className="muoto text-caption text-slate-smoke">{s.note}</span>
        </li>
      ))}
    </ol>
  );
}
