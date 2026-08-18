import Link from "next/link";

export default function Home() {
  return (
    <div className="space-y-20">
      <section className="max-w-3xl space-y-6 pt-4">
        <span className="pill bg-linen text-ash border border-mist">
          Ingestion → Evaluation → Interpretability → Loop closure
        </span>
        <h1 className="serif text-heading-lg text-graphite">
          The evaluation &amp; interpretability layer for VLA robot policies
        </h1>
        <p className="text-subheading text-ash leading-relaxed max-w-2xl font-normal">
          Aperture classifies <em className="not-italic text-charcoal">why</em> a policy
          failed, attributes it with interpretability techniques, clusters similar
          failures across a fleet, exports a scoped fine-tune dataset, and verifies the
          fix worked after retraining.
        </p>
        <div className="flex items-center gap-3 pt-1">
          <Link href="/episodes" className="btn-accent">
            Browse episodes
            <span className="inline-flex h-4 w-4 items-center justify-center rounded-full border border-signal text-[10px] leading-none">
              →
            </span>
          </Link>
          <Link href="/clusters" className="btn">
            View clusters
          </Link>
        </div>
      </section>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <Link href="/episodes" className="card p-6 hover:border-signal/50 transition group">
          <div className="text-caption uppercase tracking-wide text-ash mb-2">
            Ingestion → Evaluation → Interpretability
          </div>
          <div className="serif text-heading-sm text-graphite group-hover:text-signal transition">
            Episodes
          </div>
          <p className="text-body-sm text-ash mt-2 leading-relaxed">
            Every ingested episode with its failure surface, confidence trace, attention
            map, and instruction-sensitivity probe.
          </p>
        </Link>
        <Link href="/clusters" className="card p-6 hover:border-signal/50 transition group">
          <div className="text-caption uppercase tracking-wide text-ash mb-2">
            Clustering → Export → Loop closure
          </div>
          <div className="serif text-heading-sm text-graphite group-hover:text-signal transition">
            Failure clusters
          </div>
          <p className="text-body-sm text-ash mt-2 leading-relaxed">
            Fleet-level failure patterns. Export a scoped dataset and verify the
            before/after success-rate delta after a retrain.
          </p>
        </Link>
      </div>

      <div className="text-caption text-ash font-mono">
        No data yet? Seed the demo:{" "}
        <span className="text-charcoal">cd apps/api &amp;&amp; python -m aperture.seed</span>
      </div>
    </div>
  );
}
