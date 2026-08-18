import type { Metadata } from "next";
import Link from "next/link";
import { Fraunces, Inter } from "next/font/google";
import "./globals.css";

// Substitutes for the brand's custom faces: Fraunces stands in for the ppmondwest
// display serif, Inter for the af UI sans.
const ppmondwest = Fraunces({
  subsets: ["latin"],
  weight: ["400", "500"],
  variable: "--font-ppmondwest",
  display: "swap",
});
const af = Inter({
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
  variable: "--font-af",
  display: "swap",
});

export const metadata: Metadata = {
  title: "Aperture — VLA failure evaluation & interpretability",
  description: "Classify, attribute, cluster, and verify robot policy failures.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${ppmondwest.variable} ${af.variable}`}>
      <body>
        <div className="min-h-screen">
          {/* Floating frosted navigation pill */}
          <header className="sticky top-4 z-20 px-4">
            <nav className="mx-auto max-w-[1200px]">
              <div className="mx-auto flex w-fit items-center gap-5 rounded-nav border border-mist bg-paper/70 px-3 py-2 shadow-nav backdrop-blur-md">
                <Link href="/" className="flex items-center gap-2 pl-1 pr-1">
                  <SunGlyph />
                  <span className="serif text-subheading leading-none text-graphite">
                    Aperture
                  </span>
                </Link>
                <span className="h-4 w-px bg-mist" aria-hidden />
                <div className="flex items-center gap-4 text-body-sm font-medium text-charcoal">
                  <Link href="/episodes" className="hover:text-graphite transition">
                    Episodes
                  </Link>
                  <Link href="/clusters" className="hover:text-graphite transition">
                    Clusters
                  </Link>
                </div>
                <Link
                  href="/episodes"
                  className="flex items-center gap-1.5 rounded-lg border border-signal px-3 py-1 text-body-sm font-medium text-signal transition hover:bg-signal/5"
                >
                  Open console
                  <ArrowCircle />
                </Link>
              </div>
            </nav>
          </header>

          <main className="mx-auto max-w-[1200px] px-6 py-16">{children}</main>

          <footer className="mt-24 border-t border-mist bg-paper">
            <div className="mx-auto max-w-[1200px] px-6 py-16">
              <p className="serif text-heading-sm text-graphite max-w-2xl">
                The evaluation &amp; interpretability layer for Vision-Language-Action
                robot policies.
              </p>
              <div className="mt-8 flex flex-wrap items-center gap-6 text-body-sm text-ash">
                <Link href="/episodes" className="hover:text-charcoal transition">
                  Episodes
                </Link>
                <Link href="/clusters" className="hover:text-charcoal transition">
                  Clusters
                </Link>
                <span className="ml-auto text-caption text-fog">
                  Aperture · evaluation · interpretability layer
                </span>
              </div>
            </div>
          </footer>
        </div>
      </body>
    </html>
  );
}

// Minimal sun-over-landscape glyph, echoing the brand's nav icon.
function SunGlyph() {
  return (
    <svg
      width="18"
      height="18"
      viewBox="0 0 24 24"
      fill="none"
      stroke="#282834"
      strokeWidth="1.5"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden
    >
      <circle cx="12" cy="10" r="3.5" />
      <path d="M2 19h20" />
      <path d="M5 19l3.5-4 3 2.5L16 12l3 4" />
    </svg>
  );
}

function ArrowCircle() {
  return (
    <span className="inline-flex h-4 w-4 items-center justify-center rounded-full border border-signal text-[10px] leading-none">
      →
    </span>
  );
}
