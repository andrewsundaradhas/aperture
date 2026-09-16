import type { Metadata } from "next";
import Link from "next/link";
import { Inter, IBM_Plex_Mono, JetBrains_Mono } from "next/font/google";
import "./globals.css";
import { NavLink } from "@/components/NavLink";
import { Chrome } from "@/components/Chrome";
import { SignOut } from "@/components/SignOut";
import { authConfig } from "@/lib/session";

// Substitutes for the Operate brand faces, per its own stated fallbacks:
//   denim    → Inter            (body, headings, UI)
//   muoto    → IBM Plex Mono    (labels, captions, axis ticks)
//   cinetype → JetBrains Mono   (ALL-CAPS specimen labels at wide tracking)
const denim = Inter({
  subsets: ["latin"],
  weight: ["400", "500"],
  variable: "--font-denim",
  display: "swap",
});
const muoto = IBM_Plex_Mono({
  subsets: ["latin"],
  weight: ["400", "500"],
  variable: "--font-muoto",
  display: "swap",
});
const cinetype = JetBrains_Mono({
  subsets: ["latin"],
  weight: ["400"],
  variable: "--font-cinetype",
  display: "swap",
});

export const metadata: Metadata = {
  title: "Aperture — VLA failure evaluation & interpretability",
  description: "Classify, attribute, cluster, and verify robot policy failures.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html
      lang="en"
      className={`${denim.variable} ${muoto.variable} ${cinetype.variable}`}
    >
      <body>
        <a
          href="#main"
          className="sr-only focus:not-sr-only focus:absolute focus:left-4 focus:top-4 focus:z-30 focus:rounded-tag focus:bg-bone-white focus:px-3 focus:py-2 focus:text-body focus:shadow-hairline"
        >
          Skip to content
        </a>

        <div className="relative min-h-screen">
          <Chrome>
            <header className="sticky top-0 z-20 bg-sage-paper/85 backdrop-blur-md">
              <nav
                className="mx-auto flex max-w-page items-center gap-6 px-6 py-3"
                aria-label="Primary"
              >
                <Link href="/" className="flex items-center gap-2">
                  <ApertureGlyph />
                  <span className="text-body-lg font-medium leading-none text-forest-ink">
                    Aperture
                  </span>
                </Link>

                <span className="hidden h-3.5 w-px bg-lichen sm:block" aria-hidden />

                <div className="flex items-center gap-5 text-body">
                  <NavLink href="/episodes">Episodes</NavLink>
                  <NavLink href="/clusters">Clusters</NavLink>
                </div>

                <div className="ml-auto flex items-center gap-4">
                  {/* The layout is a server component, so reading the auth config here never
                      ships the password or its presence check to the browser. */}
                  <SignOut enabled={authConfig().configured} />
                  {/* Outlined, not filled: the system allows exactly one filled action per screen and
                      that belongs to whatever the page itself is for — exporting a dataset,
                      recomputing clusters — not to persistent chrome. */}
                  <Link href="/episodes" className="btn hidden py-2 sm:inline-flex">
                    Open console
                  </Link>
                </div>
              </nav>
              <div className="h-px w-full bg-lichen" aria-hidden />
            </header>
          </Chrome>

          {/* The version stamp: rotated margin mark, decoration-as-information. */}
          <Chrome>
            <div
              className="version-stamp pointer-events-none fixed right-2 top-1/2 hidden -translate-y-1/2 select-none xl:block"
              aria-hidden
            >
              APERTURE · REV 0.1.0 · EVAL LAYER
            </div>
          </Chrome>

          <main id="main" className="mx-auto max-w-page px-6 pb-section pt-10">
            {children}
          </main>

          <Chrome>
            <footer className="mt-section">
              <div className="h-px w-full bg-lichen" aria-hidden />
              <div className="mx-auto max-w-page px-6 py-12">
                <div className="flex flex-wrap items-start justify-between gap-8">
                  <p className="max-w-lg text-heading-sm text-forest-ink">
                    The evaluation &amp; interpretability layer for Vision-Language-Action
                    robot policies.
                  </p>
                  <div className="flex flex-col gap-2 text-body">
                    <Link href="/episodes" className="link">
                      [ Episodes ]
                    </Link>
                    <Link href="/clusters" className="link">
                      [ Clusters ]
                    </Link>
                  </div>
                </div>
                <div className="mt-10 flex flex-wrap items-center justify-between gap-3">
                  <span className="muoto text-caption text-slate-smoke">
                    ingestion → evaluation → interpretability → loop closure
                  </span>
                  <span className="cinetype text-[11px] text-slate-smoke">
                    Aperture · eval · interp
                  </span>
                </div>
              </div>
            </footer>
          </Chrome>
        </div>
      </body>
    </html>
  );
}

/** Aperture mark: an iris of hairline blades around an open centre. */
function ApertureGlyph() {
  return (
    <svg
      width="18"
      height="18"
      viewBox="0 0 24 24"
      fill="none"
      stroke="#09352e"
      strokeWidth="1.25"
      aria-hidden
    >
      <circle cx="12" cy="12" r="9" />
      <path d="M12 3v7.2M20.8 16.5l-6.2-3.6M3.2 16.5l6.2-3.6" />
      <circle cx="12" cy="12" r="2.6" fill="#85c093" stroke="none" />
    </svg>
  );
}
