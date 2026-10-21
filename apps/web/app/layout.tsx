import type { Metadata } from "next";
import Link from "next/link";
// Self-hosted fonts (bundled from npm), not next/font/google: Google Fonts are fetched at build
// time, which made every build depend on fonts.googleapis.com, and on Vercel the Turbopack
// google-font loader failed outright ("next/font/google queries have exactly one entry").
import "@fontsource/inter/400.css";
import "@fontsource/inter/500.css";
import "@fontsource/inter/600.css";
import "@fontsource/anton/400.css";
import "./globals.css";
import { NavLink } from "@/components/NavLink";
import { Chrome } from "@/components/Chrome";
import { INTRO_HEAD_SCRIPT, Intro } from "@/components/Intro";
import { SignOut } from "@/components/SignOut";
import { cookies } from "next/headers";

import { SESSION_COOKIE, authConfig, openOrgSession } from "@/lib/session";

// Substitutes for the Operate brand faces, per its own stated fallbacks:
//   denim    → Inter            (body, headings, UI)
//   muoto    → IBM Plex Mono    (labels, captions, axis ticks)
//   cinetype → JetBrains Mono   (ALL-CAPS specimen labels at wide tracking)
// The --font-* variables these map to are defined in globals.css.

export const metadata: Metadata = {
  title: "Aperture",
  description: "Find out why your robot policies fail, and fix it.",
};

export default async function RootLayout({ children }: { children: React.ReactNode }) {
  // Server component: the session is unsealed here and only the org's display name reaches the
  // page. The API key inside the cookie never leaves the server.
  // Read the cookie unconditionally: that makes every page render per request. Branching on
  // the mode first let Next pre-render pages at build time with the *build* machine's settings,
  // so a build made with single-org settings served no org name and no Sign out button even when
  // the server ran in per-org mode.
  const cookieValue = (await cookies()).get(SESSION_COOKIE)?.value;
  const auth = authConfig();
  const org = auth.mode === "org" ? await openOrgSession(cookieValue, auth.secret) : null;
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        {/* Hides the intro before first paint when it has already been seen this session. */}
        <script dangerouslySetInnerHTML={{ __html: INTRO_HEAD_SCRIPT }} />
      </head>
      <body>
        <Intro />
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
                className="mx-auto flex max-w-page items-center gap-4 px-4 py-3 sm:gap-6 sm:px-6"
                aria-label="Primary"
              >
                <Link href="/" className="flex shrink-0 items-center gap-2">
                  <ApertureGlyph />
                  <span className="font-display text-[22px] leading-none tracking-[0.02em] text-lavender">
                    Aperture
                  </span>
                </Link>

                <span className="hidden h-3.5 w-px bg-lichen sm:block" aria-hidden />

                {/* On a phone the links scroll sideways inside the bar instead of widening the page. */}
                <div className="flex min-w-0 flex-1 items-center gap-5 overflow-x-auto whitespace-nowrap py-1 text-body [scrollbar-width:none]">
                  <NavLink href="/" exact>
                    Overview
                  </NavLink>
                  <NavLink href="/episodes">Episodes</NavLink>
                  <NavLink href="/clusters">Failure modes</NavLink>
                  <NavLink href="/upload">Upload</NavLink>
                  <NavLink href="/settings">Settings</NavLink>
                </div>

                <div className="ml-auto flex items-center gap-4">
                  {/* The layout is a server component, so reading the auth config here never
                      ships the password or its presence check to the browser. */}
                  {org && (
                    <span className="muoto hidden text-caption text-slate-smoke md:inline" title={org.slug}>
                      {org.name}
                    </span>
                  )}
                  <SignOut enabled={auth.configured && (auth.mode === "password" || org !== null)} />
                </div>
              </nav>
              <div className="h-px w-full bg-lichen" aria-hidden />
            </header>
          </Chrome>


          <main id="main" className="mx-auto max-w-page px-6 pb-section pt-10">
            {children}
          </main>

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
      stroke="#e894ff"
      strokeWidth="1.25"
      aria-hidden
    >
      <circle cx="12" cy="12" r="9" />
      <path d="M12 3v7.2M20.8 16.5l-6.2-3.6M3.2 16.5l6.2-3.6" />
      <circle cx="12" cy="12" r="2.6" fill="#e894ff" stroke="none" />
    </svg>
  );
}
