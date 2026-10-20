/**
 * Gate every dashboard route behind a session.
 *
 * The API proxy (`/api/aperture/*`) is guarded too, but by the route itself rather than here —
 * see the matcher note below. Guarding only the pages would be decorative: the proxy is what
 * holds the API key.
 */

import { NextRequest, NextResponse } from "next/server";

import { SESSION_COOKIE, authConfig, hasValidSession } from "@/lib/session";

export async function proxy(request: NextRequest) {
  const { mode, required, configured } = authConfig();

  if (!required) return NextResponse.next();

  if (!configured) {
    // Production without a secret: refuse rather than serve fleet data to anyone with the URL.
    const missing = mode === "org" ? "APERTURE_SESSION_SECRET" : "APERTURE_DASHBOARD_PASSWORD";
    return NextResponse.json(
      { detail: `Dashboard auth is required but ${missing} is not set.` },
      { status: 503 },
    );
  }

  if (await hasValidSession(request.cookies.get(SESSION_COOKIE)?.value)) {
    return NextResponse.next();
  }

  // An expired session on a fetch should surface as 401 for the caller to handle; on a
  // navigation it should land the person on the login page.
  if (request.nextUrl.pathname.startsWith("/api/")) {
    return NextResponse.json({ detail: "Not authenticated." }, { status: 401 });
  }

  const login = new URL("/login", request.url);
  login.searchParams.set("next", request.nextUrl.pathname + request.nextUrl.search);
  return NextResponse.redirect(login);
}

export const config = {
  // Everything except the login page, the auth endpoints themselves, and Next's static assets —
  // excluding those is what keeps the redirect from looping.
  //
  // `icon.svg` is the App Router's generated favicon route. It is public branding, not fleet
  // data, and the browser requests it on the login page too — gated, it 307s to /login and the
  // tab renders without an icon.
  //
  // `api/aperture` (the API proxy) is excluded and authenticates itself: Next's proxy layer
  // buffers request bodies and truncates them past 10 MB, which corrupted dataset uploads.
  matcher: ["/((?!login|api/auth|api/aperture|_next/static|_next/image|favicon.ico|icon.svg).*)"],
};
