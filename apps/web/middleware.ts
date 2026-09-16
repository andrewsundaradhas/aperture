/**
 * Gate every dashboard route behind a session.
 *
 * This guards the API proxy too (`/api/aperture/*`). Guarding only the pages would be
 * decorative: the proxy is what holds the API key, so anyone who could still reach it would
 * have full access to the org's episodes without ever loading a page.
 */

import { NextRequest, NextResponse } from "next/server";

import { SESSION_COOKIE, authConfig, isValidSession } from "@/lib/session";

export async function middleware(request: NextRequest) {
  const { secret, required, configured } = authConfig();

  if (!required) return NextResponse.next();

  if (!configured) {
    // Production without a password: refuse rather than serve fleet data to anyone with the URL.
    return NextResponse.json(
      { detail: "Dashboard auth is required but APERTURE_DASHBOARD_PASSWORD is not set." },
      { status: 503 },
    );
  }

  if (await isValidSession(request.cookies.get(SESSION_COOKIE)?.value, secret)) {
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
  matcher: ["/((?!login|api/auth|_next/static|_next/image|favicon.ico|icon.svg).*)"],
};
