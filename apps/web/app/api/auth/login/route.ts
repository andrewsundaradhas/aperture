import { NextRequest, NextResponse } from "next/server";

import { SESSION_COOKIE, authConfig, issueSession, secretsMatch } from "@/lib/session";

export const dynamic = "force-dynamic";

const TTL_SECONDS = 60 * 60 * 12;

export async function POST(request: NextRequest) {
  const { password, secret, required, configured } = authConfig();

  if (!required) {
    return NextResponse.json({ detail: "Authentication is not enabled." }, { status: 400 });
  }
  if (!configured) {
    // Production with no password set: the middleware is already refusing everything. Say why,
    // rather than letting an operator conclude their password is simply wrong.
    return NextResponse.json(
      { detail: "APERTURE_DASHBOARD_PASSWORD is not set on the server." },
      { status: 503 },
    );
  }

  const body = await request.json().catch(() => ({}));
  if (typeof body?.password !== "string" || !(await secretsMatch(body.password, password!))) {
    return NextResponse.json({ detail: "Incorrect password." }, { status: 401 });
  }

  const response = NextResponse.json({ ok: true });
  response.cookies.set({
    name: SESSION_COOKIE,
    value: await issueSession(secret, TTL_SECONDS),
    path: "/",
    httpOnly: true,               // script cannot read it, so XSS cannot exfiltrate the session
    sameSite: "lax",              // not sent on cross-site POSTs
    secure: process.env.NODE_ENV === "production",
    maxAge: TTL_SECONDS,
  });
  return response;
}
