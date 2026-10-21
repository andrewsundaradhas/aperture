import { NextRequest, NextResponse } from "next/server";

import { clientId, recordFailure, recordSuccess, retryAfter } from "@/lib/loginLimit";
import { SESSION_COOKIE, authConfig, issueSession, sealOrgSession, secretsMatch } from "@/lib/session";

export const dynamic = "force-dynamic";

const TTL_SECONDS = 60 * 60 * 12;
const API_BASE = process.env.APERTURE_API_BASE_URL || "http://localhost:8000";

function cookie(response: NextResponse, value: string) {
  response.cookies.set({
    name: SESSION_COOKIE,
    value,
    path: "/",
    httpOnly: true,               // script cannot read it, so XSS cannot exfiltrate the session
    sameSite: "lax",              // not sent on cross-site POSTs
    secure: process.env.NODE_ENV === "production",
    maxAge: TTL_SECONDS,
  });
}

/** Which sign-in the login page should render. Says nothing secret. */
export async function GET() {
  const { mode, required } = authConfig();
  return NextResponse.json({ mode, required });
}

export async function POST(request: NextRequest) {
  const { mode, password, secret, required, configured } = authConfig();

  if (!required) {
    return NextResponse.json({ detail: "Authentication is not enabled." }, { status: 400 });
  }
  if (!configured) {
    // Production with no secret set: the proxy is already refusing everything. Say why, rather
    // than letting an operator conclude their credentials are simply wrong.
    const missing = mode === "org" ? "APERTURE_SESSION_SECRET" : "APERTURE_DASHBOARD_PASSWORD";
    return NextResponse.json({ detail: `${missing} is not set on the server.` }, { status: 503 });
  }

  const id = clientId(request.headers);
  const wait = retryAfter(id);
  if (wait > 0) {
    return NextResponse.json(
      { detail: `Too many failed sign-ins. Try again in ${Math.ceil(wait / 60)} min.` },
      { status: 429, headers: { "Retry-After": String(wait) } },
    );
  }

  const body = await request.json().catch(() => ({}));

  if (mode === "org") {
    const apiKey = typeof body?.apiKey === "string" ? body.apiKey.trim() : "";
    let org: { slug: string; name: string } | null = null;
    if (apiKey) {
      try {
        const res = await fetch(`${API_BASE}/v1/onboarding/me`, {
          headers: { "X-API-Key": apiKey },
          cache: "no-store",
        });
        if (res.ok) org = await res.json();
        else if (res.status !== 401 && res.status !== 403) {
          return NextResponse.json({ detail: `The Aperture API answered ${res.status}.` }, { status: 502 });
        }
      } catch {
        return NextResponse.json({ detail: `Cannot reach the Aperture API at ${API_BASE}.` }, { status: 502 });
      }
    }
    if (!org) {
      recordFailure(id);
      return NextResponse.json({ detail: "That API key was not recognised." }, { status: 401 });
    }
    recordSuccess(id);
    const response = NextResponse.json({ ok: true, org: { slug: org.slug, name: org.name } });
    cookie(response, await sealOrgSession(secret, { k: apiKey, slug: org.slug, name: org.name }, TTL_SECONDS));
    return response;
  }

  if (typeof body?.password !== "string" || !(await secretsMatch(body.password, password!))) {
    recordFailure(id);
    return NextResponse.json({ detail: "Incorrect password." }, { status: 401 });
  }
  recordSuccess(id);
  const response = NextResponse.json({ ok: true });
  cookie(response, await issueSession(secret, TTL_SECONDS));
  return response;
}
