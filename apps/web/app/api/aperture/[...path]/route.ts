/**
 * Server-side proxy to the Aperture API.
 *
 * The dashboard used to call the backend directly with `NEXT_PUBLIC_API_KEY`. Anything named
 * `NEXT_PUBLIC_*` is inlined into the JavaScript bundle, so that key was readable by everyone
 * who loaded the page — and it is an org-scoped credential granting full access to that org's
 * episodes. Any key that was ever exposed that way must be treated as compromised and revoked
 * (`DELETE /v1/onboarding/keys/{id}`), not merely moved.
 *
 * Every dashboard request now goes to this route on the same origin, and the key is attached
 * here, server-side, where the browser cannot see it: in org mode it is unsealed from the
 * signed-in person's session cookie, in single-org mode it is the server's APERTURE_API_KEY.
 *
 * Bodies are streamed through, so large uploads to the local storage backend work; on R2/S3 the
 * browser PUTs straight to the bucket's presigned URL and never comes through here.
 */

import { NextRequest } from "next/server";

import { SESSION_COOKIE, apiKeyFor, authConfig, hasValidSession } from "@/lib/session";

const API_BASE = process.env.APERTURE_API_BASE_URL || "http://localhost:8000";

// Always hit the backend: dashboard data is per-request, and a cached response would be
// another tenant's data under a different session.
export const dynamic = "force-dynamic";

async function proxy(request: NextRequest, path: string[]): Promise<Response> {
  // This route is excluded from the page guard in `proxy.ts` (Next's proxy layer truncates
  // request bodies over 10 MB, which silently corrupted dataset uploads), so it checks the
  // session itself. In single-org mode this check is what keeps the server's key from being
  // used by anyone who can reach the URL.
  const auth = authConfig();
  const cookie = request.cookies.get(SESSION_COOKIE)?.value;
  if (auth.required) {
    if (!auth.configured) {
      return Response.json({ detail: "Dashboard auth is required but not configured on the server." }, { status: 503 });
    }
    if (!(await hasValidSession(cookie))) {
      return Response.json({ detail: "Not authenticated." }, { status: 401 });
    }
  }

  const target = `${API_BASE}/${path.join("/")}${request.nextUrl.search}`;

  // Forward only what the upstream needs. Notably not cookies: the API authenticates with the
  // key attached below, and passing browser cookies upstream would widen the trust boundary.
  const headers = new Headers();
  const contentType = request.headers.get("content-type");
  if (contentType) headers.set("content-type", contentType);
  // The signed-in org's key (org mode) or the server's own key (single-org mode).
  headers.set("X-API-Key", await apiKeyFor(cookie));
  const length = request.headers.get("content-length");
  if (length) headers.set("content-length", length);

  const hasBody = request.method !== "GET" && request.method !== "HEAD";

  let upstream: Response;
  try {
    upstream = await fetch(target, {
      method: request.method,
      headers,
      // Streamed, not buffered: a dataset upload on the local storage backend comes through
      // here and can be gigabytes.
      body: hasBody ? request.body : undefined,
      cache: "no-store",
      // Required by Node's fetch to send a streaming body.
      ...(hasBody ? { duplex: "half" } : {}),
    } as RequestInit);
  } catch (error) {
    // The backend being down is a 502, not a dashboard crash.
    return Response.json(
      { detail: `Cannot reach the Aperture API at ${API_BASE}: ${(error as Error).message}` },
      { status: 502 },
    );
  }

  const responseHeaders = new Headers();
  for (const header of ["content-type", "content-disposition"]) {
    const value = upstream.headers.get(header);
    if (value) responseHeaders.set(header, value);
  }
  return new Response(upstream.body, { status: upstream.status, headers: responseHeaders });
}

type Context = { params: Promise<{ path: string[] }> };

export async function GET(request: NextRequest, ctx: Context) {
  return proxy(request, (await ctx.params).path);
}
export async function POST(request: NextRequest, ctx: Context) {
  return proxy(request, (await ctx.params).path);
}
export async function PUT(request: NextRequest, ctx: Context) {
  return proxy(request, (await ctx.params).path);
}
export async function PATCH(request: NextRequest, ctx: Context) {
  return proxy(request, (await ctx.params).path);
}
export async function DELETE(request: NextRequest, ctx: Context) {
  return proxy(request, (await ctx.params).path);
}
