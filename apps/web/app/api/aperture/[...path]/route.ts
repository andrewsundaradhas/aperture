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
 * here, server-side, where the browser cannot see it.
 *
 * NOTE: this proxies the request body through memory, which is fine for JSON but not for large
 * episode uploads. Those should go straight to object storage with a presigned URL.
 */

import { NextRequest } from "next/server";

const API_BASE = process.env.APERTURE_API_BASE_URL || "http://localhost:8000";
const API_KEY = process.env.APERTURE_API_KEY || "";

// Always hit the backend: dashboard data is per-request, and a cached response would be
// another tenant's data under a different session.
export const dynamic = "force-dynamic";

async function proxy(request: NextRequest, path: string[]): Promise<Response> {
  const target = `${API_BASE}/${path.join("/")}${request.nextUrl.search}`;

  // Forward only what the upstream needs. Notably not cookies: the API authenticates with the
  // key attached below, and passing browser cookies upstream would widen the trust boundary.
  const headers = new Headers();
  const contentType = request.headers.get("content-type");
  if (contentType) headers.set("content-type", contentType);
  headers.set("X-API-Key", API_KEY);

  const hasBody = request.method !== "GET" && request.method !== "HEAD";

  let upstream: Response;
  try {
    upstream = await fetch(target, {
      method: request.method,
      headers,
      body: hasBody ? await request.arrayBuffer() : undefined,
      cache: "no-store",
    });
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
