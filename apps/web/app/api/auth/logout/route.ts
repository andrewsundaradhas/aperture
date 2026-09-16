import { NextResponse } from "next/server";

import { SESSION_COOKIE } from "@/lib/session";

export const dynamic = "force-dynamic";

export async function POST() {
  const response = NextResponse.json({ ok: true });
  // Expire rather than delete-by-omission, so the browser drops it immediately.
  response.cookies.set({ name: SESSION_COOKIE, value: "", path: "/", httpOnly: true, maxAge: 0 });
  return response;
}
