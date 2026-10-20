/**
 * Failed-sign-in throttle.
 *
 * Without it the login endpoint answered unlimited guesses as fast as they arrived. A shared
 * dashboard password is exactly the kind of secret a person picks, so it needs a budget: after
 * MAX_FAILURES wrong attempts from one client inside WINDOW_MS, further attempts get 429 until
 * the window rolls over. A successful sign-in clears the client's count.
 *
 * In memory, per server instance — the same trade the API's rate limiter makes. On a fleet of
 * serverless instances an attacker spread across them gets N times the budget, which still turns
 * "unlimited" into "a few dozen an hour". Move to a shared store (Redis/KV) when that matters.
 */

const MAX_FAILURES = 10;
const WINDOW_MS = 15 * 60 * 1000;
const MAX_TRACKED = 10_000;

type Entry = { failures: number; resetAt: number };
const entries = new Map<string, Entry>();

export function clientId(headers: Headers): string {
  // First hop of X-Forwarded-For is the client on Vercel and behind most proxies.
  const fwd = headers.get("x-forwarded-for")?.split(",")[0]?.trim();
  return fwd || headers.get("x-real-ip") || "unknown";
}

function current(id: string, now: number): Entry | undefined {
  const e = entries.get(id);
  if (e && e.resetAt <= now) {
    entries.delete(id);
    return undefined;
  }
  return e;
}

/** Seconds until this client may try again, or 0 if it may try now. */
export function retryAfter(id: string, now = Date.now()): number {
  const e = current(id, now);
  return e && e.failures >= MAX_FAILURES ? Math.ceil((e.resetAt - now) / 1000) : 0;
}

export function recordFailure(id: string, now = Date.now()): void {
  if (entries.size >= MAX_TRACKED) {
    for (const [k, v] of entries) if (v.resetAt <= now) entries.delete(k);
    if (entries.size >= MAX_TRACKED) entries.delete(entries.keys().next().value as string);
  }
  const e = current(id, now) ?? { failures: 0, resetAt: now + WINDOW_MS };
  e.failures += 1;
  entries.set(id, e);
}

export function recordSuccess(id: string): void {
  entries.delete(id);
}
