/**
 * Dashboard session: an HMAC-signed, HttpOnly cookie.
 *
 * Deliberately no new dependency. NextAuth/Lucia/iron-session are all better answers once an
 * identity provider is chosen, and swapping this out is a contained change — but adding an auth
 * library that could not be built or exercised here would have been worse than a small amount
 * of well-understood code.
 *
 * Uses **Web Crypto**, not `node:crypto`, because Next.js middleware runs on the Edge runtime
 * where the Node builtin is unavailable. `crypto.subtle.verify` also compares in constant time,
 * so signature checking does not leak by timing.
 *
 * What this is: a shared-secret gate, appropriate for a pilot behind a VPN. What it is not:
 * per-user identity. There are no accounts, so it cannot tell you who did something. Put SSO in
 * front of it before that matters — see DEPLOY.md.
 */

export const SESSION_COOKIE = "aperture_session";
const SESSION_TTL_SECONDS = 60 * 60 * 12;

function toBase64Url(bytes: ArrayBuffer): string {
  const binary = String.fromCharCode(...new Uint8Array(bytes));
  return btoa(binary).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

function fromBase64Url(value: string): Uint8Array {
  const padded = value.replace(/-/g, "+").replace(/_/g, "/");
  const binary = atob(padded + "=".repeat((4 - (padded.length % 4)) % 4));
  return Uint8Array.from(binary, (c) => c.charCodeAt(0));
}

async function hmacKey(secret: string): Promise<CryptoKey> {
  return crypto.subtle.importKey(
    "raw",
    new TextEncoder().encode(secret),
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["sign", "verify"],
  );
}

/** A cookie value of `<expiry>.<signature>`. The expiry is signed, so it cannot be extended. */
export async function issueSession(secret: string, ttlSeconds = SESSION_TTL_SECONDS): Promise<string> {
  const expiry = String(Math.floor(Date.now() / 1000) + ttlSeconds);
  const signature = await crypto.subtle.sign(
    "HMAC",
    await hmacKey(secret),
    new TextEncoder().encode(expiry),
  );
  return `${expiry}.${toBase64Url(signature)}`;
}

export async function isValidSession(cookieValue: string | undefined, secret: string): Promise<boolean> {
  if (!cookieValue) return false;
  const [expiry, signature] = cookieValue.split(".");
  if (!expiry || !signature) return false;

  let ok = false;
  try {
    ok = await crypto.subtle.verify(
      "HMAC",
      await hmacKey(secret),
      fromBase64Url(signature),
      new TextEncoder().encode(expiry),
    );
  } catch {
    return false; // malformed base64 is just an invalid cookie
  }
  // Check the signature before trusting the expiry it protects.
  return ok && Number(expiry) > Math.floor(Date.now() / 1000);
}

/** Compare two secrets without leaking their relationship through timing.
 *
 * HMACs the candidate, then asks `verify` whether that digest is also the digest of the
 * expected value — true only when the two inputs are equal. Both the hashing and `verify`'s
 * comparison are constant time, so a wrong password takes as long as a right one and an
 * attacker learns nothing from how quickly they were rejected.
 */
export async function secretsMatch(candidate: string, expected: string): Promise<boolean> {
  const key = await hmacKey(expected);
  const candidateDigest = await crypto.subtle.sign(
    "HMAC",
    key,
    new TextEncoder().encode(candidate),
  );
  return crypto.subtle.verify("HMAC", key, candidateDigest, new TextEncoder().encode(expected));
}

/** Config read once, so the "is auth even on?" rule lives in one place. */
export function authConfig() {
  const password = process.env.APERTURE_DASHBOARD_PASSWORD;
  const secret = process.env.APERTURE_SESSION_SECRET || password || "";
  return {
    password,
    secret,
    // Fail closed in production: a deployment that forgot the password must not serve the
    // fleet's failure data to anyone who finds the URL. In development, absent means off, so
    // `npm run dev` needs no setup.
    required: Boolean(password) || process.env.NODE_ENV === "production",
    configured: Boolean(password && secret),
  };
}
