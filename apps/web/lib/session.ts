/**
 * Dashboard session: an HttpOnly cookie, in one of two modes.
 *
 * - **org** (no `APERTURE_API_KEY` on the server): each person signs in with their own
 *   organization's API key. The key is sealed into the cookie with AES-GCM, so one dashboard
 *   deployment serves every customer and the proxy attaches the key of whoever is signed in.
 *   The browser holds only ciphertext it cannot read or forge.
 * - **password** (`APERTURE_API_KEY` set): the original single-org deployment — one shared
 *   password gates the dashboard and the server's key is used for every request.
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

/** AES-GCM key derived from the session secret. */
async function aesKey(secret: string): Promise<CryptoKey> {
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(`aperture-session-v1:${secret}`));
  return crypto.subtle.importKey("raw", digest, { name: "AES-GCM" }, false, ["encrypt", "decrypt"]);
}

export type OrgSession = { k: string; exp: number; slug: string; name: string };

/** Seal an org session: `<iv>.<ciphertext>`, both base64url. GCM authenticates as well as hides. */
export async function sealOrgSession(
  secret: string,
  data: Omit<OrgSession, "exp">,
  ttlSeconds = SESSION_TTL_SECONDS,
): Promise<string> {
  const iv = crypto.getRandomValues(new Uint8Array(12));
  const payload: OrgSession = { ...data, exp: Math.floor(Date.now() / 1000) + ttlSeconds };
  const ct = await crypto.subtle.encrypt(
    { name: "AES-GCM", iv },
    await aesKey(secret),
    new TextEncoder().encode(JSON.stringify(payload)),
  );
  return `${toBase64Url(iv.buffer as ArrayBuffer)}.${toBase64Url(ct)}`;
}

/** The org session in a cookie, or null if it is missing, tampered with, or expired. */
export async function openOrgSession(cookieValue: string | undefined, secret: string): Promise<OrgSession | null> {
  if (!cookieValue || !secret) return null;
  const [iv, ct] = cookieValue.split(".");
  if (!iv || !ct) return null;
  try {
    const plain = await crypto.subtle.decrypt(
      { name: "AES-GCM", iv: fromBase64Url(iv) as BufferSource },
      await aesKey(secret),
      fromBase64Url(ct) as BufferSource,
    );
    const data = JSON.parse(new TextDecoder().decode(plain)) as OrgSession;
    if (typeof data.k !== "string" || !(data.exp > Math.floor(Date.now() / 1000))) return null;
    return data;
  } catch {
    return null;
  }
}

// Local development only: lets `npm run dev` use org sign-in without configuring a secret.
// Never used when NODE_ENV=production — there, a missing secret fails closed (503).
const DEV_ONLY_SECRET = "aperture-local-development-session-secret";

export type AuthMode = "org" | "password";

/** Config read once, so the "is auth even on, and how?" rule lives in one place. */
export function authConfig() {
  const production = process.env.NODE_ENV === "production";
  const password = process.env.APERTURE_DASHBOARD_PASSWORD;
  const serverKey = process.env.APERTURE_API_KEY;
  const mode: AuthMode = serverKey ? "password" : "org";

  if (mode === "org") {
    const secret = process.env.APERTURE_SESSION_SECRET || (production ? "" : DEV_ONLY_SECRET);
    return { mode, password, secret, required: true, configured: Boolean(secret) };
  }

  const secret = process.env.APERTURE_SESSION_SECRET || password || "";
  return {
    mode,
    password,
    secret,
    // Fail closed in production: a deployment that forgot the password must not serve the
    // fleet's failure data to anyone who finds the URL. In development, absent means off, so
    // `npm run dev` needs no setup.
    required: Boolean(password) || production,
    configured: Boolean(password && secret),
  };
}

/** Is this cookie a live session for the configured mode? */
export async function hasValidSession(cookieValue: string | undefined): Promise<boolean> {
  const { mode, secret } = authConfig();
  if (mode === "org") return (await openOrgSession(cookieValue, secret)) !== null;
  return isValidSession(cookieValue, secret);
}

/** The API key a request should be made with: the signed-in org's, or the server's own. */
export async function apiKeyFor(cookieValue: string | undefined): Promise<string> {
  const { mode, secret } = authConfig();
  if (mode === "org") return (await openOrgSession(cookieValue, secret))?.k ?? "";
  return process.env.APERTURE_API_KEY || "";
}
