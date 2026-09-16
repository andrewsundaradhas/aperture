"use client";

import { Suspense, useState } from "react";
import { useSearchParams } from "next/navigation";

import { Spinner } from "@/components/ui";

function LoginForm() {
  const params = useSearchParams();
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function submit(event: React.FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const res = await fetch("/api/auth/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ password }),
      });
      if (!res.ok) {
        setError((await res.json().catch(() => ({})))?.detail ?? "Sign in failed.");
        return;
      }
      // Only ever return to a path on this origin — a `next` of `https://elsewhere` would
      // otherwise turn the login page into an open redirect.
      const next = params.get("next");
      const destination = next && next.startsWith("/") && !next.startsWith("//") ? next : "/";

      // A full navigation rather than router.replace(). Two reasons: the session cookie has to
      // be attached by the browser to a fresh request for middleware to see it (a client-side
      // RSC navigation raced the cookie and bounced straight back to this page), and the root
      // layout reads the auth config server-side, so it needs to re-render for the nav and
      // "Sign out" to appear.
      window.location.assign(destination);
    } catch {
      setError("Could not reach the server.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="mx-auto flex min-h-[70vh] max-w-sm flex-col justify-center">
      <h1 className="serif text-heading-sm text-graphite">Aperture</h1>
      <p className="mt-1 text-body-sm text-ash">
        This dashboard shows a fleet&rsquo;s failure data. Sign in to continue.
      </p>

      <form onSubmit={submit} className="mt-6 space-y-3">
        <label className="block text-body-sm text-ash" htmlFor="password">
          Password
        </label>
        <input
          id="password"
          type="password"
          autoFocus
          autoComplete="current-password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          className="w-full rounded-md border border-mist bg-white px-3 py-2 text-body-sm"
        />
        {error && (
          <p role="alert" className="text-body-sm text-motor">
            {error}
          </p>
        )}
        <button type="submit" className="btn-filled w-full" disabled={busy || !password}>
          {busy && <Spinner />}
          Sign in
        </button>
      </form>
    </div>
  );
}

export default function LoginPage() {
  // useSearchParams needs a Suspense boundary for this route to stay statically renderable.
  return (
    <Suspense fallback={null}>
      <LoginForm />
    </Suspense>
  );
}
