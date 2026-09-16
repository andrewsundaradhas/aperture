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
    <div className="mx-auto flex min-h-[78vh] max-w-sm flex-col justify-center">
      <div className="card gridded-fine p-8">
        <span className="tag tag-ink">Restricted</span>
        <h1 className="mt-5 text-heading font-medium text-forest-ink">Aperture</h1>
        <p className="mt-2 text-body text-slate-smoke">
          This dashboard shows a fleet&rsquo;s failure data. Sign in to continue.
        </p>

        <div className="mt-6 h-px w-full bg-lichen" aria-hidden />

        <form onSubmit={submit} className="mt-6 space-y-3">
          <label className="cinetype block text-[11px] text-slate-smoke" htmlFor="password">
            PASSWORD
          </label>
          <input
            id="password"
            type="password"
            autoFocus
            autoComplete="current-password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            className="field"
          />
          {error && (
            <p role="alert" className="flex items-center gap-2 text-body text-alarm">
              <span className="h-2 w-2 shrink-0 rounded-full bg-alarm" aria-hidden />
              {error}
            </p>
          )}
          <button type="submit" className="btn-filled w-full" disabled={busy || !password}>
            {busy && <Spinner />}
            Sign in
          </button>
        </form>
      </div>

      <p className="muoto mt-6 text-center text-caption text-slate-smoke">
        aperture · evaluation &amp; interpretability layer
      </p>
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
