"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";

/** Sign-out control. Renders nothing when the dashboard has no sign-in configured, so a local
 *  `npm run dev` does not show a button that cannot do anything. */
export function SignOut({ enabled }: { enabled: boolean }) {
  const router = useRouter();
  const [busy, setBusy] = useState(false);

  if (!enabled) return null;

  return (
    <button
      type="button"
      disabled={busy}
      onClick={async () => {
        setBusy(true);
        await fetch("/api/auth/logout", { method: "POST" });
        router.replace("/login");
        router.refresh();
      }}
      className="text-body text-slate-smoke transition hover:text-forest-ink disabled:opacity-40"
    >
      Sign out
    </button>
  );
}
