"use client";

import { usePathname } from "next/navigation";

/**
 * Hides the surrounding page chrome on the sign-in screen.
 *
 * The nav links to Episodes and Clusters and offers "Sign out" — all meaningless to someone who
 * is not signed in, and the links just bounce back to /login. A signed-out visitor should see
 * one thing to do.
 */
export function Chrome({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  if (pathname === "/login") return null;
  return <>{children}</>;
}
