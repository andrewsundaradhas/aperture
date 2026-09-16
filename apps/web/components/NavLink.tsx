"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

/** Nav item that marks the section you are actually in.
 *
 * Matches on prefix so an episode or cluster detail page still highlights its section.
 */
export function NavLink({ href, children }: { href: string; children: React.ReactNode }) {
  const pathname = usePathname();
  const active = pathname === href || pathname.startsWith(`${href}/`);
  return (
    <Link
      href={href}
      aria-current={active ? "page" : undefined}
      className={`relative py-0.5 transition ${
        active
          ? "text-forest-ink after:absolute after:-bottom-1 after:left-0 after:h-px after:w-full after:bg-forest-ink after:content-['']"
          : "text-slate-smoke hover:text-forest-ink"
      }`}
    >
      {children}
    </Link>
  );
}
