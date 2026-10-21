"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

/** Nav item that marks the section you are actually in.
 *
 * Matches on prefix so an episode or cluster detail page still highlights its section.
 */
export function NavLink({
  href,
  exact = false,
  children,
}: {
  href: string;
  exact?: boolean;
  children: React.ReactNode;
}) {
  const pathname = usePathname();
  const active = pathname === href || (!exact && pathname.startsWith(`${href}/`));
  return (
    <Link
      href={href}
      aria-current={active ? "page" : undefined}
      className={`relative py-0.5 transition ${
        active
          ? "text-forest-ink after:absolute after:-bottom-1 after:left-0 after:h-0.5 after:w-full after:rounded-full after:bg-lavender after:content-['']"
          : "text-slate-smoke hover:text-forest-ink"
      }`}
    >
      {children}
    </Link>
  );
}
