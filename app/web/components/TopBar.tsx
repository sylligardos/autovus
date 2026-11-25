import Link from "next/link";
import { PAGES, State, href } from "@/lib/model";
import { GearSwitch } from "./GearSwitch";
import { HelpButton } from "./HelpButton";
import { ThemeToggle } from "./ThemeToggle";

export function TopBar({ state, current }: { state: State; current: string }) {
  return (
    <header className="topbar">
      <Link href={href("/", state)} className="wordmark">
        <svg width="22" height="22" viewBox="0 0 22 22" fill="none" aria-hidden="true">
          <rect x="5" y="5" width="4" height="11" fill="var(--accent-on-dark)" fillOpacity="0.36" />
          <path d="M4 13 L7 8 L10 13 L13 10 L16 13" stroke="var(--accent-on-dark)" strokeWidth="2" strokeLinejoin="round" strokeLinecap="round" />
        </svg>
        <span>PROTAGONIST</span>
      </Link>
      <nav className="nav" aria-label="Pages">
        {PAGES.map((p) => (
          <Link key={p.path} href={href(p.path, state)} aria-current={p.path === current ? "page" : undefined}>
            {p.label}
          </Link>
        ))}
      </nav>
      <div className="controls">
        <Link href={href("/play", state)} className="play-btn" aria-current={current === "/play" ? "page" : undefined}>
          <svg width="10" height="10" viewBox="0 0 10 10" aria-hidden="true"><path d="M1.5 0.5v9l8-4.5z" fill="currentColor" /></svg>
          Play
        </Link>
        <GearSwitch L={state.L} />
        <ThemeToggle />
        <HelpButton page={current} />
      </div>
    </header>
  );
}
