import Link from "next/link";
import { BENCHMARKS, PAGES, State, href } from "@/lib/model";
import { HelpButton } from "./HelpButton";
import { ThemeToggle } from "./ThemeToggle";
import { BenchmarkSelect } from "./BenchmarkSelect";

export function TopBar({ state, current }: { state: State; current: string }) {
  return (
    <header className="topbar">
      <Link href={href("/", state)} className="wordmark">
        <svg width="22" height="22" viewBox="0 0 22 22" fill="none" aria-hidden="true">
          <rect x="2" y="6" width="18" height="10" rx="2" stroke="currentColor" strokeWidth="1.5" />
          <path d="M4 13 L7 8 L10 13 L13 10 L16 13" stroke="var(--accent-on-dark)" strokeWidth="2" strokeLinejoin="round" strokeLinecap="round" />
        </svg>
        <span>AUTOVUS</span>
      </Link>
      <nav className="nav" aria-label="Pages">
        {PAGES.map((p) => (
          <Link key={p.path} href={href(p.path, state)} aria-current={p.path === current ? "page" : undefined}>
            {p.label}
          </Link>
        ))}
      </nav>
      <div className="controls">
        <BenchmarkSelect value={state.benchmark} options={BENCHMARKS} current={current} />
        <ThemeToggle />
        <HelpButton page={current} />
      </div>
    </header>
  );
}
