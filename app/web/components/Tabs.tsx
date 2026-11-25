import Link from "next/link";
import { href, type State } from "@/lib/model";

// The tabs of a page (?tab=, the first one has no parameter) and, when given, the one sentence under them:
// what the tab answers, then how to read it.
export function Tabs({ state, path, label, tabs, tab, line }: {
  state: State; path: string; label: string; tabs: { id: string; label: string }[]; tab: string; line?: [string, string];
}) {
  return (
    <nav aria-label={label} style={{ width: "100%", maxWidth: 1600, margin: "0 auto", padding: "14px 24px 0" }}>
      <div style={{ display: "flex", gap: 24, borderBottom: "1px solid var(--border)" }}>
        {tabs.map((t, k) => (
          <Link key={t.id} href={href(path, state, k === 0 ? {} : { tab: t.id })} aria-current={t.id === tab ? "page" : undefined}
            style={{ padding: "6px 2px 8px", marginBottom: -1, fontSize: 14, fontWeight: 600, textDecoration: "none", color: t.id === tab ? "var(--ink)" : "var(--muted)", borderBottom: `2px solid ${t.id === tab ? "var(--accent)" : "transparent"}` }}>
            {t.label}
          </Link>
        ))}
      </div>
      {line && <p className="page-line" style={{ marginTop: 14 }}>{line[0]} <span>{line[1]}</span></p>}
    </nav>
  );
}
