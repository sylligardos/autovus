import { TopBar } from "@/components/TopBar";
import Link from "next/link";
import { FinderView } from "@/components/FinderView";
import { LocalView } from "@/components/LocalView";
import { loadSeriesList, loadSummary } from "@/lib/api";
import { href, param, stateFrom, type SeriesMeta, type State } from "@/lib/model";

// The two tabs of Explore: one series with every detector, and the table of where
// the whole-series AutoVUS and the per-anomaly AutoVUS are far apart (its rows open a series).
const TABS = [{ id: "series", label: "Series" }, { id: "gaps", label: "Global vs local" }] as const;
// The one sentence on top of each tab: what it answers, then how to read it.
const LINES: Record<string, [string, string]> = {
  series: ["How well does each detector find every anomaly of a series?", "One lane per detector. The number at an anomaly is the detector's AutoVUS on it."],
  gaps: ["Where does one score per series hide what happens on its anomalies?", "Each point is a detector on a series, placed by its AutoVUS on the whole series and on its anomalies."],
};
function Tabs({ state, tab }: { state: State; tab: string }) {
  return (
    <nav aria-label="Explore" style={{ width: "100%", maxWidth: 1600, margin: "0 auto", padding: "14px 24px 0" }}>
      <div style={{ display: "flex", gap: 24, borderBottom: "1px solid var(--border)" }}>
        {TABS.map((t) => (
          <Link key={t.id} href={href("/local", state, t.id === "series" ? {} : { tab: t.id })} aria-current={t.id === tab ? "page" : undefined}
            style={{ padding: "6px 2px 8px", marginBottom: -1, fontSize: 14, fontWeight: 600, textDecoration: "none", color: t.id === tab ? "var(--ink)" : "var(--muted)", borderBottom: `2px solid ${t.id === tab ? "var(--accent)" : "transparent"}` }}>
            {t.label}
          </Link>
        ))}
      </div>
      <p className="page-line" style={{ marginTop: 14 }}>{LINES[tab][0]} <span>{LINES[tab][1]}</span></p>
    </nav>
  );
}

export default async function Local({ searchParams }: { searchParams: Promise<Record<string, string | string[] | undefined>> }) {
  const params = await searchParams;
  const state = stateFrom(params);
  const tab = param(params, "tab") === "gaps" ? "gaps" : "series";
  if (tab === "gaps") {
    return (<><TopBar state={state} current="/local" /><Tabs state={state} tab={tab} /><div className="page"><FinderView state={state} /></div></>);
  }
  const summary = await loadSummary(state.benchmark);
  if (!summary) {
    return (<><TopBar state={state} current="/local" /><div className="page"><main className="content"><div className="panel"><span className="muted">The Python side is not running.</span></div></main></div></>);
  }
  const lists = await Promise.all(summary.datasets.map(async (d) => [d, (await loadSeriesList(state.benchmark, d)) ?? []] as const));
  const seriesByDataset: Record<string, SeriesMeta[]> = Object.fromEntries(lists.filter(([, l]) => l.length));
  const det = param(params, "det");
  const v = param(params, "v")?.split(",").map(Number);
  const a = param(params, "a");
  const pick = a !== null && /^\d+$/.test(a) ? Number(a) : null;
  const view: [number, number] | null = v && v.length === 2 && v.every(Number.isFinite) && v[1] > v[0] ? [v[0], v[1]] : null;
  return (
    <>
      <TopBar state={state} current="/local" />
      <Tabs state={state} tab={tab} />
      <LocalView state={state} seriesByDataset={seriesByDataset} initialFile={param(params, "s")}
        initialDet={det ? det.split(",").filter(Boolean) : null} initialView={view} initialPick={pick} />
    </>
  );
}
