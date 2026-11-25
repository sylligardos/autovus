import { TopBar } from "@/components/TopBar";
import { LocalView } from "@/components/LocalView";
import { loadSeriesList, loadSummary } from "@/lib/api";
import { param, stateFrom, type SeriesMeta } from "@/lib/model";

// Explore: one series with every detector. The one sentence on top: what it answers, then how to read it.
const LINE: [string, string] = ["How well does each detector find every anomaly of a series?", "One lane per detector. The number at an anomaly is the detector's AutoVUS on it."];
function Line() {
  return (
    <nav aria-label="Explore" style={{ width: "100%", maxWidth: 1600, margin: "0 auto", padding: "14px 24px 0" }}>
      <p className="page-line" style={{ marginTop: 14 }}>{LINE[0]} <span>{LINE[1]}</span></p>
    </nav>
  );
}

export default async function Local({ searchParams }: { searchParams: Promise<Record<string, string | string[] | undefined>> }) {
  const params = await searchParams;
  const state = stateFrom(params);
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
      <Line />
      <LocalView state={state} seriesByDataset={seriesByDataset} initialFile={param(params, "s")}
        initialDet={det ? det.split(",").filter(Boolean) : null} initialView={view} initialPick={pick} initialMode={param(params, "show") === "best" ? "best" : null} />
    </>
  );
}
