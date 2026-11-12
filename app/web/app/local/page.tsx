import { TopBar } from "@/components/TopBar";
import { LocalView } from "@/components/LocalView";
import { loadSeriesList, loadSummary } from "@/lib/api";
import { param, stateFrom, type SeriesMeta } from "@/lib/model";

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
  const view: [number, number] | null = v && v.length === 2 && v.every(Number.isFinite) && v[1] > v[0] ? [v[0], v[1]] : null;
  return (
    <>
      <TopBar state={state} current="/local" />
      <LocalView state={state} seriesByDataset={seriesByDataset} initialFile={param(params, "s")}
        initialDet={det ? det.split(",").filter(Boolean) : null} initialView={view} />
    </>
  );
}
