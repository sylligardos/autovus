import { TopBar } from "@/components/TopBar";
import { SeriesView } from "@/components/SeriesView";
import { loadSeriesList, loadSummary } from "@/lib/api";
import { param, stateFrom, type SeriesMeta } from "@/lib/model";

export default async function Series({ searchParams }: { searchParams: Promise<Record<string, string | string[] | undefined>> }) {
  const params = await searchParams;
  const state = stateFrom(params);
  const summary = await loadSummary(state.benchmark);
  if (!summary) {
    return (<><TopBar state={state} current="/series" /><div className="page"><main className="content"><div className="panel"><span className="muted">The Python side is not running.</span></div></main></div></>);
  }
  const lists = await Promise.all(summary.datasets.map(async (d) => [d, (await loadSeriesList(state.benchmark, d)) ?? []] as const));
  const seriesByDataset: Record<string, SeriesMeta[]> = Object.fromEntries(lists.filter(([, l]) => l.length));
  const pin = param(params, "pin");
  const top = param(params, "top");
  return (
    <>
      <TopBar state={state} current="/series" />
      <div className="page">
        <SeriesView state={state} summary={summary} initialFile={param(params, "s")} initialPins={pin ? pin.split(",").filter(Boolean) : []}
          initialTop={top !== null ? Number(top) : null} seriesByDataset={seriesByDataset} />
      </div>
    </>
  );
}
