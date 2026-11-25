import { TopBar } from "@/components/TopBar";
import { RankingsView } from "@/components/RankingsView";
import { loadTables } from "@/lib/tables";
import { compact } from "@/lib/compact";
import { param, stateFrom } from "@/lib/model";

export default async function Rankings({ searchParams }: { searchParams: Promise<Record<string, string | string[] | undefined>> }) {
  const params = await searchParams;
  const state = stateFrom(params);
  const t = await loadTables(state.benchmark, `L${state.L}/scores.json`);
  return (
    <>
      <TopBar state={state} current="/rankings" />
      <div className="page">
        <p className="page-line" style={{ flex: "0 0 100%" }}>Which detectors change rank when the metric changes? <span>Two metrics side by side. Each curve between them is a detector moving up or down.</span></p>
        {t ? <RankingsView state={state} tables={compact(t)} initial={{ a: param(params, "a") ?? undefined, b: param(params, "bm") ?? undefined }} />
          : <main className="content"><div className="panel"><span className="muted">No tables yet. Run <code>conda run -n autovus python app/export_gear.py --L {state.L}</code>.</span></div></main>}
      </div>
    </>
  );
}
