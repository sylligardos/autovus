import { TopBar } from "@/components/TopBar";
import { SpeedView } from "@/components/SpeedView";
import { loadTables } from "@/lib/tables";
import { compact } from "@/lib/compact";
import { stateFrom } from "@/lib/model";

export default async function Speed({ searchParams }: { searchParams: Promise<Record<string, string | string[] | undefined>> }) {
  const state = stateFrom(await searchParams);
  const t = await loadTables(state.benchmark);
  return (
    <>
      <TopBar state={state} current="/speed" />
      <div className="page">
        {t ? <SpeedView state={state} tables={compact(t)} />
          : <main className="content"><div className="panel"><span className="muted">No tables yet. Run <code>conda run -n autovus python app/export.py</code>.</span></div></main>}
      </div>
    </>
  );
}
