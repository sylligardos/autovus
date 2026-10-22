import { TopBar } from "@/components/TopBar";
import { CasesView } from "@/components/CasesView";
import { loadTables } from "@/lib/tables";
import { compact } from "@/lib/compact";
import { param, stateFrom } from "@/lib/model";

export default async function Cases({ searchParams }: { searchParams: Promise<Record<string, string | string[] | undefined>> }) {
  const params = await searchParams;
  const state = stateFrom(params);
  const t = await loadTables(state.benchmark);
  return (
    <>
      <TopBar state={state} current="/cases" />
      <div className="page">
        {t ? (
          <CasesView state={state} tables={compact(t)}
            initial={{ x: param(params, "x") ?? undefined, y: param(params, "y") ?? undefined, d: param(params, "d") ?? undefined }} />
        ) : (
          <main className="content"><div className="panel"><span className="muted">No tables yet. Run <code>conda run -n autovus python app/export.py</code>.</span></div></main>
        )}
      </div>
    </>
  );
}
