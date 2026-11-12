"use client";
// Suspicious anomalies: the ones most detectors score among their worst in a
// series while they catch the others. Datasets, then the flagged anomalies of
// one dataset; a click shows the anomaly, its label and where the detectors
// fire. "Show in Local" opens the series zoomed on it.

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import { loadLocalTable } from "@/lib/api";
import { href, type State } from "@/lib/model";
import { seriesName, type OddItem, type OddTable } from "@/lib/local";
import { useRemembered } from "@/lib/remember";

const f2 = (v: number) => v.toFixed(2);
const line = (a: number[]) => a.map((v, i) => `${((i / Math.max(1, a.length - 1)) * 1000).toFixed(1)},${(96 - v * 92).toFixed(1)}`).join(" ");

export function SuspiciousView({ state }: { state: State }) {
  const b = state.benchmark;
  const [t, setT] = useState<OddTable | null>(null);
  const [error, setError] = useState(false);
  const [dataset, setDataset] = useRemembered<string | null>(`suspicious:${b}:dataset`, null);
  const [sel, setSel] = useState<number | null>(null);

  useEffect(() => {
    let live = true;
    loadLocalTable<OddTable>(b, "odd").then((x) => { if (live) { if (x) setT(x); else setError(true); } });
    return () => { live = false; };
  }, [b]);

  const items = useMemo(() => (t ? t.items.map((x, id) => ({ ...x, id, gap: x.others - x.this })) : []), [t]);
  const scope = items.filter((x) => !dataset || x.dataset === dataset);
  const top = [...scope].sort((x, y) => y.gap - x.gap).slice(0, 10);
  const picked = sel !== null ? items[sel] : null;
  const open = (x: { id: number; dataset: string }) => { setDataset(x.dataset); setSel(x.id); };

  const dsRows = useMemo(() => {
    if (!t) return [];
    const acc = new Map<string, { n: number; series: Set<string> }>();
    for (const x of items) { const a = acc.get(x.dataset) ?? { n: 0, series: new Set<string>() }; a.n += 1; a.series.add(x.file); acc.set(x.dataset, a); }
    return Object.keys(t.tested).map((name) => ({ name, series: t.tested[name].series, n: acc.get(name)?.n ?? 0, withFlag: acc.get(name)?.series.size ?? 0 }))
      .sort((x, y) => y.n - x.n);
  }, [t, items]);
  const mx = Math.max(1, ...dsRows.map((r) => r.n));

  return (
    <>
      <main className="content">
        <nav aria-label="Scope" style={{ display: "flex", flexWrap: "wrap", alignItems: "center", gap: 8, fontSize: 14 }}>
          <button type="button" onClick={() => { setDataset(null); setSel(null); }} style={{ padding: 0, border: 0, background: "transparent", cursor: "pointer", fontSize: 14, fontWeight: 600, color: dataset ? "var(--accent)" : "var(--ink)" }}>All datasets</button>
          {dataset && (<><span className="muted" aria-hidden="true">›</span><span style={{ fontWeight: 600 }}>{dataset}</span></>)}
        </nav>
        {error && <div className="panel"><span className="muted">No local tables. Run <code>conda run -n autovus python app/export_local.py</code>.</span></div>}
        {!t && !error && <div className="panel" style={{ height: 420 }} />}

        {picked && <Picked state={state} x={picked} />}

        {t && !dataset && (
          <section className="panel" aria-label="Datasets">
            <table className="table">
              <thead><tr><th>Dataset</th><th className="r">Time series</th><th className="r">With a flag</th><th style={{ width: 240 }}>Suspicious anomalies</th></tr></thead>
              <tbody>
                {dsRows.map((r) => (
                  <tr key={r.name} onClick={() => { setDataset(r.name); setSel(null); }}>
                    <td className="sans" style={{ fontWeight: 600 }}>{r.name}</td>
                    <td className="r mono muted">{r.series}</td>
                    <td className="r mono">{r.withFlag}</td>
                    <td>
                      <span style={{ display: "flex", alignItems: "center", gap: 10 }}>
                        <span className="mono" style={{ width: 40, textAlign: "right" }}>{r.n}</span>
                        <span style={{ flex: 1, height: 8, background: "var(--border-soft)", borderRadius: 2, position: "relative" }}><span style={{ position: "absolute", left: 0, top: 0, bottom: 0, width: `${(r.n / mx) * 100}%`, background: "var(--late)", borderRadius: 2 }} /></span>
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </section>
        )}

        {t && dataset && (
          <section className="panel" aria-label="Suspicious anomalies" style={{ overflowX: "auto" }}>
            <table className="table">
              <thead><tr><th>Time series</th><th>Anomaly</th><th className="r">Length</th><th className="r">Worst for</th><th className="r">AutoVUS: it vs others</th><th className="r">q</th></tr></thead>
              <tbody>
                {[...scope].sort((x, y) => y.gap - x.gap).map((x) => (
                  <tr key={x.id} onClick={() => setSel(x.id)} className={sel === x.id ? "picked" : undefined}>
                    <td className="mono">{seriesName(x.file)}</td>
                    <td className="sans" style={{ fontWeight: 600 }}>A{x.k + 1}</td>
                    <td className="r mono muted">{x.len}</td>
                    <td className="r mono">{x.worst} / {x.D}</td>
                    <td className="r mono"><span style={{ color: "var(--late)", fontWeight: 500 }}>{f2(x.this)}</span> vs {f2(x.others)}</td>
                    <td className="r mono muted">{x.q.toExponential(0)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </section>
        )}
      </main>

      <aside className="panel" aria-label="Strongest cases" style={{ flex: "1 1 300px", maxWidth: 380, minWidth: 260, alignSelf: "flex-start", gap: 0 }}>
        <span className="eyebrow" style={{ marginBottom: 6 }}>Top 10</span>
        {top.map((x) => (
          <button key={x.id} type="button" onClick={() => open(x)} style={{ display: "grid", gridTemplateColumns: "minmax(0, 1fr) 90px", gap: 8, alignItems: "center", width: "100%", padding: "6px 0", border: 0, borderTop: "1px solid var(--border-soft)", background: "transparent", cursor: "pointer", textAlign: "left", color: "var(--ink)" }}>
            <span style={{ display: "flex", flexDirection: "column", minWidth: 0 }}>
              <span className="mono" style={{ fontSize: 11, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>{seriesName(x.file)}</span>
              <span style={{ fontSize: 12, fontWeight: 600 }}>A{x.k + 1}</span>
            </span>
            <span className="mono" style={{ textAlign: "right", fontSize: 12 }}><span style={{ color: "var(--late)" }}>{f2(x.this)}</span> vs {f2(x.others)}</span>
          </button>
        ))}
      </aside>
    </>
  );
}

function Picked({ state, x }: { state: State; x: OddItem }) {
  const span = x.hi - x.lo;
  const left = ((x.start - x.lo) / span) * 100, w = ((x.end - x.start + 1) / span) * 100;
  const band = (opacity: number) => <div style={{ position: "absolute", top: 0, bottom: 0, left: `${left}%`, width: `${w}%`, minWidth: 3, background: "var(--label)", opacity }} />;
  return (
    <section className="panel panel-accent" aria-label="Selected anomaly">
      <div className="panel-head" style={{ alignItems: "center" }}>
        <span className="mono" style={{ fontSize: 13 }}>{seriesName(x.file)}</span>
        <span style={{ fontSize: 15, fontWeight: 700 }}>A{x.k + 1}</span>
        <span className="muted" style={{ fontSize: 13 }}>worst for {x.worst} of {x.D} detectors · AutoVUS {f2(x.this)} vs {f2(x.others)} on the other {x.K - 1}</span>
        <Link href={href("/local", state, { s: x.file, v: `${x.lo},${x.hi}` })} className="btn btn-primary" style={{ marginLeft: "auto", display: "inline-flex", alignItems: "center", textDecoration: "none" }}>Show in Local</Link>
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "130px minmax(0, 1fr)", gap: "0 14px", alignItems: "center" }}>
        <span className="muted" style={{ fontSize: 12, fontWeight: 600 }}>Series</span>
        <div style={{ position: "relative", height: 110, background: "var(--subtle)", borderRadius: 4 }}>
          {band(0.55)}
          <svg viewBox="0 0 1000 100" preserveAspectRatio="none" aria-hidden="true" style={{ position: "absolute", inset: 0, width: "100%", height: "100%" }}><polyline points={line(x.values)} fill="none" stroke="var(--ink)" strokeWidth={1.5} vectorEffect="non-scaling-stroke" /></svg>
        </div>
        <span className="muted" style={{ fontSize: 12, fontWeight: 600 }}>Median detector</span>
        <div style={{ position: "relative", height: 80, marginTop: 6, background: "var(--subtle)", borderRadius: 4 }}>
          {band(0.35)}
          <svg viewBox="0 0 1000 100" preserveAspectRatio="none" aria-hidden="true" style={{ position: "absolute", inset: 0, width: "100%", height: "100%" }}><polyline points={line(x.score)} fill="none" stroke="var(--accent)" strokeWidth={1.5} vectorEffect="non-scaling-stroke" /></svg>
        </div>
      </div>
    </section>
  );
}
