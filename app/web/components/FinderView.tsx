"use client";
// The Finder: where the whole-series AutoVUS and the per-anomaly AutoVUS
// disagree. Datasets first, then the series of one dataset; the top pairs
// of each case on the side. A series opens on the Local page.

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import { loadLocalTable } from "@/lib/api";
import { href, type State } from "@/lib/model";
import { datasetOf, seriesName, type PairsTable } from "@/lib/local";
import { useRemembered } from "@/lib/remember";

type Mode = "mean" | "worst";
const LABEL: Record<Mode, string> = { mean: "AutoVUS vs mean per-anomaly AutoVUS", worst: "AutoVUS vs lowest per-anomaly AutoVUS" };
const signed = (g: number) => (g >= 0 ? "+" : "−") + Math.abs(g).toFixed(2);
const barColor = (g: number) => (g >= 0 ? "var(--late)" : "var(--accent)");

export function FinderView({ state }: { state: State }) {
  const b = state.benchmark;
  const [t, setT] = useState<PairsTable | null>(null);
  const [error, setError] = useState(false);
  const [mode, setMode] = useRemembered<Mode>(`finder:${b}:mode`, "mean");
  const [dataset, setDataset] = useRemembered<string | null>(`finder:${b}:dataset`, null);
  const [off, setOff] = useState<number[]>([]);

  useEffect(() => {
    let live = true;
    loadLocalTable<PairsTable>(b, "pairs").then((x) => { if (live) { if (x) setT(x); else setError(true); } });
    return () => { live = false; };
  }, [b]);

  const ds = useMemo(() => (t ? t.series.map(datasetOf) : []), [t]);
  const view = useMemo(() => {
    if (!t) return null;
    const offSet = new Set(off);
    const gap = (i: number, m: Mode) => t.a[i] - (m === "mean" ? t.m[i] : t.n[i]);
    const rank = (g: number, m: Mode) => (m === "mean" ? Math.abs(g) : g);
    const keep = (i: number, m: Mode) => !offSet.has(t.d[i]) && (m === "mean" || t.K[i] > 1) && (!dataset || ds[t.s[i]] === dataset);
    const N = t.s.length;
    const acc = new Map<string | number, { sum: number; n: number; K: number; series: Set<number>; best: { d: number; g: number } | null }>();
    for (let i = 0; i < N; i++) {
      if (!keep(i, mode)) continue;
      const key = dataset ? t.s[i] : ds[t.s[i]];
      const g = gap(i, mode);
      const a = acc.get(key) ?? { sum: 0, n: 0, K: t.K[i], series: new Set<number>(), best: null };
      a.sum += g; a.n += 1; a.series.add(t.s[i]);
      if (!a.best || rank(g, mode) > rank(a.best.g, mode)) a.best = { d: t.d[i], g };
      acc.set(key, a);
    }
    const rows = [...acc.entries()].map(([key, a]) => ({ key, mean: a.sum / a.n, K: a.K, nSeries: a.series.size, best: a.best }))
      .sort((x, y) => rank(y.mean, mode) - rank(x.mean, mode));
    const mx = Math.max(1e-9, ...rows.map((r) => Math.abs(r.mean)));
    const top = (m: Mode) => {
      const idx: number[] = [];
      for (let i = 0; i < N; i++) if (keep(i, m)) idx.push(i);
      return idx.sort((x, y) => rank(gap(y, m), m) - rank(gap(x, m), m)).slice(0, 5)
        .map((i) => ({ file: t.series[t.s[i]], det: t.detectors[t.d[i]], g: gap(i, m) }));
    };
    return { rows, mx, tops: [{ m: "mean" as Mode, rows: top("mean") }, { m: "worst" as Mode, rows: top("worst") }] };
  }, [t, ds, off, mode, dataset]);

  const localHref = (file: string, det?: string) => href("/local", state, det ? { s: file, det } : { s: file });

  return (
    <>
      <aside className="rail" aria-label="Controls">
        <h2 style={{ fontSize: 15, lineHeight: 1.3 }}>Global vs Local evaluation metric</h2>
        <div className="field">
          <span>Compare AutoVUS to</span>
          <div className="seg" role="group" aria-label="Compare AutoVUS to" style={{ gridTemplateColumns: "minmax(0, 1fr)" }}>
            <button type="button" aria-pressed={mode === "mean"} onClick={() => setMode("mean")} style={{ height: 34 }}>Mean per-anomaly AutoVUS</button>
            <button type="button" aria-pressed={mode === "worst"} onClick={() => setMode("worst")} style={{ height: 34 }}>Lowest per-anomaly AutoVUS</button>
          </div>
        </div>
        <div className="field">
          <span style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>Detectors
            <span style={{ display: "flex", gap: 4, textTransform: "none", letterSpacing: 0 }}>
              <button type="button" className="btn" style={{ height: 24, padding: "0 8px", fontSize: 11 }} onClick={() => setOff([])}>All</button>
              <button type="button" className="btn" style={{ height: 24, padding: "0 8px", fontSize: 11 }} onClick={() => setOff(t ? t.detectors.map((_, i) => i) : [])}>None</button>
            </span>
          </span>
          <div style={{ maxHeight: 360, overflowY: "auto", border: "1px solid var(--border-soft)", borderRadius: 6, padding: "4px 8px" }}>
            {(t?.detectors ?? []).map((name, i) => (
              <label key={name} style={{ display: "flex", alignItems: "center", gap: 8, minHeight: 26, fontSize: 12, cursor: "pointer" }}>
                <input type="checkbox" checked={!off.includes(i)} onChange={() => setOff((o) => (o.includes(i) ? o.filter((x) => x !== i) : [...o, i]))} style={{ margin: 0 }} />
                {name}
              </label>
            ))}
          </div>
        </div>
      </aside>

      <main className="content" style={{ flexDirection: "row", flexWrap: "wrap", alignItems: "flex-start", gap: 20 }}>
        <section className="panel" aria-label="Ranking" style={{ flex: "999 1 520px", minWidth: 0, overflowX: "auto" }}>
          <nav aria-label="Scope" style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 14, paddingBottom: 8, borderBottom: "1px solid var(--border-soft)" }}>
            <button type="button" onClick={() => setDataset(null)} style={{ padding: 0, border: 0, background: "transparent", cursor: "pointer", fontSize: 14, fontWeight: 600, color: dataset ? "var(--accent)" : "var(--ink)" }}>All datasets</button>
            {dataset && (<><span className="muted" aria-hidden="true">›</span><span style={{ fontWeight: 600 }}>{dataset}</span></>)}
          </nav>
          {error && <span className="muted">No local tables. Run <code>conda run -n autovus python app/export_local.py</code>.</span>}
          {!view && !error && <div style={{ height: 400 }} />}
          {view && !dataset && (
            <table className="table">
              <thead><tr><th>Dataset</th><th className="r">Time series</th><th style={{ width: 320 }}>{LABEL[mode]}</th></tr></thead>
              <tbody>
                {view.rows.map((r) => (
                  <tr key={String(r.key)} onClick={() => setDataset(String(r.key))}>
                    <td className="sans" style={{ fontWeight: 600 }}>{String(r.key)}</td>
                    <td className="r mono muted">{r.nSeries}</td>
                    <td><Bar g={r.mean} mx={view.mx} /></td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
          {view && dataset && (
            <table className="table">
              <thead><tr><th>Time series</th><th className="r">Anomalies</th><th style={{ width: 300 }}>{LABEL[mode]}</th><th>Detector with largest gap</th></tr></thead>
              <tbody>
                {view.rows.map((r) => {
                  const file = t!.series[Number(r.key)];
                  return (
                    <tr key={String(r.key)}>
                      <td className="mono"><Link href={localHref(file, r.best ? t!.detectors[r.best.d] : undefined)} style={{ textDecoration: "none" }}>{seriesName(file)}</Link></td>
                      <td className="r mono muted">{r.K}</td>
                      <td><Bar g={r.mean} mx={view.mx} /></td>
                      <td className="sans">{r.best ? t!.detectors[r.best.d] : "–"}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          )}
        </section>

        <aside aria-label="Top pairs" style={{ flex: "1 1 300px", maxWidth: 380, minWidth: 260, display: "flex", flexDirection: "column", gap: 16 }}>
          {view?.tops.map((tp) => (
            <div key={tp.m} className="panel" style={{ gap: 0 }}>
              <span className="eyebrow" style={{ marginBottom: 6 }}>Top 5 pairs: {LABEL[tp.m]}</span>
              {tp.rows.map((p) => (
                <Link key={p.file + p.det} href={localHref(p.file, p.det)} style={{ display: "grid", gridTemplateColumns: "minmax(0, 1fr) 52px", gap: 8, alignItems: "center", padding: "6px 0", borderTop: "1px solid var(--border-soft)", textDecoration: "none" }}>
                  <span style={{ display: "flex", flexDirection: "column", minWidth: 0 }}>
                    <span className="mono" style={{ fontSize: 11, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>{seriesName(p.file)}</span>
                    <span style={{ fontSize: 12, fontWeight: 600 }}>{p.det}</span>
                  </span>
                  <span className="mono" style={{ textAlign: "right", fontSize: 13 }}>{signed(p.g)}</span>
                </Link>
              ))}
            </div>
          ))}
        </aside>
      </main>
    </>
  );
}

function Bar({ g, mx }: { g: number; mx: number }) {
  return (
    <span style={{ display: "flex", alignItems: "center", gap: 10 }}>
      <span className="mono" style={{ width: 48, textAlign: "right", fontSize: 13 }}>{signed(g)}</span>
      <span style={{ flex: 1, height: 8, background: "var(--border-soft)", borderRadius: 2, position: "relative" }}>
        <span style={{ position: "absolute", left: 0, top: 0, bottom: 0, width: `${(Math.abs(g) / mx) * 100}%`, background: barColor(g), borderRadius: 2 }} />
      </span>
    </span>
  );
}
