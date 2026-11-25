"use client";
// Where two scores of a series are far apart. On Cases: the whole-series AutoVUS
// (global) against the per-anomaly AutoVUS (local, mean or lowest). On Profiles:
// the stitched score against the best detector. The comparison is the `mode` prop.
// Datasets first, then the series of one dataset, each with its largest gap; every
// pair as a point of the scatter on top; the top pairs on the side, 20 at a time.
// A series opens in Explore.

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useMemo, useState } from "react";
import { loadLocalTable } from "@/lib/api";
import { href, type State } from "@/lib/model";
import { datasetOf, seriesName, type PairsTable, type StitchedTable } from "@/lib/local";
import { useRemembered } from "@/lib/remember";
import { GapScatter } from "./GapScatter";

export type Mode = "mean" | "worst" | "stitched";
const AXES: Record<Mode, { x: string; y: string }> = {
  mean: { x: "Mean AutoVUS Local", y: "AutoVUS Global" }, worst: { x: "Lowest AutoVUS Local", y: "AutoVUS Global" },
  stitched: { x: "Best detector AutoVUS", y: "Stitched score AutoVUS" },
};
const LABEL: Record<Mode, string> = { mean: "AutoVUS Global minus mean AutoVUS Local", worst: "AutoVUS Global minus lowest AutoVUS Local", stitched: "Stitched score AutoVUS minus best detector AutoVUS" };
const STEP = 20;   // top pairs shown at first, and added by "Show more"
const signed = (g: number) => (g >= 0 ? "+" : "−") + Math.abs(g).toFixed(2);
const barColor = (g: number) => (g >= 0 ? "var(--late)" : "var(--accent)");

export function FinderView({ state, mode }: { state: State; mode: Mode }) {
  const b = state.benchmark;
  const [t, setT] = useState<PairsTable | null>(null);
  const [error, setError] = useState(false);
  const [st, setSt] = useState<StitchedTable | null>(null);
  const [dataset, setDataset] = useRemembered<string | null>(`finder:${b}:dataset`, null);
  const [off, setOff] = useState<number[]>([]);
  const [more, setMore] = useState<{ key: string; n: number }>({ key: "", n: STEP });
  const [mark, setMark] = useState<number[]>([]);   // cases to mark on the plot, as indexes of view.top
  const router = useRouter();

  useEffect(() => {
    let live = true;
    setError(false);
    loadLocalTable<PairsTable>(b, "pairs", state.L).then((x) => { if (live) { if (x) setT(x); else setError(true); } });
    loadLocalTable<StitchedTable>(b, "stitched", state.L).then((x) => { if (live) setSt(x); });
    return () => { live = false; };
  }, [b, state.L]);

  const ds = useMemo(() => (t ? t.series.map(datasetOf) : []), [t]);
  const view = useMemo(() => {
    if (!t) return null;
    const offSet = new Set(off);
    const rank = (g: number) => (mode === "mean" ? Math.abs(g) : g);
    // Series with one anomaly are left out everywhere: their local and global scores are equal.
    const keep = (i: number) => !offSet.has(t.d[i]) && t.K[i] > 1 && (!dataset || ds[t.s[i]] === dataset);
    const N = t.s.length;
    // One case per pair: x the local score, y the global one. Stitched mode, one case per series:
    // x the best chosen detector's AutoVUS Global, y the stitched score's AutoVUS.
    let cases: { s: number; d: number; K: number; x: number; y: number; g: number }[] = [];
    if (mode === "stitched") {
      const top1 = new Map<number, { d: number; a: number; K: number }>();
      if (st) for (let i = 0; i < N; i++) {
        if (!keep(i) || st.score[t.series[t.s[i]]] === undefined) continue;
        const cur = top1.get(t.s[i]);
        if (!cur || t.a[i] > cur.a) top1.set(t.s[i], { d: t.d[i], a: t.a[i], K: t.K[i] });
      }
      cases = [...top1.entries()].map(([s, x]) => ({ s, d: x.d, K: x.K, x: x.a, y: st!.score[t.series[s]], g: st!.score[t.series[s]] - x.a }));
    } else for (let i = 0; i < N; i++) {
      if (!keep(i)) continue;
      const x = mode === "mean" ? t.m[i] : t.n[i];
      cases.push({ s: t.s[i], d: t.d[i], K: t.K[i], x, y: t.a[i], g: t.a[i] - x });
    }
    cases.sort((p, q) => rank(q.g) - rank(p.g));
    // A dataset row is the mean gap of its cases; a series row is its largest gap, so the first rows are the first cases.
    // idx: the row's cases on the plot, every case of a dataset, the largest-gap one of a series.
    const acc = new Map<string | number, { sum: number; n: number; K: number; series: Set<number>; best: { d: number; g: number }; idx: number[] }>();
    cases.forEach((c, i) => {
      const key = dataset ? c.s : ds[c.s];
      const a = acc.get(key);
      if (a) { a.sum += c.g; a.n += 1; a.series.add(c.s); if (!dataset) a.idx.push(i); }
      else acc.set(key, { sum: c.g, n: 1, K: c.K, series: new Set([c.s]), best: { d: c.d, g: c.g }, idx: [i] });
    });
    const rows = [...acc.entries()].map(([key, a]) => ({ key, g: dataset ? a.best.g : a.sum / a.n, K: a.K, nSeries: a.series.size, best: a.best, idx: a.idx }))
      .sort((x, y) => rank(y.g) - rank(x.g));
    const mx = Math.max(1e-9, ...rows.map((r) => Math.abs(r.g)));
    const top = cases.map((c) => ({ file: t.series[c.s], det: t.detectors[c.d], x: c.x, y: c.y, g: c.g, title: seriesName(t.series[c.s]), sub: t.detectors[c.d] }));
    return { rows, mx, top };
  }, [t, st, ds, off, mode, dataset]);

  // "Show more" count, back to the first 20 when the case, the dataset or the detectors change.
  const moreKey = `${mode}|${dataset ?? ""}|${off.join(",")}`;
  const shown = more.key === moreKey ? more.n : STEP;

  // A stitched case opens on Best coverage, with the best single detector pinned: one click from "Stitch winners".
  const localHref = (file: string, det?: string) => href("/local", state, { s: file, ...(det ? { det } : {}), ...(mode === "stitched" ? { show: "best" } : {}) });

  return (
    <>
      <aside className="rail" aria-label="Controls">
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
        <div style={{ flex: "999 1 520px", minWidth: 0, display: "flex", flexDirection: "column", gap: 16 }}>
        {view && (
          <section className="panel" aria-label="Cases">
            <GapScatter points={view.top} xlabel={AXES[mode].x} ylabel={AXES[mode].y} glabel={LABEL[mode]} mark={mark}
              onPick={(i) => router.push(localHref(view.top[i].file, view.top[i].det))} />
          </section>
        )}
        <section className="panel" aria-label="Ranking" style={{ minWidth: 0, overflowX: "auto" }}>
          <nav aria-label="Scope" style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 14, paddingBottom: 8, borderBottom: "1px solid var(--border-soft)" }}>
            <button type="button" onClick={() => setDataset(null)} style={{ padding: 0, border: 0, background: "transparent", cursor: "pointer", fontSize: 14, fontWeight: 600, color: dataset ? "var(--accent)" : "var(--ink)" }}>All datasets</button>
            {dataset && (<><span className="muted" aria-hidden="true">›</span><span style={{ fontWeight: 600 }}>{dataset}</span></>)}
          </nav>
          {error && <span className="muted">No local tables. Run <code>conda run -n autovus python app/export_gear.py --L {state.L}</code>.</span>}
          {!view && !error && <div style={{ height: 400 }} />}
          <div style={{ maxHeight: "max(240px, calc(100vh - 684px))", overflowY: "auto" }}>
          {view && !dataset && (
            <table className="table">
              <thead><tr><th>Dataset</th><th className="r">Time series</th><th style={{ width: 320 }}>{LABEL[mode]}</th></tr></thead>
              <tbody>
                {view.rows.map((r) => (
                  <tr key={String(r.key)} onClick={() => { setMark([]); setDataset(String(r.key)); }} onMouseEnter={() => setMark(r.idx)} onMouseLeave={() => setMark([])}>
                    <td className="sans" style={{ fontWeight: 600 }}>{String(r.key)}</td>
                    <td className="r mono muted">{r.nSeries}</td>
                    <td><Bar g={r.g} mx={view.mx} /></td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
          {view && dataset && (
            <table className="table">
              <thead><tr><th>Time series</th><th className="r">Anomalies</th><th style={{ width: 190, whiteSpace: "normal" }}>{LABEL[mode]}</th><th>{mode === "stitched" ? "Best detector" : "Detector"}</th></tr></thead>
              <tbody>
                {view.rows.map((r) => {
                  const file = t!.series[Number(r.key)];
                  return (
                    <tr key={String(r.key)} onClick={() => router.push(localHref(file, t!.detectors[r.best.d]))} onMouseEnter={() => setMark(r.idx)} onMouseLeave={() => setMark([])}>
                      <td className="mono" style={{ maxWidth: 230, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}><Link href={localHref(file, t!.detectors[r.best.d])} title={seriesName(file)} style={{ textDecoration: "none" }}>{seriesName(file)}</Link></td>
                      <td className="r mono muted">{r.K}</td>
                      <td><Bar g={r.g} mx={view.mx} /></td>
                      <td className="sans">{t!.detectors[r.best.d]}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          )}
          </div>
        </section>
        </div>

        <aside aria-label="Top pairs" style={{ flex: "1 1 300px", maxWidth: 380, minWidth: 260, display: "flex", flexDirection: "column", gap: 16 }}>
          {view && (
            <div className="panel" style={{ gap: 0, maxHeight: "max(240px, calc(100vh - 162px))", overflowY: "auto" }}>
              <span className="eyebrow" style={{ marginBottom: 6 }}>Top {Math.min(shown, view.top.length)} {mode === "stitched" ? "series" : "pairs"}: {LABEL[mode]}</span>
              {view.top.slice(0, shown).map((p, i) => (
                <Link key={p.file + p.det} href={localHref(p.file, p.det)} onMouseEnter={() => setMark([i])} onMouseLeave={() => setMark([])} style={{ display: "grid", gridTemplateColumns: "minmax(0, 1fr) 52px", gap: 8, alignItems: "center", padding: "6px 0", borderTop: "1px solid var(--border-soft)", textDecoration: "none" }}>
                  <span style={{ display: "flex", flexDirection: "column", minWidth: 0 }}>
                    <span className="mono" style={{ fontSize: 11, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>{seriesName(p.file)}</span>
                    <span style={{ fontSize: 12, fontWeight: 600 }}>{p.det}</span>
                  </span>
                  <span className="mono" style={{ textAlign: "right", fontSize: 13 }}>{signed(p.g)}</span>
                </Link>
              ))}
              {view.top.length > shown && (
                <button type="button" className="btn" style={{ marginTop: 8 }} onClick={() => setMore({ key: moreKey, n: shown + STEP })}>Show more</button>
              )}
            </div>
          )}
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
