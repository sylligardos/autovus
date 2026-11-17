"use client";
// The Local page: one series, its anomalies, and per detector the local
// AutoVUS of every anomaly, as numbers on the score lanes (Scores) or as
// cells (Heat map). Everything at L = 20. Drag to zoom, Back to
// step out, double click for all.

import { Fragment, useCallback, useEffect, useMemo, useState } from "react";
import { loadAucPr, loadCurves, loadLocalSeries, loadOracle, loadScoreDoc, loadSelect, loadSeriesDoc, type OracleDoc, type OracleNorm } from "@/lib/api";
import { TYPE_COLOR, f2, fmt, type Anomaly, type CurvesDoc, type ScoreDoc, type SeriesDoc, type SeriesMeta, type State } from "@/lib/model";
import { GRADIENT_CSS, MAX_SCORED, chipColors, gradient, oracle, seriesName, spreadMarkers, type LocalSeries } from "@/lib/local";
import { useRemembered } from "@/lib/remember";
import { Drawer } from "./Drawer";
import { Lane, type Band, type Zone } from "./Lane";
import { NotesButton } from "./NotesButton";

type View = [number, number];
type Mode = "top" | "best" | "all";
type Sort = "ap" | "vus" | "auto" | "local";         // the column the detectors are ordered by
const SORT_KEY = { ap: "ap", vus: "vv", auto: "gv", local: "lv" } as const;
const L = 20;                                      // the one buffer length of the demo
const COLS = "150px minmax(0, 1fr) 56px 56px 64px 56px";
const NORMS: [OracleNorm, string][] = [["minmax", "Stitched score"], ["median", "Stitched score (median)"]];
type Oracles = Partial<Record<OracleNorm, OracleDoc>>;
const GAP = 4;                                     // percent between two anomaly markers

export function LocalView({ state, seriesByDataset, initialFile, initialDet, initialView, initialPick = null }: {
  state: State; seriesByDataset: Record<string, SeriesMeta[]>;
  initialFile: string | null; initialDet: string[] | null; initialView: View | null;
  initialPick?: number | null;                     // an anomaly whose buffer is explained on arrival, for the first pinned detector
}) {
  const b = state.benchmark;
  const datasets = Object.keys(seriesByDataset);
  const k = `local:${b}:`;
  const firstFile = seriesByDataset.YAHOO?.find((s) => s.file.startsWith("682_"))?.file ?? seriesByDataset[datasets[0]]?.[0]?.file ?? "";
  const [file, setFile] = useRemembered(k + "file", firstFile, initialFile);
  const dataset = useMemo(() => datasets.find((d) => seriesByDataset[d].some((s) => s.file === file)) ?? datasets[0], [datasets, seriesByDataset, file]);
  const [tab, setTab] = useRemembered<"scores" | "map">(k + "tab", "scores");
  const [mode, setMode] = useRemembered<Mode>(k + "mode", "top");
  const [topK, setTopK] = useRemembered(k + "top", 5, initialDet ? 0 : null);
  const [sort, setSort] = useRemembered<Sort>(k + "sort", "auto");
  const [pins, setPins] = useState<string[]>(initialDet ?? []);
  const [data, setData] = useState<LocalSeries | null>(null);
  const [aucpr, setAucpr] = useState<Record<string, number | null>>({});
  const [doc, setDoc] = useState<SeriesDoc | null>(null);
  const [scores, setScores] = useState<Record<string, ScoreDoc>>({});
  const [view, setView] = useState<View | null>(initialView);
  const [hist, setHist] = useState<(View | null)[]>([]);
  const [zoomed, setZoomed] = useState<{ doc: SeriesDoc | null; scores: Record<string, ScoreDoc> }>({ doc: null, scores: {} });
  const [hk, setHk] = useState(-1);
  const [error, setError] = useState<string | null>(null);
  const [firstView, setFirstView] = useState(initialView !== null || initialPick !== null);
  const [orc, setOrc] = useState<{ docs: Oracles; zoom: Oracles; busy: boolean }>({ docs: {}, zoom: {}, busy: false });
  const [picked, setPicked] = useState<{ detector: string; k: number } | null>(initialPick !== null && initialDet?.[0] ? { detector: initialDet[0], k: initialPick } : null);
  const [why, setWhy] = useState<{ a: Anomaly; curves: CurvesDoc } | null>(null);

  // The series: its local table and its values.
  useEffect(() => {
    let live = true;
    setData(null); setDoc(null); setScores({}); setZoomed({ doc: null, scores: {} }); setHk(-1); setError(null); setOrc({ docs: {}, zoom: {}, busy: false });
    if (firstView) setFirstView(false); else { setView(null); setHist([]); setPicked(null); }
    loadLocalSeries<LocalSeries>(b, file).then((d) => { if (live) { if (d) setData(d); else setError("No local table for this series."); } });
    loadSeriesDoc(b, file).then((d) => { if (live && d) setDoc(d); });
    setAucpr({});
    loadAucPr(b, file).then((d) => { if (live && d) setAucpr(d); });
    return () => { live = false; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [b, file]);

  // The URL follows the series, the pinned detectors and the zoom.
  useEffect(() => {
    const q = new URLSearchParams(window.location.search);
    q.set("s", file);
    if (pins.length) q.set("det", pins.join(",")); else q.delete("det");
    if (view) q.set("v", `${view[0]},${view[1]}`); else q.delete("v");
    window.history.replaceState(null, "", `${window.location.pathname}?${q}`);
  }, [file, pins, view]);

  const local = data ? data.auto : null;
  const rows = useMemo(() => {
    if (!data || !local) return [];
    const r = data.detectors.map((name, j) => {
      const loc = local[j];
      return { name, loc, ap: aucpr[name] ?? null, gv: data.autoG[j], vv: data.vusG[j][Math.min(L, data.vusG[j].length - 1)], lv: loc.reduce((s, x) => s + x, 0) / Math.max(1, loc.length) };
    });
    const key = SORT_KEY[sort] ?? "gv";
    return r.sort((x, y) => (y[key] ?? -1) - (x[key] ?? -1));
  }, [data, local, sort, aucpr]);

  const best = useMemo(() => {
    if (!data || !local || !data.anomalies.length) return { set: new Set<string>(), winner: [] as string[] };
    const o = oracle(local);
    return { set: new Set(o.set.map((d) => data.detectors[d])), winner: o.winner.map((d) => data.detectors[d]) };
  }, [data, local]);

  const shown = useMemo(() => {
    let names: string[];
    if (mode === "top") names = rows.slice(0, topK).map((r) => r.name);
    else if (mode === "best") names = rows.filter((r) => best.set.has(r.name)).map((r) => r.name);
    else names = rows.map((r) => r.name);
    const set = new Set(names);
    for (const p of pins) set.add(p);
    return rows.filter((r) => set.has(r.name));
  }, [rows, mode, topK, best, pins]);

  // The stitched score: region k (midpoint split) takes the score of anomaly k's winner; metrics recomputed on it.
  const stitch = () => {
    if (!best.winner.length) return;
    setOrc({ docs: {}, zoom: {}, busy: true });
    Promise.all(NORMS.map(([nm]) => loadOracle(b, file, best.winner, nm))).then((ds) => {
      const docs: Oracles = {};
      ds.forEach((d, i) => { if (d) docs[NORMS[i][0]] = d; });
      setOrc({ docs, zoom: {}, busy: false });
    });
  };
  const oracleRows = useMemo(() => {
    if (mode !== "best") return [];
    return NORMS.flatMap(([nm, name]) => {
      const o = orc.docs[nm];
      if (!o) return [];
      const loc = o.auto;
      return [{ name, norm: nm, loc, ap: o.aucpr, gv: o.autoG, vv: o.vusG[Math.min(L, o.vusG.length - 1)], lv: loc.reduce((s, x) => s + x, 0) / Math.max(1, loc.length) }];
    });
  }, [orc.docs, mode]);
  const lanes: { name: string; norm?: OracleNorm; loc: number[]; ap?: number | null; gv: number; vv: number; lv: number }[] =
    [...oracleRows, ...shown];
  // The best row of each metric among the rows shown.
  const top = useMemo(() => {
    const mx = (f: (r: (typeof lanes)[number]) => number | null | undefined) =>
      lanes.reduce((m, r) => { const v = f(r); return v !== null && v !== undefined && Number.isFinite(v) && v > m ? v : m; }, -Infinity);
    return { ap: mx((r) => r.ap), vv: mx((r) => r.vv), gv: mx((r) => r.gv), lv: mx((r) => r.lv) };
  }, [lanes]);

  // Scores of the shown detectors, once each; finer bins for the zoom window.
  useEffect(() => {
    let live = true;
    for (const r of shown) if (!scores[r.name]) loadScoreDoc(b, file, r.name).then((s) => { if (live && s) setScores((m) => ({ ...m, [r.name]: s })); });
    return () => { live = false; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [shown, b, file]);
  useEffect(() => {
    if (!view) return;
    let live = true;
    const t = setTimeout(() => {
      loadSeriesDoc(b, file, 2000, view).then((d) => { if (live && d) setZoomed((z) => ({ ...z, doc: d })); });
      for (const r of shown) loadScoreDoc(b, file, r.name, 2000, view).then((s) => { if (live && s) setZoomed((z) => ({ ...z, scores: { ...z.scores, [r.name]: s } })); });
      for (const [nm] of NORMS) {
        const o = orc.docs[nm];
        if (o) loadOracle(b, file, o.winners, nm, 2000, view).then((d) => { if (live && d) setOrc((x) => ({ ...x, zoom: { ...x.zoom, [nm]: d } })); });
      }
    }, 200);
    return () => { live = false; clearTimeout(t); };
  }, [view, b, file, shown, orc.docs]);
  // Why this buffer: the clicked anomaly of one detector, scored live.
  useEffect(() => {
    setWhy(null);
    if (!picked) return;
    let live = true;
    Promise.all([loadSelect(b, file, picked.detector, L), loadCurves(b, file, picked.detector, picked.k, L)]).then(([sel, c]) => {
      if (live && sel && c && sel.anomalies[picked.k]) setWhy({ a: sel.anomalies[picked.k], curves: c });
    });
    return () => { live = false; };
  }, [picked, b, file]);
  const pick = (detector: string, kk: number) => setPicked((p) => (p && p.detector === detector && p.k === kk ? null : { detector, k: kk }));
  const covers = (d: { start?: number; end?: number } | null | undefined) => !!(d && view && (d.start ?? 0) <= view[0] && (d.end ?? Infinity) >= view[1]);

  const zoomTo = useCallback((v: View | null) => {
    if (v) { setHist((h) => [...h, view]); setView(v); } else { setView(null); setHist([]); }
  }, [view]);
  const back = () => { const c = hist.slice(); const prev = c.pop() ?? null; setHist(c); setView(prev); };

  const step = (dir: 1 | -1) => {
    const list = seriesByDataset[dataset] ?? [];
    const i = list.findIndex((s) => s.file === file);
    const next = list[(i + dir + list.length) % list.length];
    if (next) setFile(next.file);
  };
  const sample = () => {
    const all = datasets.flatMap((d) => seriesByDataset[d]);
    if (all.length) setFile(all[Math.floor(Math.random() * all.length)].file);
  };

  // The anomalies in view, their marker positions, and the leader lines to them.
  const n = doc?.n ?? 1;
  const [v0, v1] = view ?? [0, n];
  const span = Math.max(v1 - v0, 1);
  const P = (x: number) => ((x - v0) / span) * 100;
  const anomalies = data?.anomalies ?? [];
  const vis = anomalies.map(([s, e], kk) => ({ k: kk, s, e })).filter((a) => a.e + 1 > v0 && a.s < v1);
  const crowded = vis.length > MAX_SCORED;
  const scored = crowded ? [] : vis;
  const centres = scored.map((a) => P((a.s + a.e + 1) / 2));
  const pos = spreadMarkers(centres, GAP);
  const leaders = scored.map((_, i) => (Math.abs(pos[i] - centres[i]) > 0.2 ? `M${pos[i].toFixed(2)} 0 L${Math.min(100, Math.max(0, centres[i])).toFixed(2)} 8` : "")).join(" ") || "M0 0";
  const labelBands: Band[] = anomalies.map(([s, e], kk) => ({ from: s, to: e + 1, color: "var(--label)", opacity: hk < 0 ? 0.45 : hk === kk ? 0.7 : 0.12, key: `l${kk}` }));
  const regions = (loc: number[]): Band[] => vis.map((a) => {
    const lo = a.k === 0 ? 0 : (anomalies[a.k - 1][1] + a.s) / 2;
    const hi = a.k === anomalies.length - 1 ? n : (a.e + anomalies[a.k + 1][0]) / 2;
    return { from: lo, to: hi, color: gradient(loc[a.k]), opacity: 1, key: `r${a.k}` };
  });
  const seriesData = covers(zoomed.doc) ? zoomed.doc! : doc;
  const dim = (kk: number) => (hk < 0 || hk === kk ? 1 : 0.18);
  const isMap = tab === "map";

  const seg = (on: boolean) => ({ "aria-pressed": on });
  // A column head: click to order the detectors by it; the marker shows the one in use.
  const head = (key: Sort, label: string, full = label) => (
    <button key={key} type="button" aria-pressed={sort === key} title={`Order by ${full}`} onClick={() => setSort(key)}
      style={{ marginLeft: "auto", display: "inline-flex", alignItems: "center", gap: 4, padding: 0, border: 0, background: "transparent", font: "inherit", fontSize: 12, fontWeight: 600, cursor: "pointer", color: sort === key ? "var(--ink)" : "var(--muted)" }}>
      {sort === key && <svg width="8" height="8" viewBox="0 0 8 8" aria-hidden="true"><path d="M0.5 2 L7.5 2 L4 7 Z" fill="var(--accent)" /></svg>}{label}
    </button>
  );
  return (
    <div style={{ width: "100%", maxWidth: 1600, margin: "0 auto", padding: "16px 24px 40px", display: "flex", flexDirection: "column", gap: 12 }}>
      <div className="panel" style={{ flexDirection: "row", flexWrap: "wrap", alignItems: "center", gap: 10, padding: "10px 14px" }} aria-label="Series">
        <select className="select" aria-label="Dataset" value={dataset} onChange={(e) => setFile(seriesByDataset[e.target.value][0].file)} style={{ height: 32 }}>
          {datasets.map((d) => <option key={d} value={d}>{d}</option>)}
        </select>
        <select className="select mono" aria-label="Series" value={file} onChange={(e) => setFile(e.target.value)} style={{ height: 32, fontSize: 12, flex: "0 1 420px", minWidth: 0 }}>
          {(seriesByDataset[dataset] ?? []).map((s) => <option key={s.file} value={s.file}>{seriesName(s.file)}</option>)}
        </select>
        <button type="button" className="btn" onClick={() => step(-1)}>Prev</button>
        <button type="button" className="btn" onClick={() => step(1)}>Next</button>
        <button type="button" className="btn btn-primary" onClick={sample}>Sample</button>
        <span style={{ marginLeft: "auto" }}><NotesButton file={file} name={seriesName(file)} /></span>
      </div>

      <section className="panel" aria-label="Series and detectors" style={{ padding: 0, overflowX: "auto", gap: 0 }}>
        <div style={{ minWidth: 900, padding: "0 20px 12px" }}>
          <div style={{ position: "sticky", top: 0, zIndex: 3, background: "var(--panel)", borderBottom: "1px solid var(--border)" }}>
            <div style={{ display: "flex", flexWrap: "wrap", alignItems: "center", gap: 12, padding: "12px 0", borderBottom: "1px solid var(--border-soft)" }}>
              <div className="seg" role="group" aria-label="View" style={{ display: "flex" }}>
                <button type="button" {...seg(!isMap)} onClick={() => setTab("scores")} style={{ padding: "0 14px" }}>Scores</button>
                <button type="button" {...seg(isMap)} onClick={() => setTab("map")} style={{ padding: "0 14px" }}>Heat map</button>
              </div>
              <span aria-hidden="true" style={{ width: 1, height: 22, background: "var(--border)" }} />
              <div className="seg" role="group" aria-label="Show" style={{ display: "flex" }}>
                <button type="button" {...seg(mode === "top")} onClick={() => setMode("top")} style={{ padding: "0 12px" }}>Top k</button>
                <button type="button" {...seg(mode === "best")} onClick={() => setMode("best")} style={{ padding: "0 12px" }}>Best coverage</button>
                <button type="button" {...seg(mode === "all")} onClick={() => setMode("all")} style={{ padding: "0 12px" }}>All</button>
              </div>
              {mode === "best" && (
                <button type="button" className="btn btn-primary" onClick={stitch} disabled={orc.busy || !best.winner.length} style={{ height: 30 }}>
                  {orc.busy ? "Stitching…" : "Stitch winners"}
                </button>
              )}
              {mode === "top" && (
                <label className="muted" style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 12 }}>k
                  <input type="number" className="input mono" min={0} max={32} value={topK} onChange={(e) => setTopK(Math.max(0, Math.min(32, Number(e.target.value) || 0)))} style={{ width: 56, height: 30, fontSize: 12 }} />
                </label>
              )}
              {hist.length > 0 && (
                <button type="button" className="btn" onClick={back} style={{ display: "inline-flex", alignItems: "center", gap: 6, height: 30, fontWeight: 600 }}>
                  <svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true"><path d="M7.5 2 L3.5 6 L7.5 10" fill="none" stroke="currentColor" strokeWidth="1.6" /></svg>Back
                </button>
              )}
              {isMap && (
                <div aria-label="Local colour scale, 0 to 1" className="muted" style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 12 }}>AutoVUS Local
                  <span className="mono" style={{ fontSize: 11 }}>0</span>
                  <span style={{ width: 120, height: 10, borderRadius: 2, background: GRADIENT_CSS, border: "1px solid var(--border)" }} />
                  <span className="mono" style={{ fontSize: 11 }}>1</span>
                </div>
              )}
              <label className="muted" style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 12, marginLeft: "auto" }}>Pin
                <select className="select" value="" onChange={(e) => { const v = e.target.value; if (v) setPins((p) => (p.includes(v) ? p : [...p, v])); }} style={{ height: 30, fontSize: 12 }}>
                  <option value="">…</option>
                  {rows.filter((r) => !shown.some((x) => x.name === r.name)).map((r) => <option key={r.name} value={r.name}>{r.name}</option>)}
                </select>
              </label>
            </div>

            {crowded && (
              <div role="status" style={{ display: "flex", alignItems: "center", gap: 10, marginTop: 10, padding: "8px 12px", borderRadius: 6, background: "color-mix(in srgb, var(--label) 18%, var(--panel))", border: "1px solid var(--label)", fontSize: 13 }}>
                <svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true" style={{ flex: "none" }}><path d="M8 1.5 L15 14 L1 14 Z" fill="none" stroke="var(--late)" strokeWidth="1.5" strokeLinejoin="round" /><path d="M8 6 L8 9.5 M8 11.5 L8 12" stroke="var(--late)" strokeWidth="1.6" strokeLinecap="round" /></svg>
                <span>Too many anomalies ({vis.length}), cannot show all at once. Zoom in to see their scores.</span>
              </div>
            )}

            <div style={{ display: "grid", gridTemplateColumns: COLS, gap: "0 16px", alignItems: "end", paddingTop: 10 }}>
              <div className="lane-name" style={{ paddingBottom: 8 }}>
                {error && <span style={{ color: "var(--late)", fontSize: 11 }}>{error}</span>}
                {doc && <span className="muted" style={{ fontSize: 11 }}>{doc.n.toLocaleString("en-US")} points · {anomalies.length} anomalies</span>}
              </div>
              <div style={{ position: "relative", paddingTop: scored.length ? 40 : 6 }}>
                {scored.length > 0 && (
                  <>
                    <div style={{ position: "absolute", left: 0, right: 0, top: 4, height: 24 }}>
                      {scored.map((a, i) => (
                        <button key={a.k} type="button" onMouseEnter={() => setHk(a.k)} onMouseLeave={() => setHk(-1)} onFocus={() => setHk(a.k)} onBlur={() => setHk(-1)}
                          className="mono" style={{ position: "absolute", top: 0, left: `${pos[i]}%`, transform: "translateX(-50%)", height: 24, minWidth: 38, padding: "0 6px", border: "1px solid var(--label)", borderRadius: 4, background: hk === a.k ? "var(--label)" : "color-mix(in srgb, var(--label) 22%, var(--panel))", color: "var(--ink)", fontSize: 12, cursor: "default" }}>
                          A{a.k + 1}
                        </button>
                      ))}
                    </div>
                    <svg viewBox="0 0 100 8" preserveAspectRatio="none" aria-hidden="true" style={{ position: "absolute", left: 0, top: 28, width: "100%", height: 12 }}>
                      <path d={leaders} fill="none" stroke="var(--missed)" strokeWidth={1} vectorEffect="non-scaling-stroke" />
                    </svg>
                  </>
                )}
                {seriesData ? (
                  <Lane lo={seriesData.lo} hi={seriesData.hi} bins={seriesData.bins} n={seriesData.n} start={seriesData.start} end={seriesData.end} view={view} onZoom={zoomTo} height={120} bands={labelBands} />
                ) : <div style={{ height: 120, background: "var(--subtle)", borderRadius: 4 }} />}
              </div>
              <div style={{ paddingBottom: 8, display: "flex" }}>{head("ap", "AUC")}</div>
              <div style={{ paddingBottom: 8, display: "flex" }}>{head("vus", "VUS")}</div>
              <div style={{ gridColumn: "span 2", display: "grid", gridTemplateColumns: "64px 56px", gap: "3px 16px", paddingBottom: 8, fontSize: 12, fontWeight: 600 }}>
                <span style={{ gridColumn: "span 2", textAlign: "center", color: "var(--accent)", borderBottom: "1px solid var(--accent)", paddingBottom: 3 }}>AutoVUS</span>
                {head("auto", "Global", "AutoVUS Global")}
                {head("local", "Local", "AutoVUS Local")}
              </div>
            </div>
            {isMap && scored.length > 0 && (
              <div style={{ display: "grid", gridTemplateColumns: COLS, gap: "0 16px", padding: "8px 0 6px" }}>
                <span />
                <div style={{ display: "grid", gridTemplateColumns: `repeat(${scored.length}, minmax(40px, 1fr))`, gap: 4 }}>
                  {scored.map((a) => <span key={a.k} className="mono" style={{ textAlign: "center", fontSize: 11, fontWeight: 500, color: hk === a.k ? "var(--ink)" : "var(--muted)" }} onMouseEnter={() => setHk(a.k)} onMouseLeave={() => setHk(-1)}>A{a.k + 1}</span>)}
                </div>
                <span /><span /><span /><span />
              </div>
            )}
          </div>

          {!data && !error && <div style={{ height: 320 }} />}
          {lanes.map((r) => {
            const isOracle = !!r.norm;
            const lastOracle = isOracle && r === oracleRows[oracleRows.length - 1];
            const pinned = pins.includes(r.name);
            const sc = r.norm ? (covers(orc.zoom[r.norm]) ? orc.zoom[r.norm] : orc.docs[r.norm]) : covers(zoomed.scores[r.name]) ? zoomed.scores[r.name] : scores[r.name];
            const chips = scored.map((a, i) => ({ a, left: pos[i], v: r.loc[a.k], ring: !isOracle && mode === "best" && best.winner[a.k] === r.name, on: !isOracle && picked?.detector === r.name && picked.k === a.k }));
            const open = isOracle ? undefined : (kk: number) => ({ role: "button", tabIndex: 0, onClick: () => pick(r.name, kk), onKeyDown: (e: React.KeyboardEvent) => { if (e.key === "Enter") pick(r.name, kk); } });
            const zones: Zone[] = isOracle ? [] : vis.map((a) => ({
              from: a.s - L, to: a.e + 1 + L, color: "transparent", key: `z${a.k}`, selected: picked?.detector === r.name && picked.k === a.k,
              onEnter: () => setHk(a.k), onLeave: () => setHk(-1), onClick: () => pick(r.name, a.k),
            }));
            const mine = !isOracle && picked?.detector === r.name ? picked : null;
            return (
              <Fragment key={r.name}>
              <div style={{ display: "grid", gridTemplateColumns: COLS, gap: "0 16px", alignItems: "center", borderBottom: lastOracle ? "2px solid var(--border)" : "1px solid var(--border-soft)", padding: isMap ? "6px 0" : "8px 0" }}>
                <div style={{ display: "flex", alignItems: "center", gap: 6, minWidth: 0, fontSize: 13, fontWeight: 600 }}>
                  {pinned && <svg width="9" height="9" viewBox="0 0 10 10" aria-hidden="true" style={{ flex: "none" }}><path d="M5 0 L10 5 L5 10 L0 5 Z" fill="var(--accent)" /></svg>}
                  <span style={{ whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>{r.name}</span>
                  {!isOracle && <button type="button" className="unpin" aria-label={pinned ? `Unpin ${r.name}` : `Pin ${r.name}`} title={pinned ? "Unpin" : "Pin"}
                    onClick={() => setPins((p) => (pinned ? p.filter((x) => x !== r.name) : [...p, r.name]))}>{pinned ? "×" : "+"}</button>}
                </div>

                {!isMap && (
                  <div style={{ position: "relative", paddingTop: chips.length ? 30 : 0 }}>
                    {chips.length > 0 && (
                      <>
                        <div style={{ position: "absolute", left: 0, right: 0, top: 0, height: 22 }}>
                          {chips.map(({ a, left, v, ring, on }) => {
                            const c = chipColors(v);
                            return (
                              <span key={a.k} onMouseEnter={() => setHk(a.k)} onMouseLeave={() => setHk(-1)} className="mono" {...open?.(a.k)}
                                style={{ position: "absolute", top: 0, left: `${left}%`, transform: "translateX(-50%)", height: 22, minWidth: 38, padding: "0 4px", boxSizing: "border-box", borderRadius: 4, display: "inline-flex", alignItems: "center", justifyContent: "center", background: c.bg, color: c.fg, opacity: dim(a.k), outline: on ? "2px solid var(--accent)" : ring ? "2px solid var(--ink)" : "none", outlineOffset: 1, fontSize: 12, fontWeight: 500, cursor: open ? "pointer" : undefined }}>
                                {v.toFixed(2)}
                              </span>
                            );
                          })}
                        </div>
                        <svg viewBox="0 0 100 8" preserveAspectRatio="none" aria-hidden="true" style={{ position: "absolute", left: 0, top: 22, width: "100%", height: 8 }}>
                          <path d={leaders} fill="none" stroke="var(--border)" strokeWidth={1} vectorEffect="non-scaling-stroke" />
                        </svg>
                      </>
                    )}
                    {sc ? (
                      <Lane lo={sc.lo} hi={sc.hi} bins={sc.bins} n={sc.n} start={sc.start} end={sc.end} view={view} onZoom={zoomTo} ymin={0} ymax={1} height={76} bands={labelBands} zones={zones} stroke="var(--accent)" />
                    ) : <div style={{ height: 76, background: "var(--subtle)", borderRadius: 4 }} />}
                  </div>
                )}

                {isMap && crowded && (
                  <Lane lo={[0]} hi={[0]} bins={1} n={n} view={view} onZoom={zoomTo} height={34} bands={regions(r.loc)} stroke="transparent" />
                )}
                {isMap && !crowded && (
                  <div style={{ display: "grid", gridTemplateColumns: `repeat(${Math.max(1, scored.length)}, minmax(40px, 1fr))`, gap: 4 }}>
                    {chips.map(({ a, v, ring, on }) => {
                      const c = chipColors(v);
                      return (
                        <span key={a.k} onMouseEnter={() => setHk(a.k)} onMouseLeave={() => setHk(-1)} className="mono" {...open?.(a.k)}
                          style={{ height: 34, borderRadius: 4, display: "flex", alignItems: "center", justifyContent: "center", background: c.bg, color: c.fg, opacity: dim(a.k), outline: on ? "2px solid var(--accent)" : ring ? "2px solid var(--ink)" : "none", outlineOffset: -2, fontSize: 13, fontWeight: 500, cursor: open ? "pointer" : undefined }}>
                          {v.toFixed(2)}
                        </span>
                      );
                    })}
                  </div>
                )}

                {([["ap", r.ap, true], ["vv", r.vv, true], ["gv", r.gv, false], ["lv", r.lv, false]] as const).map(([key, v, grey]) => {
                  const hit = v !== null && v !== undefined && v === top[key];
                  return (
                    <span key={key} className={`mono${grey && !hit ? " muted" : ""}`}
                      style={{ textAlign: "right", fontSize: 14, fontWeight: hit ? 700 : 400, color: hit ? "var(--accent)" : undefined }}>
                      {v === null || v === undefined ? "–" : f2(v)}
                    </span>
                  );
                })}
              </div>
              {mine && (
                <div className="panel panel-accent" aria-label="Selected anomaly" style={{ margin: "10px 0 14px" }}>
                  <div className="panel-head">
                    <h2>Why this buffer</h2>
                    <span className="mono" style={{ fontSize: 12 }}>
                      A{mine.k + 1} · {r.name}{why && <> · [{fmt(why.a.start)}, {fmt(why.a.end)}] · <span style={{ color: TYPE_COLOR[why.a.type], fontWeight: 600 }}>{why.a.type}</span> · b* = ({why.a.b_left}, {why.a.b_right})</>}
                    </span>
                    <button type="button" className="btn" style={{ marginLeft: "auto", height: 28 }} onClick={() => setPicked(null)}>Close</button>
                  </div>
                  {why ? <Drawer curves={why.curves} a={why.a} L={L} /> : <div className="muted" style={{ fontSize: 12 }}>…</div>}
                </div>
              )}
              </Fragment>
            );
          })}
        </div>
      </section>
    </div>
  );
}
