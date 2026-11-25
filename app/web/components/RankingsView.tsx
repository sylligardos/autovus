"use client";
// The Rankings page: two metrics side by side, one box per detector, with
// every rank move drawn between them, the datasets where the pair disagrees
// most, the series behind the biggest move, and every metric against every other.

import { useEffect, useMemo, useState } from "react";
import { METRICS as ALL_METRICS, f2, shortFile, type MetricKey, type PairRow, type State } from "@/lib/model";
import { toMap, weightedKendall } from "@/lib/stats";
import { expand, type CompactTables } from "@/lib/compact";
import { loadLocalTable } from "@/lib/api";
import type { CoverTable } from "@/lib/local";
import { useRemembered } from "@/lib/remember";
import { SeriesPair } from "./SeriesPair";

const ROW = 18, HEAD = 22, AXIS = 14;
// The table of the gear (app/export_gear.py): AutoVUS, VUS-PR and AUC-PR; references = the oracle and the stitched score.
// Two more from the local export (cover.json): AutoVUS Local, as its mean over the anomalies of a series
// (one value per series) and as it is (one value per anomaly). The references have no local scores.
type RKey = MetricKey | "local" | "local_anomaly";
const METRICS: { key: RKey; label: string }[] = [
  ...ALL_METRICS.filter((m) => !["vus_roc", "auc_roc", "adj_vus"].includes(m.key)),
  { key: "local", label: "AutoVUS Local" }, { key: "local_anomaly", label: "AutoVUS Local per anomaly" },
];
const metricLabel = (k: RKey) => METRICS.find((m) => m.key === k)?.label ?? k;
const REFERENCES = ["Oracle", "Stitched score"];
const known = (k?: string) => (METRICS.some((m) => m.key === k) ? (k as RKey) : undefined);

type XRow = PairRow & { local: number | null; locals: number[] | null };
type Box = { detector: string; value: number; n: number; lo: number; q1: number; med: number; q3: number; hi: number; mean: number };
// The values a row gives to a metric: one, or one per anomaly.
const valuesOf = (r: XRow, k: RKey): number[] => {
  if (k === "local_anomaly") return r.locals ?? [];
  const v = r[k];
  return v === null ? [] : [v];
};
const quantile = (xs: number[], q: number) => { const p = (xs.length - 1) * q, i = Math.floor(p); return xs[i] + (xs[Math.min(i + 1, xs.length - 1)] - xs[i]) * (p - i); };
/** One box per detector over the rows (whiskers: the furthest values within 1.5 box lengths), best first by mean or median. */
function boxes(rows: XRow[], k: RKey, agg: "mean" | "median"): Box[] {
  const by = new Map<string, number[]>();
  for (const r of rows) { const xs = valuesOf(r, k); if (!xs.length) continue; let a = by.get(r.detector); if (!a) { a = []; by.set(r.detector, a); } for (const x of xs) a.push(x); }
  return Array.from(by, ([detector, xs]) => {
    xs.sort((a, b) => a - b);
    const q1 = quantile(xs, 0.25), med = quantile(xs, 0.5), q3 = quantile(xs, 0.75), reach = 1.5 * (q3 - q1);
    const mean = xs.reduce((a, b) => a + b, 0) / xs.length;
    return { detector, value: agg === "mean" ? mean : med, n: xs.length, lo: xs.find((x) => x >= q1 - reach)!, q1, med, q3, hi: xs.findLast((x) => x <= q3 + reach)!, mean };
  }).filter((x) => Number.isFinite(x.value)).sort((a, b) => b.value - a.value);
}

export function RankingsView({ state, tables: packed, initial }: { state: State; tables: CompactTables; initial: { a?: string; b?: string } }) {
  const tables = useMemo(() => expand(packed), [packed]);
  const k = `rankings:${state.benchmark}:`;
  const [amRaw, setAm] = useRemembered<RKey>(k + "a", "auc_pr", known(initial.a));
  const [bmRaw, setBm] = useRemembered<RKey>(k + "b", "autovus", known(initial.b));
  const am = known(amRaw) ?? "auc_pr", bm = known(bmRaw) ?? "autovus";
  const [agg, setAgg] = useRemembered<"mean" | "median">(k + "agg", "mean");
  const [refs, setRefs] = useRemembered(k + "refs", true);
  const [countSel, setCount] = useRemembered<number | null>(k + "count", null);
  const [scope, setScope] = useRemembered(k + "scope", "");
  const [file, setFile] = useRemembered<string | null>(k + "file", null);

  const [cover, setCover] = useState<CoverTable | null>(null);
  useEffect(() => {
    let live = true;
    loadLocalTable<CoverTable>(state.benchmark, "cover", state.L).then((c) => { if (live) setCover(c); });
    return () => { live = false; };
  }, [state.benchmark, state.L]);
  // Every row with its local scores (per anomaly, in hundredths in cover.json; -999 = no score).
  const xrows = useMemo<XRow[]>(() => {
    const at = new Map<string, number[]>();
    if (cover) cover.series.forEach((file, s) => {
      const K = cover.K[s], M = cover.M[s];
      cover.detectors.forEach((det, d) => { if (M[d * K] !== -999) at.set(file + "|" + det, M.slice(d * K, (d + 1) * K).map((v) => v / 100)); });
    });
    return tables.rows.map((r) => {
      const xs = at.get(r.file + "|" + r.detector) ?? null;
      return { ...r, locals: xs, local: xs ? xs.reduce((a, b) => a + b, 0) / xs.length : null };
    });
  }, [tables.rows, cover]);

  // The references have no local scores: with a local metric in the pair they leave both columns, so the ranks compare.
  const showRefs = refs && !am.startsWith("local") && !bm.startsWith("local");
  const rows = useMemo(() => xrows.filter((r) => (showRefs || !REFERENCES.includes(r.detector)) && (!scope || r.dataset === scope)), [xrows, showRefs, scope]);
  // The reference detectors appear in the pair table only, never in the tables below it.
  const real = useMemo(() => rows.filter((r) => !REFERENCES.includes(r.detector)), [rows]);
  const rA = useMemo(() => boxes(rows, am, agg), [rows, am, agg]);
  const rB = useMemo(() => boxes(rows, bm, agg), [rows, bm, agg]);
  const most = Math.max(rA.length, rB.length);
  const count = Math.min(countSel ?? most, most);
  const posA = new Map(rA.map((r, i) => [r.detector, i])), posB = new Map(rB.map((r, i) => [r.detector, i]));
  const shownA = rA.slice(0, count), shownB = rB.slice(0, count);
  // A detector without one of the two metrics (a reference with a local metric) has no move.
  const inView = new Set([...shownA, ...shownB].map((r) => r.detector).filter((d) => posA.has(d) && posB.has(d)));
  // Both columns share one axis, cut at -0.1: a chance-adjusted score has a long tail below 0.
  const lo = Math.max(-0.1, Math.min(0, ...shownA.map((r) => r.lo), ...shownB.map((r) => r.lo))), hi = 1;
  const moves = new Map<string, number>();
  for (const d of inView) moves.set(d, (posB.get(d) ?? 99) - (posA.get(d) ?? 99));
  const lineColor = (d: string) => {
    const m = moves.get(d) ?? 0;
    return m < 0 ? "var(--up)" : m > 0 ? "var(--down)" : "var(--missed)";
  };

  const datasets = useMemo(() => Array.from(new Set(tables.rows.map((r) => r.dataset))).sort(), [tables.rows]);
  // Datasets (whole benchmark) or series (one dataset) by weighted tau between the pair, lowest first.
  const groups = useMemo(() => {
    const by = new Map<string, XRow[]>();
    for (const r of real) { const k = scope ? r.file : r.dataset; if (!by.has(k)) by.set(k, []); by.get(k)!.push(r); }
    return Array.from(by, ([key, rs]) => {
      const a = boxes(rs, am, agg), b = boxes(rs, bm, agg);
      const pa = new Map(a.map((r, i) => [r.detector, i + 1])), pb = new Map(b.map((r, i) => [r.detector, i + 1]));
      // The biggest move, weighted to the top as on Cases: |1/rank A - 1/rank B|.
      let mover = "", mv = 0, from = 0, to = 0;
      for (const [det, p] of pa) { const q = pb.get(det); const w = q === undefined ? 0 : Math.abs(1 / p - 1 / q); if (w > mv) { mv = w; mover = det; from = p; to = q!; } }
      return { key, n: new Set(rs.map((r) => r.file)).size, tau: weightedKendall(toMap(a), toMap(b)), winA: a[0]?.detector ?? "", winB: b[0]?.detector ?? "", mover, from, to };
    }).sort((x, y) => (x.tau ?? 1) - (y.tau ?? 1)).slice(0, scope ? 10 : undefined);
  }, [real, scope, am, bm, agg]);
  const pick = scope ? groups.find((g) => g.key === file) ?? groups[0] ?? null : null;

  const pairs = useMemo(() => METRICS.map((m) => m.key), []);
  const taus = useMemo(() => {
    const ranks = Object.fromEntries(pairs.map((k) => [k, toMap(boxes(real, k, agg))])) as Record<RKey, Map<string, number>>;
    return pairs.map((x) => pairs.map((y) => weightedKendall(ranks[x], ranks[y])));
  }, [real, agg, pairs]);
  // Continuous blue scale stretched over the off-diagonal taus on screen.
  const off = taus.flatMap((r, i) => r.slice(0, i)).filter((t): t is number => t !== null);
  const tLo = off.length ? Math.min(...off) : 0, tHi = off.length ? Math.max(...off) : 1;
  const frac = (t: number) => (tHi > tLo ? (t - tLo) / (tHi - tLo) : 1);
  const shade = (t: number | null) => (t === null ? "var(--border-soft)" : `color-mix(in oklab, var(--accent) ${Math.round(10 + 90 * frac(t))}%, var(--border-soft))`);
  const tauText = (t: number) => { const v = Math.abs(t).toFixed(2); return (t < 0 ? "-" : "") + (v.startsWith("0") ? v.slice(1) : v); };

  return (
    <>
      <aside className="rail" aria-label="Controls">
        <div className="field"><span>Scope</span>
          <select className="select" aria-label="Scope" value={scope} onChange={(e) => setScope(e.target.value)}><option value="">whole benchmark</option>{datasets.map((d) => <option key={d} value={d}>{d}</option>)}</select>
        </div>
        <div className="field"><span>Pair</span>
          <label className="muted" style={{ fontSize: 12, display: "flex", flexDirection: "column", gap: 3 }}>A<select className="select" style={{ height: 32 }} value={am} onChange={(e) => setAm(e.target.value as RKey)}>{METRICS.map((m) => <option key={m.key} value={m.key}>{m.label}</option>)}</select></label>
          <label className="muted" style={{ fontSize: 12, display: "flex", flexDirection: "column", gap: 3 }}>B<select className="select" style={{ height: 32, fontWeight: 600 }} value={bm} onChange={(e) => setBm(e.target.value as RKey)}>{METRICS.map((m) => <option key={m.key} value={m.key}>{m.label}</option>)}</select></label>
          <button type="button" className="btn" onClick={() => { setAm(bm); setBm(am); }}>Swap A and B</button>
        </div>
        <div className="field"><span>Aggregate</span>
          <span className="seg" role="group" aria-label="Aggregate"><button type="button" aria-pressed={agg === "mean"} onClick={() => setAgg("mean")}>Mean</button><button type="button" aria-pressed={agg === "median"} onClick={() => setAgg("median")}>Median</button></span>
        </div>
        <div className="field" style={{ gap: 6 }}><span>Show</span>
          <label style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 13 }}><input type="checkbox" checked={showRefs} disabled={refs && !showRefs} onChange={(e) => setRefs(e.target.checked)} />Reference detectors</label>
          <label className="muted" style={{ display: "flex", flexDirection: "column", gap: 4, fontSize: 12, marginTop: 4 }}>
            <span style={{ display: "flex", justifyContent: "space-between" }}>Detectors <span className="mono" style={{ color: "var(--ink)" }}>{count} of {most}</span></span>
            <input type="range" min={5} max={Math.max(most, 5)} value={count} onChange={(e) => setCount(Number(e.target.value))} />
          </label>
        </div>
      </aside>

      <main className="content">
        <section className="panel" aria-label="The pair">
          <div style={{ display: "grid", gridTemplateColumns: "minmax(0, 5fr) minmax(0, 3fr) minmax(0, 5fr)", gap: "0 12px", alignItems: "start" }}>
            <Bars rows={shownA} title={`A · ${metricLabel(am)}`} color={lineColor} accent={false} lo={lo} hi={hi} />
            <div>
              <div className="eyebrow" style={{ height: HEAD + AXIS, display: "flex", alignItems: "center", justifyContent: "center" }}>rank moves</div>
              <svg viewBox={`0 0 200 ${ROW * count}`} width="100%" height={ROW * count} preserveAspectRatio="none" style={{ display: "block" }} aria-label="Rank transitions" role="img">
                {Array.from(inView).map((d) => {
                  const a = posA.get(d) ?? 99, b = posB.get(d) ?? 99;
                  const ya = a < count ? a * ROW + ROW / 2 : ROW * count + 20, yb = b < count ? b * ROW + ROW / 2 : ROW * count + 20;
                  const c = lineColor(d);
                  return <path key={d} d={`M0,${ya} C100,${ya} 100,${yb} 200,${yb}`} fill="none" stroke={c} strokeWidth={c === "var(--missed)" ? 1.5 : REFERENCES.includes(d) ? 2 : 3} strokeDasharray={REFERENCES.includes(d) ? "5 4" : undefined} vectorEffect="non-scaling-stroke" />;
                })}
              </svg>
            </div>
            <Bars rows={shownB} title={`B · ${metricLabel(bm)}`} color={lineColor} accent lo={lo} hi={hi} />
          </div>
        </section>

        <section className="panel" aria-label="Where the pair disagrees">
          <div className="panel-head">{scope && <button type="button" className="btn" aria-label="Back to the whole benchmark" onClick={() => setScope("")}>← All datasets</button>}<h2>Where the pair disagrees{scope && <span className="muted" style={{ fontWeight: 400 }}> · {scope}</span>}</h2></div>
          <div style={{ overflowX: "auto" }}>
            <table className="table">
              <thead><tr><th>{scope ? "series" : "dataset"}</th>{!scope && <th className="r">series</th>}<th className="r">weighted tau</th><th>winner A</th><th>winner B</th><th>biggest move</th></tr></thead>
              <tbody>
                {groups.map((g) => (
                  <tr key={g.key} className={g.key === pick?.key ? "picked" : undefined} onClick={() => (scope ? setFile(g.key) : setScope(g.key))}>
                    <td className={scope ? "mono" : undefined} style={{ fontWeight: 600 }}>{scope ? shortFile(g.key) : g.key}</td>{!scope && <td className="r mono">{g.n}</td>}<td className="r mono">{g.tau === null ? "–" : g.tau.toFixed(2)}</td>
                    <td>{g.winA}</td><td>{g.winB}</td>
                    <td className="mono" style={{ color: g.to < g.from ? "var(--up)" : g.to > g.from ? "var(--down)" : undefined }}>{g.mover} {g.from} to {g.to}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>

        {pick && <SeriesPair state={state} file={pick.key} a={pick.winB} b={pick.mover} move={pick.to - pick.from} am="autovus" bm="autovus" rows={tables.rows}
          show={(det) => { const r = xrows.find((x) => x.file === pick.key && x.detector === det); return [am, bm].map((m) => { const xs = r ? valuesOf(r, m) : []; return `${metricLabel(m)} ${f2(xs.length ? xs.reduce((p, q) => p + q, 0) / xs.length : null)}`; }); }} />}

        <section className="panel" aria-label="Agreement between metrics">
          <div className="panel-head"><h2>Every metric against every other</h2><span className="muted" style={{ fontSize: 12 }}>weighted tau, {scope || "whole benchmark"}</span><span className="muted" style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: 8, fontSize: 12 }}><span className="mono">{tauText(tLo)}</span><span style={{ width: 90, height: 10, borderRadius: 3, background: `linear-gradient(90deg, ${shade(tLo)}, ${shade(tHi)})` }} /><span className="mono">{tauText(tHi)}</span></span></div>
          <table style={{ borderCollapse: "separate", borderSpacing: 3, fontSize: 12 }}>
            <thead><tr><th className="eyebrow" style={{ width: 150, textAlign: "right", paddingRight: 6, fontWeight: 500 }}>A \ B</th>{pairs.map((k) => <th key={k} className="eyebrow" style={{ width: 72, verticalAlign: "bottom", fontWeight: k === bm ? 700 : 500, textTransform: "none", letterSpacing: 0, color: k === bm ? "var(--accent)" : undefined }}>{metricLabel(k)}</th>)}</tr></thead>
            <tbody className="mono" style={{ textAlign: "center" }}>
              {pairs.map((x, i) => (
                <tr key={x}>
                  <th scope="row" className="sans" style={{ textAlign: "right", color: x === am ? "var(--accent)" : "var(--muted)", fontWeight: x === am ? 700 : 500, fontSize: 11, paddingRight: 6 }}>{metricLabel(x)}</th>
                  {pairs.map((y, j) => {
                    const t = i === j ? 1 : taus[i][j];
                    const sel = x === am && y === bm;
                    return <td key={y} onClick={() => { if (i !== j) { setAm(x); setBm(y); } }} style={{ height: 40, borderRadius: 4, cursor: i !== j ? "pointer" : "default", background: i === j ? "var(--ink)" : shade(t), color: i === j || (t !== null && frac(t) >= 0.55) ? "var(--panel)" : "var(--ink)", outline: sel ? "3px solid var(--ink)" : undefined, outlineOffset: -1, fontWeight: sel ? 600 : 400 }}>{t === null ? "–" : i === j ? "1" : tauText(t)}</td>;
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      </main>
    </>
  );
}

function Bars({ rows, title, color, accent, lo, hi }: { rows: Box[]; title: string; color: (d: string) => string; accent: boolean; lo: number; hi: number }) {
  const at = (v: number) => `${((v - lo) / (hi - lo)) * 100}%`;
  return (
    <div>
      <div className="eyebrow" style={{ height: HEAD, display: "flex", alignItems: "center", color: accent ? "var(--accent)" : undefined }}>{title}</div>
      <div style={{ display: "grid", gridTemplateColumns: "18px 130px minmax(0, 1fr) 36px", gap: "0 8px", alignItems: "center", fontSize: 11, gridAutoRows: ROW }}>
        <span style={{ height: AXIS }} /><span />
        <span className="mono muted" aria-hidden="true" style={{ position: "relative", height: AXIS, fontSize: 10 }}>
          {[0, 0.5, 1].map((v) => <span key={v} style={{ position: "absolute", left: at(v), transform: `translateX(${v === 1 ? "-100%" : v === 0 ? "0" : "-50%"})` }}>{v}</span>)}
        </span><span />
        {rows.map((r, i) => <Row key={r.detector} i={i} r={r} ref_={REFERENCES.includes(r.detector)} c={color(r.detector)} at={at} />)}
      </div>
    </div>
  );
}
function Row({ i, r, ref_, c, at }: { i: number; r: Box; ref_: boolean; c: string; at: (v: number) => string }) {
  const strong = c !== "var(--missed)" && c !== "var(--accent)";
  const w = (a: number, b: number) => `calc(${at(b)} - ${at(a)})`;
  return (
    <>
      <span className="mono muted">{i + 1}</span>
      <span style={{ display: "flex", alignItems: "center", gap: 5, color: strong && !ref_ ? c : undefined, fontWeight: strong && !ref_ ? 600 : 400, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
        {ref_ && <span className="ref-dot" style={{ borderColor: "var(--ink)" }} />}{r.detector}
      </span>
      <span title={`median ${f2(r.med)} · mean ${f2(r.mean)} · box ${f2(r.q1)} to ${f2(r.q3)} · whiskers ${f2(r.lo)} to ${f2(r.hi)} · ${r.n} values`} style={{ position: "relative", height: 12, background: "var(--border-soft)", borderRadius: 3, overflow: "hidden" }}>
        <span style={{ position: "absolute", left: at(r.lo), width: w(r.lo, r.hi), top: 5.5, height: 1, background: c }} />
        <span style={{ position: "absolute", left: at(r.q1), width: w(r.q1, r.q3), top: 1, bottom: 1, minWidth: 2, borderRadius: 2, border: `1px ${ref_ ? "dashed" : "solid"} ${ref_ ? "var(--ink)" : c}`, background: ref_ ? "transparent" : `color-mix(in srgb, ${c} 45%, var(--panel))` }} />
        <span style={{ position: "absolute", left: at(r.med), top: 0, bottom: 0, width: 2, marginLeft: -1, background: "var(--ink)" }} />
        <span style={{ position: "absolute", left: at(r.mean), top: 3, width: 6, height: 6, marginLeft: -3, borderRadius: "50%", background: "var(--ink)", border: "1px solid var(--panel)" }} />
      </span>
      <span className="mono" style={{ textAlign: "right" }}>{f2(r.value)}</span>
    </>
  );
}
