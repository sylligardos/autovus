"use client";
// The Rankings page: two metrics side by side with every rank move drawn
// between them, the datasets where the pair disagrees most, the series
// behind the biggest move, and every metric against every other.

import { useMemo } from "react";
import { METRICS as ALL_METRICS, f2, metricLabel, shortFile, type MetricKey, type PairRow, type State } from "@/lib/model";
import { rankDetectors, toMap, weightedKendall } from "@/lib/stats";
import { expand, type CompactTables } from "@/lib/compact";
import { useRemembered } from "@/lib/remember";
import { SeriesPair } from "./SeriesPair";

const ROW = 18, HEAD = 22;
// L=20 table (app/export_rankings20.py): no ROC metrics, references = the oracle and the two stitched scores.
const METRICS = ALL_METRICS.filter((m) => !["vus_roc", "auc_roc"].includes(m.key));
const REFERENCES = ["Oracle", "Stitched min-max", "Stitched median"];
const known = (k?: string) => (METRICS.some((m) => m.key === k) ? (k as MetricKey) : undefined);

export function RankingsView({ state, tables: packed, initial }: { state: State; tables: CompactTables; initial: { a?: string; b?: string } }) {
  const tables = useMemo(() => expand(packed), [packed]);
  const k = `rankings:${state.benchmark}:`;
  const [amRaw, setAm] = useRemembered<MetricKey>(k + "a", "auc_pr", known(initial.a));
  const [bmRaw, setBm] = useRemembered<MetricKey>(k + "b", "autovus", known(initial.b));
  const am = known(amRaw) ?? "auc_pr", bm = known(bmRaw) ?? "autovus";
  const [agg, setAgg] = useRemembered<"mean" | "median">(k + "agg", "mean");
  const [refs, setRefs] = useRemembered(k + "refs", true);
  const [countSel, setCount] = useRemembered<number | null>(k + "count", null);
  const [scope, setScope] = useRemembered(k + "scope", "");
  const [file, setFile] = useRemembered<string | null>(k + "file", null);

  const rows = useMemo(() => tables.rows.filter((r) => (refs || !REFERENCES.includes(r.detector)) && (!scope || r.dataset === scope)), [tables.rows, refs, scope]);
  // The reference detectors appear in the pair table only, never in the tables below it.
  const real = useMemo(() => rows.filter((r) => !REFERENCES.includes(r.detector)), [rows]);
  const rA = useMemo(() => rankDetectors(rows, am, agg), [rows, am, agg]);
  const rB = useMemo(() => rankDetectors(rows, bm, agg), [rows, bm, agg]);
  const count = Math.min(countSel ?? rA.length, rA.length);
  const posA = new Map(rA.map((r, i) => [r.detector, i])), posB = new Map(rB.map((r, i) => [r.detector, i]));
  const shownA = rA.slice(0, count), shownB = rB.slice(0, count);
  const inView = new Set([...shownA, ...shownB].map((r) => r.detector));
  const moves = new Map<string, number>();
  for (const d of inView) moves.set(d, (posB.get(d) ?? 99) - (posA.get(d) ?? 99));
  const lineColor = (d: string) => {
    const m = moves.get(d) ?? 0;
    return m < 0 ? "var(--up)" : m > 0 ? "var(--down)" : "var(--missed)";
  };

  const datasets = useMemo(() => Array.from(new Set(tables.rows.map((r) => r.dataset))).sort(), [tables.rows]);
  // Datasets (whole benchmark) or series (one dataset) by weighted tau between the pair, lowest first.
  const groups = useMemo(() => {
    const by = new Map<string, PairRow[]>();
    for (const r of real) { const k = scope ? r.file : r.dataset; if (!by.has(k)) by.set(k, []); by.get(k)!.push(r); }
    return Array.from(by, ([key, rs]) => {
      const a = rankDetectors(rs, am, agg), b = rankDetectors(rs, bm, agg);
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
    const ranks = Object.fromEntries(pairs.map((k) => [k, toMap(rankDetectors(real, k, agg))])) as Record<MetricKey, Map<string, number>>;
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
          <label className="muted" style={{ fontSize: 12, display: "flex", flexDirection: "column", gap: 3 }}>A<select className="select" style={{ height: 32 }} value={am} onChange={(e) => setAm(e.target.value as MetricKey)}>{METRICS.map((m) => <option key={m.key} value={m.key}>{m.label}</option>)}</select></label>
          <label className="muted" style={{ fontSize: 12, display: "flex", flexDirection: "column", gap: 3 }}>B<select className="select" style={{ height: 32, fontWeight: 600 }} value={bm} onChange={(e) => setBm(e.target.value as MetricKey)}>{METRICS.map((m) => <option key={m.key} value={m.key}>{m.label}</option>)}</select></label>
          <button type="button" className="btn" onClick={() => { setAm(bm); setBm(am); }}>Swap A and B</button>
        </div>
        <div className="field"><span>Aggregate</span>
          <span className="seg" role="group" aria-label="Aggregate"><button type="button" aria-pressed={agg === "mean"} onClick={() => setAgg("mean")}>Mean</button><button type="button" aria-pressed={agg === "median"} onClick={() => setAgg("median")}>Median</button></span>
        </div>
        <div className="field" style={{ gap: 6 }}><span>Show</span>
          <label style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 13 }}><input type="checkbox" checked={refs} onChange={(e) => setRefs(e.target.checked)} />Reference detectors</label>
          <label className="muted" style={{ display: "flex", flexDirection: "column", gap: 4, fontSize: 12, marginTop: 4 }}>
            <span style={{ display: "flex", justifyContent: "space-between" }}>Detectors <span className="mono" style={{ color: "var(--ink)" }}>{count} of {rA.length}</span></span>
            <input type="range" min={5} max={Math.max(rA.length, 5)} value={count} onChange={(e) => setCount(Number(e.target.value))} />
          </label>
        </div>
      </aside>

      <main className="content">
        <section className="panel" aria-label="The pair">
          <div style={{ display: "grid", gridTemplateColumns: "minmax(0, 5fr) minmax(0, 3fr) minmax(0, 5fr)", gap: "0 12px", alignItems: "start" }}>
            <Bars rows={shownA} title={`A · ${metricLabel(am)}`} color={lineColor} accent={false} />
            <div>
              <div className="eyebrow" style={{ height: HEAD, display: "flex", alignItems: "center", justifyContent: "center" }}>rank moves</div>
              <svg viewBox={`0 0 200 ${ROW * count}`} width="100%" height={ROW * count} preserveAspectRatio="none" style={{ display: "block" }} aria-label="Rank transitions" role="img">
                {Array.from(inView).map((d) => {
                  const a = posA.get(d) ?? 99, b = posB.get(d) ?? 99;
                  const ya = a < count ? a * ROW + ROW / 2 : ROW * count + 20, yb = b < count ? b * ROW + ROW / 2 : ROW * count + 20;
                  const c = lineColor(d);
                  return <path key={d} d={`M0,${ya} C100,${ya} 100,${yb} 200,${yb}`} fill="none" stroke={c} strokeWidth={c === "var(--missed)" ? 1.5 : REFERENCES.includes(d) ? 2 : 3} strokeDasharray={REFERENCES.includes(d) ? "5 4" : undefined} vectorEffect="non-scaling-stroke" />;
                })}
              </svg>
            </div>
            <Bars rows={shownB} title={`B · ${metricLabel(bm)}`} color={lineColor} accent />
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

        {pick && <SeriesPair state={state} file={pick.key} a={pick.winB} b={pick.mover} move={pick.to - pick.from} am={am} bm={bm} rows={tables.rows} />}

        <section className="panel" aria-label="Agreement between metrics">
          <div className="panel-head"><h2>Every metric against every other</h2><span className="muted" style={{ fontSize: 12 }}>weighted tau, {scope || "whole benchmark"}</span><span className="muted" style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: 8, fontSize: 12 }}><span className="mono">{tauText(tLo)}</span><span style={{ width: 90, height: 10, borderRadius: 3, background: `linear-gradient(90deg, ${shade(tLo)}, ${shade(tHi)})` }} /><span className="mono">{tauText(tHi)}</span></span></div>
          <table style={{ borderCollapse: "separate", borderSpacing: 3, fontSize: 12 }}>
            <thead><tr><th className="eyebrow" style={{ width: 100, textAlign: "right", paddingRight: 6, fontWeight: 500 }}>A \ B</th>{pairs.map((k) => <th key={k} className="eyebrow" style={{ width: 56, fontWeight: k === bm ? 700 : 500, textTransform: "none", letterSpacing: 0, color: k === bm ? "var(--accent)" : undefined }}>{metricLabel(k)}</th>)}</tr></thead>
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

function Bars({ rows, title, color, accent }: { rows: { detector: string; value: number }[]; title: string; color: (d: string) => string; accent: boolean }) {
  const max = Math.max(...rows.map((r) => r.value), 1e-9);
  return (
    <div>
      <div className="eyebrow" style={{ height: HEAD, display: "flex", alignItems: "center", color: accent ? "var(--accent)" : undefined }}>{title}</div>
      <div style={{ display: "grid", gridTemplateColumns: "18px 130px minmax(0, 1fr) 36px", gap: "0 8px", alignItems: "center", fontSize: 11, gridAutoRows: ROW }}>
        {rows.map((r, i) => <Row key={r.detector} i={i} r={r} ref_={REFERENCES.includes(r.detector)} c={color(r.detector)} max={max} />)}
      </div>
    </div>
  );
}
function Row({ i, r, ref_, c, max }: { i: number; r: { detector: string; value: number }; ref_: boolean; c: string; max: number }) {
  const strong = c !== "var(--missed)" && c !== "var(--accent)";
  return (
    <>
      <span className="mono muted">{i + 1}</span>
      <span style={{ display: "flex", alignItems: "center", gap: 5, color: strong && !ref_ ? c : undefined, fontWeight: strong && !ref_ ? 600 : 400, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
        {ref_ && <span className="ref-dot" style={{ borderColor: "var(--ink)" }} />}{r.detector}
      </span>
      <span style={{ height: 10, background: "var(--border-soft)", borderRadius: 3, overflow: "hidden" }}>
        <span style={{ display: "block", width: `${Math.max(0, r.value) / max * 100}%`, height: "100%", background: ref_ ? `repeating-linear-gradient(45deg, var(--ink) 0 3px, var(--panel) 3px 6px)` : c }} />
      </span>
      <span className="mono" style={{ textAlign: "right" }}>{f2(r.value)}</span>
    </>
  );
}
