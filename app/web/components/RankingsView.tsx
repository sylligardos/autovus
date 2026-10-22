"use client";
// The Rankings page: two metrics side by side with every rank move drawn
// between them, the datasets where the pair disagrees most, the series
// behind the biggest move, and every metric against every other.

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import { loadScoreDoc, loadSelect, loadSeriesDoc } from "@/lib/api";
import { METRICS, REFERENCES, TYPE_COLOR, f2, fmt, href, metricLabel, shortFile, type MetricKey, type PairRow, type ScoreDoc, type SelectDoc, type SeriesDoc, type State, type Tables } from "@/lib/model";
import { kendall, rankDetectors, ranksPerSeries, reason, toMap } from "@/lib/stats";
import { expand, type CompactTables } from "@/lib/compact";
import { Lane, type Band } from "./Lane";

const MOVER_COLOR: Record<string, string> = { late: "var(--late)", early: "var(--early)", missed: "var(--missed)", squashed: "var(--accent)", exact: "var(--exact)" };
const ROW = 26, HEAD = 22;

export function RankingsView({ state, tables: packed, initial }: { state: State; tables: CompactTables; initial: { a?: string; b?: string } }) {
  const tables = useMemo(() => expand(packed), [packed]);
  const [am, setAm] = useState<MetricKey>((initial.a as MetricKey) ?? "auc_pr");
  const [bm, setBm] = useState<MetricKey>((initial.b as MetricKey) ?? "autovus");
  const [agg, setAgg] = useState<"mean" | "median">("mean");
  const [refs, setRefs] = useState(true);
  const [count, setCount] = useState(14);
  const [scope, setScope] = useState("");
  const [dataset, setDataset] = useState<string | null>(null);
  const [file, setFile] = useState<string | null>(null);

  const rows = useMemo(() => tables.rows.filter((r) => (refs || !REFERENCES.includes(r.detector)) && (!scope || r.dataset === scope)), [tables.rows, refs, scope]);
  const rA = useMemo(() => rankDetectors(rows, am, agg), [rows, am, agg]);
  const rB = useMemo(() => rankDetectors(rows, bm, agg), [rows, bm, agg]);
  const tau = kendall(toMap(rA), toMap(rB));
  const posA = new Map(rA.map((r, i) => [r.detector, i])), posB = new Map(rB.map((r, i) => [r.detector, i]));
  const shownA = rA.slice(0, count), shownB = rB.slice(0, count);
  const inView = new Set([...shownA, ...shownB].map((r) => r.detector));
  const moves = new Map<string, number>();
  for (const d of inView) moves.set(d, (posB.get(d) ?? 99) - (posA.get(d) ?? 99));
  const whyOf = useMemo(() => {
    const by = new Map<string, Record<string, number>>();
    for (const r of rows) {
      const w = by.get(r.detector) ?? { late: 0, early: 0, missed: 0, exact: 0 };
      w.late += r.n_late; w.early += r.n_early; w.missed += r.n_missed; w.exact += r.n_exact;
      by.set(r.detector, w);
    }
    return (d: string) => { const w = by.get(d); if (!w) return "exact"; return Object.entries(w).sort((x, y) => y[1] - x[1])[0][0]; };
  }, [rows]);
  const lineColor = (d: string) => {
    if (REFERENCES.includes(d)) return d === "Random" ? "var(--missed)" : "var(--ink)";
    const m = moves.get(d) ?? 0;
    if (Math.abs(m) >= 3) return MOVER_COLOR[whyOf(d)] ?? "var(--accent)";
    return "var(--border)";
  };
  const big = Array.from(moves.entries()).filter(([d, m]) => Math.abs(m) >= 3 && !REFERENCES.includes(d)).length;

  // Datasets by disagreement.
  const byDataset = useMemo(() => {
    const ds = Array.from(new Set(tables.rows.map((r) => r.dataset))).sort();
    return ds.map((d) => {
      const rs = tables.rows.filter((r) => r.dataset === d && (refs || !REFERENCES.includes(r.detector)));
      const a = rankDetectors(rs, am, agg), b = rankDetectors(rs, bm, agg);
      const pa = new Map(a.map((r, i) => [r.detector, i + 1])), pb = new Map(b.map((r, i) => [r.detector, i + 1]));
      let mover = "", mv = 0, from = 0, to = 0;
      for (const [det, p] of pa) { const q = pb.get(det); if (q !== undefined && Math.abs(q - p) > Math.abs(mv)) { mv = q - p; mover = det; from = p; to = q; } }
      const mix = { late: 0, early: 0, missed: 0, exact: 0 };
      for (const r of rs) { mix.late += r.n_late; mix.early += r.n_early; mix.missed += r.n_missed; mix.exact += r.n_exact; }
      const tot = Object.values(mix).reduce((x, y) => x + y, 0) || 1;
      return { dataset: d, n: new Set(rs.map((r) => r.file)).size, tau: kendall(toMap(a), toMap(b)), winA: a[0]?.detector ?? "", winB: b[0]?.detector ?? "", mover, from, to, mix: Object.fromEntries(Object.entries(mix).map(([k, v]) => [k, v / tot])) };
    }).sort((x, y) => (x.tau ?? 1) - (y.tau ?? 1));
  }, [tables.rows, am, bm, agg, refs]);
  useEffect(() => { if (!dataset && byDataset.length) setDataset(byDataset[0].dataset); }, [byDataset, dataset]);

  // The series of the chosen dataset, by the rank change of their biggest mover.
  const seriesOf = useMemo(() => {
    if (!dataset) return [];
    const rs = tables.rows.filter((r) => r.dataset === dataset && (refs || !REFERENCES.includes(r.detector)));
    const ra = ranksPerSeries(rs, am), rb = ranksPerSeries(rs, bm);
    const out: { file: string; mover: string; from: number; to: number; winner: string; why: string }[] = [];
    for (const [f, pa] of ra) {
      const pb = rb.get(f); if (!pb) continue;
      let mover = "", mv = 0, from = 0, to = 0;
      for (const [det, p] of pa) { const q = pb.get(det); if (q !== undefined && q - p > mv) { mv = q - p; mover = det; from = p; to = q; } }
      if (!mover) continue;
      const winner = Array.from(pb.entries()).find(([, q]) => q === 1)?.[0] ?? "";
      const row = rs.find((r) => r.file === f && r.detector === mover)!;
      out.push({ file: f, mover, from, to, winner, why: reason(row, row[am], row[bm]) });
    }
    return out.sort((x, y) => (y.to - y.from) - (x.to - x.from)).slice(0, 6);
  }, [tables.rows, dataset, am, bm, refs]);
  useEffect(() => { setFile(seriesOf[0]?.file ?? null); }, [seriesOf]);
  const pick = seriesOf.find((s) => s.file === file) ?? null;

  const pairs = useMemo(() => METRICS.map((m) => m.key), []);
  const taus = useMemo(() => {
    const ranks = Object.fromEntries(pairs.map((k) => [k, toMap(rankDetectors(rows, k, agg))])) as Record<MetricKey, Map<string, number>>;
    return pairs.map((x) => pairs.map((y) => kendall(ranks[x], ranks[y])));
  }, [rows, agg, pairs]);
  const shade = (t: number | null) => (t === null ? "var(--border-soft)" : t >= 0.8 ? "var(--accent)" : t >= 0.7 ? "#5B7FD6" : t >= 0.6 ? "#A3B6E8" : t >= 0.5 ? "#D5DFF5" : "var(--border-soft)");

  return (
    <>
      <aside className="rail" aria-label="Controls">
        <div className="field"><span>Scope</span>
          <select className="select" aria-label="Scope" value={scope} onChange={(e) => setScope(e.target.value)}><option value="">whole benchmark</option>{byDataset.map((d) => <option key={d.dataset} value={d.dataset}>{d.dataset}</option>)}</select>
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
          <div className="panel-head">
            <h2>{metricLabel(am)} <span className="muted" style={{ fontWeight: 400 }}>to</span> <span style={{ color: "var(--accent)" }}>{metricLabel(bm)}</span></h2>
            <span className="muted" style={{ fontSize: 12 }}>{agg} over {fmt(new Set(rows.map((r) => r.file)).size)} series · L = {tables.L}</span>
            <span className="mono muted" style={{ marginLeft: "auto", fontSize: 12 }}>Kendall tau {tau === null ? "–" : tau.toFixed(2)} · {big} detectors move 3 places or more</span>
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "minmax(0, 5fr) minmax(0, 3fr) minmax(0, 5fr)", gap: "0 12px", alignItems: "start" }}>
            <Bars rows={shownA} title={`A · ${metricLabel(am)}`} color={(d) => lineColor(d) === "var(--border)" ? "var(--missed)" : lineColor(d)} accent={false} />
            <div>
              <div className="eyebrow" style={{ height: HEAD, display: "flex", alignItems: "center", justifyContent: "center" }}>rank moves</div>
              <svg viewBox={`0 0 200 ${ROW * count}`} width="100%" height={ROW * count} preserveAspectRatio="none" style={{ display: "block" }} aria-label="Rank transitions" role="img">
                {Array.from(inView).map((d) => {
                  const a = posA.get(d) ?? 99, b = posB.get(d) ?? 99;
                  const ya = a < count ? a * ROW + ROW / 2 : ROW * count + 20, yb = b < count ? b * ROW + ROW / 2 : ROW * count + 20;
                  const c = lineColor(d);
                  return <path key={d} d={`M0,${ya} C100,${ya} 100,${yb} 200,${yb}`} fill="none" stroke={c} strokeWidth={c === "var(--border)" ? 1.5 : REFERENCES.includes(d) ? 2 : 3} strokeDasharray={REFERENCES.includes(d) ? "5 4" : undefined} vectorEffect="non-scaling-stroke" />;
                })}
              </svg>
            </div>
            <Bars rows={shownB} title={`B · ${metricLabel(bm)}`} color={(d) => lineColor(d) === "var(--border)" ? "var(--accent)" : lineColor(d)} accent />
          </div>
          <div className="muted" style={{ display: "flex", gap: 14, flexWrap: "wrap", fontSize: 12, paddingTop: 4, borderTop: "1px solid var(--border-soft)" }}>
            <span style={{ display: "flex", alignItems: "center", gap: 6 }}><span className="ref-dot" style={{ borderColor: "var(--ink)" }} />Delayed oracle: the label shifted {String(tables.summary?.oracle_delay ?? 64)} points later</span>
            <span style={{ display: "flex", alignItems: "center", gap: 6 }}><span className="ref-dot" style={{ borderColor: "var(--muted)" }} />Random: a uniform score</span>
            {Object.entries(MOVER_COLOR).filter(([k]) => k !== "squashed").map(([k, c]) => <span key={k} style={{ display: "flex", alignItems: "center", gap: 6 }}><span style={{ width: 14, height: 3, background: c }} />moves 3 or more, {k}</span>)}
          </div>
        </section>

        <div style={{ display: "flex", flexWrap: "wrap", gap: 16, alignItems: "stretch" }}>
          <section className="panel" style={{ flex: "1 1 460px", minWidth: 0 }} aria-label="Datasets by change">
            <div className="panel-head"><h2>Where the pair disagrees</h2></div>
            <div style={{ overflowX: "auto" }}>
              <table className="table">
                <thead><tr><th>dataset</th><th className="r">series</th><th className="r">tau</th><th>winner A</th><th>winner B</th><th>biggest move</th><th style={{ minWidth: 90 }}>why</th></tr></thead>
                <tbody>
                  {byDataset.map((d) => (
                    <tr key={d.dataset} className={d.dataset === dataset ? "picked" : undefined} onClick={() => setDataset(d.dataset)}>
                      <td style={{ fontWeight: 600 }}>{d.dataset}</td><td className="r mono">{d.n}</td><td className="r mono">{d.tau === null ? "–" : d.tau.toFixed(2)}</td>
                      <td style={{ color: d.winA !== d.winB ? "var(--late)" : undefined }}>{d.winA}</td><td>{d.winB}</td>
                      <td className="mono">{d.mover} {d.from} to {d.to}</td>
                      <td><span style={{ display: "flex", height: 8, borderRadius: 4, overflow: "hidden" }}>{(["late", "early", "missed", "exact"] as const).map((k) => <span key={k} style={{ width: `${d.mix[k] * 100}%`, background: TYPE_COLOR[k] }} />)}</span></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="muted" style={{ display: "flex", gap: 10, fontSize: 12, justifyContent: "flex-end" }}>
              {(["late", "early", "missed", "exact"] as const).map((k) => <span key={k} style={{ display: "flex", alignItems: "center", gap: 4 }}><span className="swatch" style={{ width: 8, height: 8, background: TYPE_COLOR[k] }} />{k}</span>)}
            </div>
          </section>
          <section className="panel panel-accent" style={{ flex: "0 1 300px", minWidth: 240 }} aria-label="Series of the chosen dataset">
            <div style={{ display: "flex", flexDirection: "column", gap: 2 }}><h2>{dataset ?? "…"} · {byDataset.find((d) => d.dataset === dataset)?.n ?? 0} series</h2><span className="muted" style={{ fontSize: 12 }}>by rank change of the biggest mover</span></div>
            <ol style={{ margin: 0, padding: 0, listStyle: "none", display: "flex", flexDirection: "column", gap: 4, fontSize: 12 }}>
              {seriesOf.map((s) => (
                <li key={s.file} onClick={() => setFile(s.file)} style={{ display: "grid", gridTemplateColumns: "minmax(0, 1fr) auto", gap: 8, padding: "7px 10px", borderRadius: 6, cursor: "pointer", background: s.file === file ? "var(--ink)" : undefined, color: s.file === file ? "var(--panel)" : undefined, border: s.file === file ? undefined : "1px solid var(--border-soft)" }}>
                  <span className="mono">{shortFile(s.file)}</span><span style={{ color: MOVER_COLOR[s.why] }}>{s.mover} {s.from} to {s.to}</span>
                </li>
              ))}
            </ol>
            {dataset && <Link href={href("/cases", state, { d: dataset, x: am, y: bm })} style={{ marginTop: "auto", fontSize: 12, fontWeight: 600, color: "var(--accent)" }}>All {dataset} cases</Link>}
          </section>
        </div>

        {pick && <Swap state={state} file={pick.file} a={pick.winner} b={pick.mover} am={am} bm={bm} rows={tables.rows} L={tables.L} />}

        <section className="panel" aria-label="Agreement between metrics">
          <div className="panel-head"><h2>Every metric against every other</h2><span className="muted" style={{ fontSize: 12 }}>Kendall tau, {scope || "whole benchmark"}</span></div>
          <table style={{ borderCollapse: "separate", borderSpacing: 3, fontSize: 12 }}>
            <thead><tr><th style={{ width: 100 }} />{pairs.map((k) => <th key={k} className="eyebrow" style={{ width: 56, fontWeight: 500, textTransform: "none", letterSpacing: 0, color: METRICS.find((m) => m.key === k)?.accent ? "var(--accent)" : undefined }}>{metricLabel(k)}</th>)}</tr></thead>
            <tbody className="mono" style={{ textAlign: "center" }}>
              {pairs.map((x, i) => (
                <tr key={x}>
                  <th scope="row" className="sans" style={{ textAlign: "right", fontWeight: 500, color: METRICS.find((m) => m.key === x)?.accent ? "var(--accent)" : "var(--muted)", fontSize: 11, paddingRight: 6 }}>{metricLabel(x)}</th>
                  {pairs.slice(0, i + 1).map((y, j) => {
                    const t = i === j ? 1 : taus[i][j];
                    const sel = (x === bm && y === am) || (x === am && y === bm);
                    return <td key={y} onClick={() => { if (i !== j) { setAm(y); setBm(x); } }} style={{ height: 40, borderRadius: 4, cursor: i !== j ? "pointer" : "default", background: i === j ? "var(--ink)" : shade(t), color: i === j || (t ?? 0) >= 0.7 ? "var(--panel)" : "var(--ink)", outline: sel ? "3px solid var(--ink)" : undefined, outlineOffset: -1, fontWeight: sel ? 600 : 400 }}>{t === null ? "–" : i === j ? "1" : t.toFixed(2).slice(1)}</td>;
                  })}
                  {i === 0 && (
                    <td colSpan={pairs.length - 1} rowSpan={pairs.length - 1} className="sans muted" style={{ verticalAlign: "top", textAlign: "left", padding: "4px 8px 0 24px", fontSize: 12 }}>
                      <span style={{ display: "flex", alignItems: "center", gap: 8 }}><span style={{ display: "inline-flex", gap: 2 }}>{["var(--border-soft)", "#D5DFF5", "#A3B6E8", "#5B7FD6", "var(--accent)"].map((c) => <span key={c} className="swatch" style={{ width: 14, height: 14, background: c }} />)}</span>tau from 0.4 to 0.9</span>
                    </td>
                  )}
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
      <div style={{ display: "grid", gridTemplateColumns: "18px 110px minmax(0, 1fr) 36px", gap: "0 8px", alignItems: "center", fontSize: 12, gridAutoRows: ROW }}>
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
        {ref_ && <span className="ref-dot" style={{ borderColor: r.detector === "Random" ? "var(--muted)" : "var(--ink)" }} />}{r.detector}
      </span>
      <span style={{ height: 12, background: "var(--border-soft)", borderRadius: 3, overflow: "hidden" }}>
        <span style={{ display: "block", width: `${Math.max(0, r.value) / max * 100}%`, height: "100%", background: ref_ ? `repeating-linear-gradient(45deg, ${r.detector === "Random" ? "var(--missed)" : "var(--ink)"} 0 3px, var(--panel) 3px 6px)` : c }} />
      </span>
      <span className="mono" style={{ textAlign: "right" }}>{f2(r.value)}</span>
    </>
  );
}

function Swap({ state, file, a, b, am, bm, rows, L }: { state: State; file: string; a: string; b: string; am: MetricKey; bm: MetricKey; rows: PairRow[]; L: number }) {
  const [doc, setDoc] = useState<SeriesDoc | null>(null);
  const [lanes, setLanes] = useState<Record<string, { score: ScoreDoc; sel: SelectDoc }>>({});
  useEffect(() => {
    let live = true;
    setDoc(null); setLanes({});
    loadSeriesDoc(state.benchmark, file).then((d) => { if (live) setDoc(d); });
    for (const det of Array.from(new Set([a, b]))) {
      Promise.all([loadScoreDoc(state.benchmark, file, det), loadSelect(state.benchmark, file, det, L, "per_side")]).then(([score, sel]) => {
        if (live && score && sel) setLanes((m) => ({ ...m, [det]: { score, sel } }));
      });
    }
    return () => { live = false; };
  }, [state.benchmark, file, a, b, L]);
  const labels: Band[] = (doc?.labels ?? []).map(([s, e], i) => ({ from: s, to: e + 1, color: "var(--label)", opacity: 0.45, key: `l${i}` }));
  const rowOf = (det: string) => rows.find((r) => r.file === file && r.detector === det);
  return (
    <section className="panel" aria-label="The swap on the chosen series">
      <div className="panel-head">
        <h2 style={{ textTransform: "none" }}>{file.replace(/\.csv$/, "")}</h2>
        {doc && <span className="muted" style={{ fontSize: 12 }}>{fmt(doc.n)} points · {doc.labels.length} anomalies</span>}
        <Link href={href("/series", state, { s: file, pin: Array.from(new Set([a, b])).join(","), top: "0" })} style={{ marginLeft: "auto", fontSize: 12, fontWeight: 600, color: "var(--accent)" }}>Open in Series with both pinned</Link>
      </div>
      <div className="lane-row"><div className="lane-name"><span style={{ fontWeight: 600 }}>value</span><span className="muted" style={{ fontSize: 11, display: "flex", alignItems: "center", gap: 6 }}><span className="swatch" style={{ width: 10, height: 8, background: "var(--label)" }} />labelled anomaly</span></div>
        {doc ? <Lane lo={doc.lo} hi={doc.hi} bins={doc.bins} n={doc.n} height={100} bands={labels} /> : <div style={{ height: 100, background: "var(--subtle)", borderRadius: 4 }} />}</div>
      {Array.from(new Set([a, b])).map((det) => {
        const l = lanes[det], r = rowOf(det);
        const bands: Band[] = [...labels];
        if (l) for (const x of l.sel.anomalies) {
          const c = TYPE_COLOR[x.type];
          if (x.type === "missed") bands.push({ from: x.start, to: x.end + 1, color: "var(--missed)", opacity: 0.55, key: `m${x.k}` });
          if (x.b_left > 0) bands.push({ from: x.start - x.b_left, to: x.start, color: c, opacity: 0.4, key: `bl${x.k}` });
          if (x.b_right > 0) bands.push({ from: x.end + 1, to: x.end + 1 + x.b_right, color: c, opacity: 0.4, key: `br${x.k}` });
        }
        const types = l ? l.sel.anomalies.reduce((m, x) => ({ ...m, [x.type]: (m[x.type] ?? 0) + 1 }), {} as Record<string, number>) : {};
        const main = Object.entries(types).sort((x, y) => y[1] - x[1])[0];
        return (
          <div key={det} className="lane-row">
            <div className="lane-name">
              <span style={{ fontWeight: 600, color: det === b ? "var(--late)" : undefined }}>{det}</span>
              <span className="mono muted" style={{ fontSize: 11 }}>{r ? `${metricLabel(am)} ${f2(r[am])} · ${metricLabel(bm)} ${f2(r[bm])}` : "…"}</span>
              {main && <span style={{ fontSize: 11, fontWeight: 600, color: TYPE_COLOR[main[0] as keyof typeof TYPE_COLOR] }}>{main[0]} on {main[1]} of {l!.sel.anomalies.length}</span>}
            </div>
            {l ? <Lane lo={l.score.lo} hi={l.score.hi} bins={l.score.bins} n={l.score.n} ymin={0} ymax={1} bands={bands} /> : <div style={{ height: 56, background: "var(--subtle)", borderRadius: 4 }} />}
          </div>
        );
      })}
    </section>
  );
}
