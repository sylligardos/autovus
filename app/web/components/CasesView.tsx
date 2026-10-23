"use client";
// The Cases page: every (series, detector) as a point under two metrics,
// coloured by how its rank on the series moved, the biggest moves ranked,
// and the picked case drawn below. The reference detectors are left out.

import { useEffect, useMemo, useRef, useState } from "react";
import { METRICS, REFERENCES, f2, fmt, metricLabel, shortFile, type MetricKey, type PairRow, type State } from "@/lib/model";
import { ranksPerSeries } from "@/lib/stats";
import { expand, type CompactTables } from "@/lib/compact";
import { useRemembered } from "@/lib/remember";
import { SeriesPair } from "./SeriesPair";

type Case = PairRow & { x: number; y: number; rx: number; ry: number; change: number; dist: number };
// Up the ranking from x to y in green, down in red, no move in grey.
const moveColor = (change: number) => (change < 0 ? "var(--up)" : change > 0 ? "var(--down)" : "var(--missed)");

export function CasesView({ state, tables: packed, initial }: { state: State; tables: CompactTables; initial: { x?: string; y?: string; d?: string } }) {
  const rows = useMemo(() => expand(packed).rows.filter((r) => !REFERENCES.includes(r.detector)), [packed]);
  const datasets = useMemo(() => Array.from(new Set(rows.map((r) => r.dataset))).sort(), [rows]);
  const k = `cases:${state.benchmark}:`;
  const [xm, setXm] = useRemembered<MetricKey>(k + "x", "auc_pr", initial.x as MetricKey);
  const [ym, setYm] = useRemembered<MetricKey>(k + "y", "autovus", initial.y as MetricKey);
  const [score, setScore] = useRemembered<"change" | "dist">(k + "score", "change");
  const [dataset, setDataset] = useRemembered(k + "d", "", initial.d);
  const [filter, setFilter] = useRemembered(k + "filter", "");
  const [limit, setLimit] = useRemembered(k + "limit", 20);
  const [hover, setHover] = useState<Case | null>(null);
  const [picked, setPicked] = useState<Case | null>(null);
  const canvas = useRef<HTMLCanvasElement>(null);

  const scoped = useMemo(() => (dataset ? rows.filter((r) => r.dataset === dataset) : rows), [rows, dataset]);
  const cases = useMemo<Case[]>(() => {
    const rx = ranksPerSeries(scoped, xm), ry = ranksPerSeries(scoped, ym);
    return scoped.filter((r) => r[xm] !== null && r[ym] !== null).map((r) => {
      const x = r[xm] as number, y = r[ym] as number;
      const a = rx.get(r.file)?.get(r.detector) ?? 0, b = ry.get(r.file)?.get(r.detector) ?? 0;
      return { ...r, x, y, rx: a, ry: b, change: b - a, dist: Math.abs(y - x) / Math.SQRT2 };
    });
  }, [scoped, xm, ym]);
  // Rank change weighted to the top: |1/rank x - 1/rank y|, the 1/rank weight of the weighted tau.
  const key = score === "change" ? (c: Case) => (c.rx && c.ry ? Math.abs(1 / c.rx - 1 / c.ry) : 0) : (c: Case) => c.dist;
  const ranked = useMemo(() => {
    const f = filter.toLowerCase();
    return cases.filter((c) => !f || c.file.toLowerCase().includes(f) || c.detector.toLowerCase().includes(f))
      .sort((a, b) => key(b) - key(a) || Math.abs(b.change) - Math.abs(a.change) || b.dist - a.dist);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cases, filter, score]);
  // The cloud on a canvas, the highlighted on top in SVG.
  const W = 720, H = 520, P = 36;
  const sx = (v: number) => P + v * (W - 2 * P), sy = (v: number) => H - P - v * (H - 2 * P);
  useEffect(() => {
    const el = canvas.current; if (!el) return;
    const ctx = el.getContext("2d"); if (!ctx) return;
    ctx.clearRect(0, 0, W, H);
    const css = getComputedStyle(el), col = (v: string, d: string) => css.getPropertyValue(v).trim() || d;
    const fill = { up: col("--up", "#45A86C"), down: col("--down", "#D86A62"), same: col("--missed", "#8A96A8") };
    ctx.globalAlpha = 0.35;
    // Unmoved first, so the moves sit on top.
    for (const pass of ["same", "move"] as const) for (const c of cases) {
      if ((pass === "same") !== (c.change === 0)) continue;
      ctx.fillStyle = c.change < 0 ? fill.up : c.change > 0 ? fill.down : fill.same;
      ctx.beginPath(); ctx.arc(sx(c.x), sy(c.y), 2.2, 0, Math.PI * 2); ctx.fill();
    }
    ctx.globalAlpha = 1;
  }, [cases]);
  // The big dots are the rows loaded in the table.
  const top = ranked.slice(0, limit);
  const onMove = (e: React.MouseEvent<SVGSVGElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    const px = ((e.clientX - r.left) / r.width) * W, py = ((e.clientY - r.top) / r.height) * H;
    let best: Case | null = null, bd = 12 * 12;
    for (const c of cases) { const d = (sx(c.x) - px) ** 2 + (sy(c.y) - py) ** 2; if (d < bd) { bd = d; best = c; } }
    setHover(best);
  };
  const id = (c: Case | null) => (c ? `${c.file}|${c.detector}` : "");
  const pick = cases.find((c) => id(c) === id(picked)) ?? ranked[0] ?? null;
  // The picked case is drawn with the detector ranked first under y on its series.
  const winner = pick ? cases.filter((c) => c.file === pick.file).reduce((w, c) => (c.y > w.y ? c : w), pick).detector : "";

  return (
    <>
      <aside className="rail" aria-label="Controls">
        <div className="field">
          <span>Scope</span>
          <select className="select" aria-label="Dataset" value={dataset} onChange={(e) => setDataset(e.target.value)}>
            <option value="">whole benchmark</option>
            {datasets.map((d) => <option key={d} value={d}>{d}</option>)}
          </select>
        </div>
        <div className="field">
          <span>Compare</span>
          <label className="muted" style={{ fontSize: 12, display: "flex", flexDirection: "column", gap: 3 }}>x axis
            <select className="select" style={{ height: 32 }} value={xm} onChange={(e) => setXm(e.target.value as MetricKey)}>{METRICS.map((m) => <option key={m.key} value={m.key}>{m.label}</option>)}</select>
          </label>
          <label className="muted" style={{ fontSize: 12, display: "flex", flexDirection: "column", gap: 3 }}>y axis
            <select className="select" style={{ height: 32, fontWeight: 600 }} value={ym} onChange={(e) => setYm(e.target.value as MetricKey)}>{METRICS.map((m) => <option key={m.key} value={m.key}>{m.label}</option>)}</select>
          </label>
        </div>
        <div className="field">
          <span>Interest score</span>
          <span className="seg" role="group" aria-label="Interest score">
            <button type="button" aria-pressed={score === "change"} onClick={() => setScore("change")}>Rank change</button>
            <button type="button" aria-pressed={score === "dist"} onClick={() => setScore("dist")}>Distance</button>
          </span>
        </div>
        <div className="field" style={{ gap: 6, paddingTop: 12, borderTop: "1px solid var(--border)" }}>
          <span>Shortcuts</span>
          <button type="button" className="btn" style={{ textAlign: "left" }} onClick={() => { const pool = ranked.slice(0, 200); setPicked(pool[Math.floor(Math.random() * pool.length)] ?? null); }}>Surprise me</button>
          <button type="button" className="btn" style={{ textAlign: "left" }} onClick={() => { setScore("change"); setPicked(ranked[0] ?? null); }}>Biggest swing</button>
        </div>
      </aside>

      <main className="content">
        <div style={{ display: "grid", gridTemplateColumns: "minmax(0, 2fr) minmax(0, 1fr)", gap: 16, alignItems: "stretch" }}>
          <section className="panel" aria-label="Scatter">
            <div style={{ position: "relative" }}>
              <canvas ref={canvas} width={W} height={H} style={{ position: "absolute", inset: 0, width: "100%", height: "100%", background: "var(--subtle)", borderRadius: 4 }} aria-hidden="true" />
              <svg viewBox={`0 0 ${W} ${H}`} width="100%" height="auto" style={{ display: "block", position: "relative" }} role="img" aria-label="Scatter of every series and detector pair" onMouseMove={onMove} onMouseLeave={() => setHover(null)} onClick={() => { if (hover) setPicked(hover); }}>
                <line x1={P} y1={H - P} x2={W - P} y2={H - P} stroke="var(--border)" /><line x1={P} y1={H - P} x2={P} y2={P} stroke="var(--border)" />
                <line x1={P} y1={H - P} x2={W - P} y2={P} stroke="var(--ink)" strokeDasharray="4 4" opacity={0.5} />
                {[0, 0.5, 1].map((v) => <text key={`x${v}`} x={sx(v)} y={H - P + 14} fontFamily="var(--mono)" fontSize={10} fill="var(--muted)" textAnchor="middle">{v}</text>)}
                {[0.5, 1].map((v) => <text key={`y${v}`} x={P - 6} y={sy(v) + 4} fontFamily="var(--mono)" fontSize={10} fill="var(--muted)" textAnchor="end">{v}</text>)}
                <text x={W / 2} y={H - 6} fontFamily="var(--sans)" fontSize={11} fill="var(--muted)" textAnchor="middle">{metricLabel(xm)}</text>
                <text x={12} y={H / 2} fontFamily="var(--sans)" fontSize={11} fill="var(--muted)" textAnchor="middle" transform={`rotate(-90 12 ${H / 2})`}>{metricLabel(ym)}</text>
                {top.map((c) => <circle key={`${c.file}|${c.detector}`} cx={sx(c.x)} cy={sy(c.y)} r={5.5} fill={moveColor(c.change)} stroke="var(--panel)" strokeWidth={1.5} />)}
                {hover && <circle cx={sx(hover.x)} cy={sy(hover.y)} r={6.5} fill="none" stroke="var(--ink)" strokeWidth={1.5} />}
                {pick && <circle cx={sx(pick.x)} cy={sy(pick.y)} r={8} fill={moveColor(pick.change)} stroke="var(--ink)" strokeWidth={2} />}
              </svg>
              {hover && (
                <div className="hover-card" style={{ left: `${(sx(hover.x) / W) * 100}%`, top: `${(sy(hover.y) / H) * 100}%`, bottom: "auto", transform: "translate(-50%, calc(-100% - 10px))" }}>
                  <span><b>{hover.detector}</b> on {shortFile(hover.file)}</span>
                  <span className="mono">{metricLabel(xm)} {f2(hover.x)} · {metricLabel(ym)} {f2(hover.y)} · rank <span style={{ color: moveColor(hover.change), fontWeight: 600 }}>{hover.rx} to {hover.ry}</span></span>
                </div>
              )}
            </div>
          </section>

          <section className="panel" aria-label="Ranked cases" style={{ minHeight: 0 }}>
            <div className="panel-head">
              <h2>Most interesting first</h2>
              <span className="muted" style={{ fontSize: 12 }}>{fmt(ranked.length)}</span>
            </div>
            <input type="search" className="input" placeholder="filter by series or detector" aria-label="Filter cases" value={filter} onChange={(e) => setFilter(e.target.value)} style={{ height: 30, fontSize: 12 }} />
            <div style={{ flex: "1 1 0", minHeight: 0, overflowY: "auto" }}>
              <table className="table" style={{ fontSize: 12 }}>
                <thead><tr><th>series</th><th>detector</th><th>rank</th></tr></thead>
                <tbody className="mono">
                  {ranked.slice(0, limit).map((c) => (
                    <tr key={id(c)} className={id(c) === id(pick) ? "picked" : undefined} onClick={() => setPicked(c)}>
                      <td className="sans" style={{ whiteSpace: "nowrap" }} title={c.file}>{shortFile(c.file).split(" · ")[0]}</td><td className="sans" style={{ fontWeight: 600 }}>{c.detector}</td>
                      <td style={{ whiteSpace: "nowrap", color: moveColor(c.change) }}>{c.rx} to {c.ry}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {limit < ranked.length && <button type="button" className="btn" style={{ marginTop: 8 }} onClick={() => setLimit(limit + 20)}>Show 20 more</button>}
            </div>
          </section>
        </div>

        {pick && <SeriesPair state={state} file={pick.file} a={winner} b={pick.detector} move={pick.change} am={xm} bm={ym} rows={rows} />}
      </main>
    </>
  );
}
