"use client";
// The Cases page: every (series, detector) as a point under two metrics,
// the ones far from the diagonal ranked, each with its reason.

import Link from "next/link";
import { useEffect, useMemo, useRef, useState } from "react";
import { METRICS, TYPE_COLOR, f2, fmt, href, metricLabel, shortFile, type MetricKey, type PairRow, type State } from "@/lib/model";
import { ranksPerSeries, reason } from "@/lib/stats";
import { expand, type CompactTables } from "@/lib/compact";

const REASON_COLOR: Record<string, string> = { late: "var(--late)", early: "var(--early)", missed: "var(--missed)", squashed: "var(--accent)", exact: "var(--exact)" };
type Case = PairRow & { x: number; y: number; rx: number; ry: number; change: number; dist: number; why: string };

export function CasesView({ state, tables: packed, initial }: { state: State; tables: CompactTables; initial: { x?: string; y?: string; d?: string } }) {
  const rows = useMemo(() => expand(packed).rows, [packed]);
  const datasets = useMemo(() => Array.from(new Set(rows.map((r) => r.dataset))).sort(), [rows]);
  const [xm, setXm] = useState<MetricKey>((initial.x as MetricKey) ?? "auc_pr");
  const [ym, setYm] = useState<MetricKey>((initial.y as MetricKey) ?? "autovus");
  const [score, setScore] = useState<"change" | "dist">("change");
  const [dataset, setDataset] = useState(initial.d ?? "");
  const [why, setWhy] = useState<Record<string, boolean>>({ late: true, early: true, missed: true, squashed: true, exact: true });
  const [filter, setFilter] = useState("");
  const [limit, setLimit] = useState(20);
  const [hover, setHover] = useState<Case | null>(null);
  const [picked, setPicked] = useState<Case | null>(null);
  const canvas = useRef<HTMLCanvasElement>(null);

  const scoped = useMemo(() => (dataset ? rows.filter((r) => r.dataset === dataset) : rows), [rows, dataset]);
  const cases = useMemo<Case[]>(() => {
    const rx = ranksPerSeries(scoped, xm), ry = ranksPerSeries(scoped, ym);
    return scoped.filter((r) => r[xm] !== null && r[ym] !== null).map((r) => {
      const x = r[xm] as number, y = r[ym] as number;
      const a = rx.get(r.file)?.get(r.detector) ?? 0, b = ry.get(r.file)?.get(r.detector) ?? 0;
      return { ...r, x, y, rx: a, ry: b, change: b - a, dist: Math.abs(y - x) / Math.SQRT2, why: reason(r, x, y) };
    });
  }, [scoped, xm, ym]);
  const key = score === "change" ? (c: Case) => Math.abs(c.change) : (c: Case) => c.dist;
  const ranked = useMemo(() => {
    const f = filter.toLowerCase();
    return cases.filter((c) => why[c.why] && (!f || c.file.toLowerCase().includes(f) || c.detector.toLowerCase().includes(f)))
      .sort((a, b) => key(b) - key(a) || b.dist - a.dist);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cases, why, filter, score]);
  const off = cases.filter((c) => c.dist > 0.05 / Math.SQRT2).length;
  const winners = useMemo(() => {
    const best = new Map<string, { a: string; b: string; va: number; vb: number }>();
    for (const c of cases) {
      const w = best.get(c.file) ?? { a: "", b: "", va: -1, vb: -1 };
      if (c.x > w.va) { w.a = c.detector; w.va = c.x; }
      if (c.y > w.vb) { w.b = c.detector; w.vb = c.y; }
      best.set(c.file, w);
    }
    let changed = 0;
    for (const w of best.values()) if (w.a !== w.b) changed++;
    return { changed, n: best.size };
  }, [cases]);
  const mix = useMemo(() => {
    const m: Record<string, number> = { late: 0, early: 0, missed: 0, squashed: 0, exact: 0 };
    for (const c of cases) if (c.dist > 0.05 / Math.SQRT2) m[c.why]++;
    const t = Object.values(m).reduce((a, b) => a + b, 0) || 1;
    return Object.entries(m).map(([k, v]) => [k, v / t] as const);
  }, [cases]);

  // The cloud on a canvas, the highlighted on top in SVG.
  const W = 720, H = 520, P = 36;
  const sx = (v: number) => P + v * (W - 2 * P), sy = (v: number) => H - P - v * (H - 2 * P);
  useEffect(() => {
    const el = canvas.current; if (!el) return;
    const ctx = el.getContext("2d"); if (!ctx) return;
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = getComputedStyle(el).getPropertyValue("--missed") || "#8A96A8";
    ctx.globalAlpha = 0.35;
    for (const c of cases) { ctx.beginPath(); ctx.arc(sx(c.x), sy(c.y), 2.2, 0, Math.PI * 2); ctx.fill(); }
    ctx.globalAlpha = 1;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cases]);
  const top = ranked.slice(0, 60);
  const onMove = (e: React.MouseEvent<SVGSVGElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    const px = ((e.clientX - r.left) / r.width) * W, py = ((e.clientY - r.top) / r.height) * H;
    let best: Case | null = null, bd = 12 * 12;
    for (const c of top) { const d = (sx(c.x) - px) ** 2 + (sy(c.y) - py) ** 2; if (d < bd) { bd = d; best = c; } }
    setHover(best);
  };
  const open = (c: Case) => href("/series", state, { s: c.file, pin: c.detector, top: "3" });

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
        <div className="field" style={{ gap: 6 }}>
          <span>Why</span>
          {Object.keys(why).map((k) => (
            <label key={k} style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 12 }}>
              <input type="checkbox" checked={why[k]} onChange={(e) => setWhy({ ...why, [k]: e.target.checked })} /><span className="swatch" style={{ width: 10, height: 10, background: REASON_COLOR[k] }} />{k}
            </label>
          ))}
        </div>
        <div className="field" style={{ gap: 6, paddingTop: 12, borderTop: "1px solid var(--border)" }}>
          <span>Shortcuts</span>
          <button type="button" className="btn" style={{ textAlign: "left" }} onClick={() => { const pool = ranked.slice(0, 200); setPicked(pool[Math.floor(Math.random() * pool.length)] ?? null); }}>Surprise me</button>
          <button type="button" className="btn" style={{ textAlign: "left" }} onClick={() => { setScore("change"); setPicked(ranked[0] ?? null); }}>Biggest swing</button>
          <button type="button" className="btn" style={{ textAlign: "left" }} onClick={() => { setWhy({ late: true, early: false, missed: false, squashed: false, exact: false }); setPicked(null); }}>Most lag</button>
          <button type="button" className="btn" style={{ textAlign: "left" }} onClick={() => { setWhy({ late: false, early: false, missed: false, squashed: true, exact: false }); setPicked(null); }}>Where {metricLabel(xm)} says 1.0</button>
        </div>
      </aside>

      <main className="content">
        <div style={{ display: "flex", flexWrap: "wrap", gap: 16, alignItems: "stretch" }}>
          <section className="panel" style={{ flex: "1 1 480px", minWidth: 0 }} aria-label="Scatter">
            <div className="panel-head">
              <h2>{metricLabel(xm)} against {metricLabel(ym)}</h2>
              <span className="muted" style={{ fontSize: 12 }}>{fmt(cases.length)} pairs · L = 256</span>
              <span className="muted" style={{ marginLeft: "auto", fontSize: 12 }}>grey: within 0.05 of the diagonal</span>
            </div>
            <div style={{ position: "relative" }}>
              <canvas ref={canvas} width={W} height={H} style={{ position: "absolute", inset: 0, width: "100%", height: "100%", background: "var(--subtle)", borderRadius: 4 }} aria-hidden="true" />
              <svg viewBox={`0 0 ${W} ${H}`} width="100%" height="auto" style={{ display: "block", position: "relative" }} role="img" aria-label="Scatter of every series and detector pair" onMouseMove={onMove} onMouseLeave={() => setHover(null)} onClick={() => { if (hover) setPicked(hover); }}>
                <line x1={P} y1={H - P} x2={W - P} y2={H - P} stroke="var(--border)" /><line x1={P} y1={H - P} x2={P} y2={P} stroke="var(--border)" />
                <line x1={P} y1={H - P} x2={W - P} y2={P} stroke="var(--ink)" strokeDasharray="4 4" opacity={0.5} />
                {[0, 0.5, 1].map((v) => <text key={`x${v}`} x={sx(v)} y={H - P + 14} fontFamily="var(--mono)" fontSize={10} fill="var(--muted)" textAnchor="middle">{v}</text>)}
                {[0.5, 1].map((v) => <text key={`y${v}`} x={P - 6} y={sy(v) + 4} fontFamily="var(--mono)" fontSize={10} fill="var(--muted)" textAnchor="end">{v}</text>)}
                <text x={W / 2} y={H - 6} fontFamily="var(--sans)" fontSize={11} fill="var(--muted)" textAnchor="middle">{metricLabel(xm)}</text>
                <text x={12} y={H / 2} fontFamily="var(--sans)" fontSize={11} fill="var(--muted)" textAnchor="middle" transform={`rotate(-90 12 ${H / 2})`}>{metricLabel(ym)}</text>
                {top.map((c) => <circle key={`${c.file}|${c.detector}`} cx={sx(c.x)} cy={sy(c.y)} r={picked === c ? 8 : 5.5} fill={REASON_COLOR[c.why]} stroke="var(--panel)" strokeWidth={1.5} />)}
              </svg>
              {hover && (
                <div className="hover-card" style={{ left: `${(sx(hover.x) / W) * 100}%`, top: `${(sy(hover.y) / H) * 100}%`, bottom: "auto", transform: "translate(-50%, calc(-100% - 10px))" }}>
                  <span><b>{hover.detector}</b> on {shortFile(hover.file)}</span>
                  <span className="mono">{metricLabel(xm)} {f2(hover.x)} · {metricLabel(ym)} {f2(hover.y)} · rank {hover.rx} to {hover.ry} · <span style={{ color: REASON_COLOR[hover.why] }}>{hover.why}</span></span>
                </div>
              )}
            </div>
            <div className="muted" style={{ display: "flex", gap: 14, flexWrap: "wrap", fontSize: 12 }}>
              {Object.entries(REASON_COLOR).map(([k, c]) => <span key={k} style={{ display: "flex", alignItems: "center", gap: 6 }}><span className="swatch" style={{ borderRadius: "50%", width: 10, height: 10, background: c }} />{k}</span>)}
            </div>
          </section>
          <section style={{ flex: "0 1 260px", minWidth: 200, display: "flex", flexDirection: "column", gap: 10 }} aria-label="Summary">
            <div className="tile"><span className="eyebrow">Pairs off the diagonal</span><span className="n">{fmt(off)}</span><span className="l">of {fmt(cases.length)}, more than 0.05 apart</span></div>
            <div className="tile"><span className="eyebrow">Series where the winner changes</span><span className="n">{fmt(winners.changed)}</span><span className="l">of {fmt(winners.n)}</span></div>
            <div className="tile" style={{ gap: 8 }}><span className="eyebrow">Why they differ</span>
              {mix.map(([k, v]) => (
                <span key={k} style={{ display: "grid", gridTemplateColumns: "60px minmax(0, 1fr) 40px", alignItems: "center", gap: 8, fontSize: 12 }}>
                  <span>{k}</span><span style={{ height: 8, background: "var(--border-soft)", borderRadius: 4, overflow: "hidden" }}><span style={{ display: "block", width: `${v * 100}%`, height: "100%", background: REASON_COLOR[k] }} /></span><span className="mono" style={{ textAlign: "right" }}>{(v * 100).toFixed(0)}%</span>
                </span>
              ))}
            </div>
          </section>
        </div>

        <section className="panel" aria-label="Ranked cases">
          <div className="panel-head">
            <h2>Most interesting first</h2>
            <span className="muted" style={{ fontSize: 12 }}>by {score === "change" ? "rank change" : "distance"}, {Math.min(limit, ranked.length)} of {fmt(ranked.length)}</span>
            <input type="search" className="input" placeholder="filter by series or detector" aria-label="Filter cases" value={filter} onChange={(e) => setFilter(e.target.value)} style={{ marginLeft: "auto", height: 30, fontSize: 12, width: 220 }} />
          </div>
          <div style={{ overflowX: "auto" }}>
            <table className="table">
              <thead><tr><th>#</th><th>series</th><th>detector</th><th className="r">{metricLabel(xm)}</th><th className="r">{metricLabel(ym)}</th><th>rank</th><th>why</th><th>anomalies</th><th></th></tr></thead>
              <tbody className="mono">
                {ranked.slice(0, limit).map((c, i) => (
                  <tr key={`${c.file}|${c.detector}`} className={picked === c ? "picked" : undefined} onClick={() => setPicked(c)}>
                    <td>{i + 1}</td><td className="sans">{shortFile(c.file)}</td><td className="sans" style={{ fontWeight: 600 }}>{c.detector}</td>
                    <td className="r">{f2(c.x)}</td><td className="r">{f2(c.y)}</td>
                    <td>{c.rx} <span className="muted">to</span> {c.ry} <span style={{ color: c.change > 0 ? "var(--late)" : "var(--exact)" }}>{c.change > 0 ? "▼" : c.change < 0 ? "▲" : ""} {Math.abs(c.change) || ""}</span></td>
                    <td><span className="chip" style={{ background: REASON_COLOR[c.why] }}>{c.why}</span></td>
                    <td>{c.n_anomalies}, {c.why === "squashed" || c.why === "exact" ? `${c.n_exact} exact` : `${c.why === "late" ? c.n_late : c.why === "early" ? c.n_early : c.n_missed} ${c.why}`}</td>
                    <td className="r"><Link className="sans" style={{ fontWeight: 600, color: "var(--accent)" }} href={open(c)}>Open</Link></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
            {limit < ranked.length && <button type="button" className="btn" onClick={() => setLimit(limit + 20)}>Show 20 more</button>}
            {picked && <span className="mono" style={{ fontSize: 12 }}>picked: {picked.detector} on {shortFile(picked.file)} · <Link href={open(picked)} style={{ fontWeight: 600, color: "var(--accent)" }}>Open in Series</Link></span>}
          </div>
        </section>
      </main>
    </>
  );
}
