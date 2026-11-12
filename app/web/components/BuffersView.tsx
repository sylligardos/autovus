"use client";
// Buffers: how detectors are late, early, wide, exact or missed, per dataset
// and detector. A type picks the share shown in the matrix; a dataset opens
// its series; a cell lists the series of that pair.

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import { loadLocalTable } from "@/lib/api";
import { TYPE_COLOR, href, type DetectionType, type State } from "@/lib/model";
import { datasetOf, seriesName, type BuffersTable } from "@/lib/local";
import { useRemembered } from "@/lib/remember";

const LABEL: Record<DetectionType, string> = { exact: "Exact", early: "Early", late: "Late", wide: "Wide", missed: "Missed" };
// The type colours as hex, to tint the cells (the CSS variables cannot be mixed in JS).
const HEX: Record<DetectionType, string> = { exact: "#178F6F", early: "#3B6FE0", late: "#E07B1A", wide: "#8E5BD0", missed: "#8A96A8" };
const sum = (a: number[]) => a.reduce((x, y) => x + y, 0);

function tint(hex: string, share: number) {
  const c = [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16));
  const m = c.map((x) => Math.round(255 + (x - 255) * share));
  const lum = (0.2126 * m[0] + 0.7152 * m[1] + 0.0722 * m[2]) / 255;
  return { bg: `rgb(${m.join(",")})`, fg: lum < 0.5 ? "#FFFFFF" : "#152033" };
}

export function BuffersView({ state }: { state: State }) {
  const b = state.benchmark;
  const [L, setL] = useRemembered<20 | 128>(`buffers:${b}:L`, 20);
  const [type, setType] = useRemembered<DetectionType>(`buffers:${b}:type`, "exact");
  const [dataset, setDataset] = useRemembered<string | null>(`buffers:${b}:dataset`, null);
  const [cell, setCell] = useState<{ key: string; d: number } | null>(null);
  const [tables, setTables] = useState<Partial<Record<20 | 128, BuffersTable>>>({});
  const [error, setError] = useState(false);
  const t = tables[L] ?? null;

  useEffect(() => {
    if (tables[L]) return;
    let live = true;
    loadLocalTable<BuffersTable>(b, `buffers_${L}`).then((x) => { if (live) { if (x) setTables((m) => ({ ...m, [L]: x })); else setError(true); } });
    return () => { live = false; };
  }, [b, L, tables]);

  const agg = useMemo(() => {
    if (!t) return null;
    const ti = t.types.indexOf(type);
    const ds = t.series.map(datasetOf);
    const zero = () => [0, 0, 0, 0, 0];
    const total = zero();
    const byRow = new Map<string, number[]>(), byCol = t.detectors.map(zero), byCell = new Map<string, number[]>();
    const add = (acc: number[], r: number[]) => { for (let j = 0; j < 5; j++) acc[j] += r[2 + j]; };
    for (const r of t.rows) {
      if (dataset && ds[r[0]] !== dataset) continue;
      const key = dataset ? String(r[0]) : ds[r[0]];
      add(total, r); add(byCol[r[1]], r);
      if (!byRow.has(key)) byRow.set(key, zero());
      add(byRow.get(key)!, r);
      const ck = `${key}|${r[1]}`;
      if (!byCell.has(ck)) byCell.set(ck, zero());
      add(byCell.get(ck)!, r);
    }
    const share = (c: number[]) => c[ti] / Math.max(1, sum(c));
    const cols = t.detectors.map((_, j) => j).sort((x, y) => share(byCol[y]) - share(byCol[x]));
    const rows = [...byRow.keys()].sort((x, y) => share(byRow.get(y)!) - share(byRow.get(x)!));
    return { ds, total, byRow, byCol, byCell, share, cols, rows };
  }, [t, type, dataset]);

  const mix = (c: number[], vertical = false) => {
    const n = Math.max(1, sum(c));
    return (t?.types ?? []).map((ty, j) => (
      <span key={ty} title={`${LABEL[ty]} ${c[j]}`} style={vertical ? { height: `${(c[j] / n) * 100}%`, background: TYPE_COLOR[ty] } : { width: `${(c[j] / n) * 100}%`, background: TYPE_COLOR[ty] }} />
    ));
  };

  const D = t?.detectors.length ?? 32;
  const cols = `${dataset ? "300px" : "130px"} 110px repeat(${D}, 38px)`;
  const detail = t && agg && cell ? t.rows.filter((r) => r[1] === cell.d && (dataset ? String(r[0]) === cell.key : agg.ds[r[0]] === cell.key))
    .sort((x, y) => agg.share(y.slice(2, 7)) - agg.share(x.slice(2, 7))) : [];

  return (
    <main className="content" style={{ minWidth: 0 }}>
      <nav aria-label="Scope" style={{ display: "flex", flexWrap: "wrap", alignItems: "center", gap: 8, fontSize: 14 }}>
        <button type="button" onClick={() => { setDataset(null); setCell(null); }} style={{ padding: 0, border: 0, background: "transparent", cursor: "pointer", fontSize: 14, fontWeight: 600, color: dataset ? "var(--accent)" : "var(--ink)" }}>All datasets</button>
        {dataset && (<><span className="muted" aria-hidden="true">›</span><span style={{ fontWeight: 600 }}>{dataset}</span></>)}
        <div role="group" aria-label="Largest buffer L" className="muted" style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: 8, fontSize: 12 }}>L
          <div className="seg" style={{ display: "flex" }}>
            {([20, 128] as const).map((v) => <button key={v} type="button" aria-pressed={L === v} onClick={() => { setL(v); setCell(null); }} className="mono" style={{ padding: "0 14px" }}>{v}</button>)}
          </div>
        </div>
      </nav>

      {error && <div className="panel"><span className="muted">No local tables. Run <code>conda run -n autovus python app/export_local.py</code>.</span></div>}
      {!agg && !error && <div className="panel" style={{ height: 480 }} />}
      {agg && t && (
        <>
          <section className="panel" aria-label="Detection types">
            <div style={{ display: "flex", height: 26, borderRadius: 4, overflow: "hidden" }}>
              {t.types.map((ty, j) => (
                <button key={ty} type="button" aria-label={`${LABEL[ty]} ${((agg.total[j] / Math.max(1, sum(agg.total))) * 100).toFixed(1)}%`} onClick={() => setType(ty)}
                  style={{ width: `${(agg.total[j] / Math.max(1, sum(agg.total))) * 100}%`, height: 26, padding: 0, border: 0, background: TYPE_COLOR[ty], opacity: ty === type ? 1 : 0.55, cursor: "pointer" }} />
              ))}
            </div>
            <div role="group" aria-label="Detection type" style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
              {t.types.map((ty, j) => (
                <button key={ty} type="button" className="btn" aria-pressed={ty === type} onClick={() => setType(ty)}
                  style={{ display: "inline-flex", alignItems: "center", gap: 8, borderColor: ty === type ? "var(--ink)" : "var(--border)", fontWeight: ty === type ? 600 : 400, fontSize: 13 }}>
                  <span className="swatch" style={{ width: 10, height: 10, background: TYPE_COLOR[ty] }} />{LABEL[ty]}
                  <span className="mono muted" style={{ fontSize: 12 }}>{((agg.total[j] / Math.max(1, sum(agg.total))) * 100).toFixed(1)}%</span>
                </button>
              ))}
            </div>
          </section>

          <section className="panel" aria-label="Detection types by dataset and detector" style={{ padding: 0, overflow: "auto", maxHeight: 820 }}>
            <div role="table" aria-label={`Share of ${LABEL[type].toLowerCase()} detections`} style={{ display: "grid", gridTemplateColumns: cols, gap: 2, padding: "0 14px 12px", minWidth: "max-content" }}>
              <span role="columnheader" className="muted" style={{ position: "sticky", top: 0, zIndex: 2, background: "var(--panel)", display: "flex", alignItems: "flex-end", paddingBottom: 6, fontSize: 12, fontWeight: 600 }}>{dataset ? "Time series" : "Dataset"}</span>
              <span role="columnheader" className="muted" style={{ position: "sticky", top: 0, zIndex: 2, background: "var(--panel)", display: "flex", alignItems: "flex-end", paddingBottom: 6, fontSize: 12, fontWeight: 600 }}>Mix</span>
              {agg.cols.map((j) => (
                <span key={j} role="columnheader" title={t.detectors[j]} style={{ position: "sticky", top: 0, zIndex: 2, background: "var(--panel)", height: 128, display: "flex", alignItems: "flex-end", justifyContent: "center", paddingBottom: 6 }}>
                  <span style={{ writingMode: "vertical-rl", transform: "rotate(180deg)", fontSize: 11, fontWeight: 600, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis", maxHeight: 120 }}>{t.detectors[j]}</span>
                </span>
              ))}

              <span role="rowheader" className="muted" style={{ display: "flex", alignItems: "center", fontSize: 12, fontWeight: 600 }}>All</span>
              <span role="cell" style={{ display: "flex", alignItems: "center" }}><span style={{ display: "flex", width: "100%", height: 10, borderRadius: 2, overflow: "hidden" }}>{mix(agg.total)}</span></span>
              {agg.cols.map((j) => <span key={j} role="cell" style={{ display: "flex", flexDirection: "column", justifyContent: "flex-end", height: 44, borderRadius: 2, overflow: "hidden", background: "var(--subtle)" }}>{mix(agg.byCol[j], true)}</span>)}

              {agg.rows.map((key) => {
                const name = dataset ? seriesName(t.series[Number(key)]) : key;
                return (
                  <div key={key} style={{ display: "contents" }}>
                    <button type="button" role="rowheader" title={name} onClick={() => { if (!dataset) { setDataset(key); setCell(null); } }} className={dataset ? "mono" : undefined}
                      style={{ display: "flex", alignItems: "center", height: 30, padding: 0, border: 0, background: "transparent", cursor: dataset ? "default" : "pointer", textAlign: "left", fontSize: dataset ? 11 : 13, fontWeight: 600, color: "var(--ink)", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>{name}</button>
                    <span role="cell" style={{ display: "flex", alignItems: "center" }}><span style={{ display: "flex", width: "100%", height: 10, borderRadius: 2, overflow: "hidden" }}>{mix(agg.byRow.get(key)!)}</span></span>
                    {agg.cols.map((j) => {
                      const c = agg.byCell.get(`${key}|${j}`);
                      if (!c) return <span key={j} role="cell" style={{ height: 30, borderRadius: 3, background: "var(--subtle)" }} />;
                      const sh = agg.share(c);
                      const col = tint(HEX[type], sh);
                      const sel = cell && cell.key === key && cell.d === j;
                      return (
                        <button key={j} type="button" role="cell" onClick={() => setCell({ key, d: j })} aria-label={`${name} × ${t.detectors[j]}: ${Math.round(sh * 100)}% ${LABEL[type].toLowerCase()}`}
                          className="mono" style={{ width: "100%", height: 30, padding: 0, border: 0, borderRadius: 3, background: col.bg, color: col.fg, outline: sel ? "2px solid var(--ink)" : "none", outlineOffset: -2, cursor: "pointer", fontSize: 11 }}>
                          {Math.round(sh * 100)}%
                        </button>
                      );
                    })}
                  </div>
                );
              })}
            </div>
          </section>

          {cell && (
            <section className="panel" aria-label="Detail">
              <div className="panel-head">
                <span style={{ fontSize: 15, fontWeight: 700 }}>{dataset ? seriesName(t.series[Number(cell.key)]) : cell.key} × {t.detectors[cell.d]}</span>
                <span className="mono muted" style={{ fontSize: 12 }}>{(() => { const c = agg.byCell.get(`${cell.key}|${cell.d}`) ?? [0, 0, 0, 0, 0]; return `${sum(c)} anomalies · ${Math.round(agg.share(c) * 100)}% ${LABEL[type].toLowerCase()}`; })()}</span>
                <button type="button" className="icon-btn" aria-label="Close" onClick={() => setCell(null)} style={{ marginLeft: "auto" }}>×</button>
              </div>
              <table className="table">
                <thead><tr><th>Time series</th><th className="r">Anomalies</th><th style={{ width: 280 }}>Types</th><th className="r">Before</th><th className="r">After</th><th /></tr></thead>
                <tbody>
                  {detail.map((r) => (
                    <tr key={r[0]} style={{ cursor: "default" }}>
                      <td className="mono">{seriesName(t.series[r[0]])}</td>
                      <td className="r mono muted">{sum(r.slice(2, 7))}</td>
                      <td><span style={{ display: "flex", height: 12, borderRadius: 2, overflow: "hidden" }}>{mix(r.slice(2, 7))}</span></td>
                      <td className="r mono">{r[7].toFixed(1)}</td>
                      <td className="r mono">{r[8].toFixed(1)}</td>
                      <td className="r"><Link href={href("/local", state, { s: t.series[r[0]], det: t.detectors[cell.d] })} style={{ fontWeight: 600, color: "var(--accent)", textDecoration: "none" }}>Local</Link></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </section>
          )}
        </>
      )}
    </main>
  );
}
