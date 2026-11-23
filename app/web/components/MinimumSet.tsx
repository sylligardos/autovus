"use client";
// Coverage, second tab: the minimum set. A detector finds an anomaly when its
// local AutoVUS reaches the threshold; the fewest detectors that find every
// anomaly some detector finds, and the anomalies found by none. Datasets,
// then series; a series shows its set's lanes and its anomalies found by none.

import Link from "next/link";
import { useDeferredValue, useEffect, useMemo, useState } from "react";
import { loadScoreDoc, loadSeriesDoc } from "@/lib/api";
import { href, type ScoreDoc, type SeriesDoc, type State } from "@/lib/model";
import { SET_COLORS, datasetOf, minSet, seriesName, type CoverTable, type MinSet } from "@/lib/local";
import { useRemembered } from "@/lib/remember";
import { Lane, type Band } from "./Lane";

const f2 = (v: number) => v.toFixed(2);
const NONE = "var(--down)";
type Sort = "n" | "none";

export function MinimumSet({ state, t, Ms, dataset, series, setDataset, setSeries }: {
  state: State; t: CoverTable; Ms: number[][][]; dataset: string | null; series: number | null;
  setDataset: (d: string | null) => void; setSeries: (i: number | null) => void;
}) {
  const b = state.benchmark;
  const [tau, setTau] = useRemembered(`coverage:${b}:tau`, 20);
  const [sort, setSort] = useRemembered<Sort>(`coverage:${b}:minsort`, "n");
  const cut = useDeferredValue(tau);
  const res = useMemo(() => Ms.map((M) => minSet(M, cut / 100)), [Ms, cut]);

  const byDs = useMemo(() => {
    const m: Record<string, number[]> = {};
    t.series.forEach((s, i) => { (m[datasetOf(s)] ??= []).push(i); });
    return m;
  }, [t]);

  const slider = (
    <label style={{ marginLeft: "auto", alignSelf: "center", display: "flex", flexDirection: "column", gap: 4, fontSize: 12, fontWeight: 600 }}>
      <span>Found: local AutoVUS ≥ <span className="mono">{f2(tau / 100)}</span></span>
      <input type="range" min={5} max={100} step={5} value={tau} onChange={(e) => setTau(Number(e.target.value))} style={{ width: 220 }} />
    </label>
  );

  if (series !== null) return <SeriesMinimum key={`${b}:${t.series[series]}`} state={state} t={t} i={series} r={res[series]} M={Ms[series]} slider={slider} />;

  const summarize = (idx: number[]) => {
    let n = 0, none = 0, K = 0, hit = 0;
    for (const i of idx) { n += res[i].S.length; none += res[i].none.length; K += t.K[i]; if (res[i].none.length) hit++; }
    return { n: n / Math.max(1, idx.length), none, K, hit, count: idx.length };
  };
  const scope = dataset ? byDs[dataset] ?? [] : Object.values(byDs).flat();
  const total = summarize(scope);
  const rows = (dataset
    ? scope.map((i) => ({ key: String(i), name: seriesName(t.series[i]), count: t.K[i], none: res[i].none.length, n: res[i].S.length, label: String(res[i].S.length), open: () => setSeries(i), mono: true }))
    : Object.keys(byDs).map((name) => { const r = summarize(byDs[name]); return { key: name, name, count: r.count, none: r.none, n: r.n, label: f2(r.n), open: () => setDataset(name), mono: false }; })
  ).sort((x, y) => (sort === "n" ? y.n - x.n || y.none - x.none : y.none - x.none || y.n - x.n));
  const head = (key: Sort, text: string) => (
    <th className="r" aria-sort={sort === key ? "descending" : undefined}>
      <button type="button" title={`Order by ${text.toLowerCase()}`} onClick={() => setSort(key)} style={{ padding: 0, border: 0, background: "transparent", cursor: "pointer", font: "inherit", letterSpacing: "inherit", textTransform: "inherit", color: sort === key ? "var(--ink)" : "inherit" }}>{text}{sort === key ? " ▾" : ""}</button>
    </th>
  );

  return (
    <>
      <Stats items={[
        ["Minimum set", f2(total.n), `detectors, mean over ${total.count} time series`, "var(--accent)"],
        ["Found by none", total.none.toLocaleString("en-US"), `of ${total.K.toLocaleString("en-US")} anomalies · in ${total.hit} time series`, "var(--ink)"],
      ]} action={slider} />
      <section className="panel" aria-label={dataset ? "Time series" : "Datasets"} style={{ overflowX: "auto" }}>
        <table className="table">
          <thead><tr><th>{dataset ? "Time series" : "Dataset"}</th><th className="r">{dataset ? "Anomalies" : "Time series"}</th>{head("none", "Found by none")}{head("n", "Detectors")}</tr></thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.key} onClick={r.open}>
                <td className={r.mono ? "mono" : "sans"} style={{ fontWeight: 600 }}>{r.name}</td>
                <td className="r mono muted">{r.count}</td>
                <td className={`r mono${r.none ? "" : " muted"}`}>{r.none}</td>
                <td className="r mono">{r.label}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
    </>
  );
}

function Stats({ items, action }: { items: [string, string, string, string][]; action?: React.ReactNode }) {
  return (
    <section className="panel" aria-label="Minimum set" style={{ flexDirection: "row", flexWrap: "wrap", alignItems: "flex-end", gap: 32, padding: "16px 20px" }}>
      {items.map(([label, value, note, color]) => (
        <div key={label} style={{ display: "flex", flexDirection: "column", gap: 2 }}>
          <span className="muted" style={{ fontSize: 12, fontWeight: 600 }}>{label}</span>
          <span className="mono" style={{ fontSize: 34, lineHeight: 1.1, color }}>{value}</span>
          <span className="muted" style={{ fontSize: 13 }}>{note}</span>
        </div>
      ))}
      {action}
    </section>
  );
}

function SeriesMinimum({ state, t, i, r, M, slider }: { state: State; t: CoverTable; i: number; r: MinSet; M: number[][]; slider: React.ReactNode }) {
  const file = t.series[i];
  const K = t.K[i];
  const setKey = r.S.join(",");
  const color = new Map(r.S.map((d, j) => [d, SET_COLORS[j % SET_COLORS.length]]));
  const takes = (d: number) => r.finder.reduce((s, w) => s + (w === d ? 1 : 0), 0);
  const [doc, setDoc] = useState<SeriesDoc | null>(null);
  const [scores, setScores] = useState<Record<string, ScoreDoc>>({});
  const [view, setView] = useState<[number, number] | null>(null);

  // Keyed by the series: the state starts empty on a new one, and a detector's score is kept when the set changes.
  useEffect(() => {
    let live = true;
    loadSeriesDoc(state.benchmark, file).then((d) => { if (live) setDoc(d); });
    return () => { live = false; };
  }, [state.benchmark, file]);
  useEffect(() => {
    let live = true;
    for (const d of setKey ? setKey.split(",").map(Number) : []) loadScoreDoc(state.benchmark, file, t.detectors[d]).then((s) => { if (live && s) setScores((m) => ({ ...m, [t.detectors[d]]: s })); });
    return () => { live = false; };
  }, [state.benchmark, file, setKey, t]);

  const anomalies = doc?.labels ?? [];
  const finderBands: Band[] = anomalies.map(([s, e], a) => ({ from: s, to: e + 1, color: r.finder[a] < 0 ? NONE : color.get(r.finder[a])!, opacity: r.finder[a] < 0 ? 0.8 : 0.55, key: `f${a}` }));
  const takenBands = (d: number): Band[] => anomalies.flatMap(([s, e], a) => (r.finder[a] === d ? [{ from: s, to: e + 1, color: color.get(d)!, opacity: 0.25, key: `b${a}` }] : []));
  const local = href("/local", state, { s: file, det: r.S.map((d) => t.detectors[d]).join(",") });
  // The best detector on an anomaly found by none.
  const best = (a: number) => M.reduce((w, row, d) => (Number.isFinite(row[a]) && (w < 0 || row[a] > M[w][a]) ? d : w), -1);
  const zoom = (a: number) => {
    if (!doc || !anomalies[a]) return;
    const [s, e] = anomalies[a], pad = Math.max(50, 5 * (e - s + 1));
    setView([Math.max(0, s - pad), Math.min(doc.n, e + 1 + pad)]);
  };

  return (
    <>
      <Stats items={[
        ["Minimum set", String(r.S.length), r.S.length === 1 ? "detector" : "detectors", "var(--accent)"],
        ["Found by none", String(r.none.length), `of ${K} ${K === 1 ? "anomaly" : "anomalies"}`, "var(--ink)"],
      ]} action={<>{slider}{r.S.length > 0 && <Link href={local} className="btn btn-primary" style={{ alignSelf: "center", display: "inline-flex", alignItems: "center", height: 36, textDecoration: "none" }}>Show in Local</Link>}</>} />

      <section className="panel" aria-label="The series and the minimum-set detectors">
        <div style={{ display: "grid", gridTemplateColumns: "150px minmax(0, 1fr)", gap: "8px 16px", alignItems: "center" }}>
          <span className="muted mono" style={{ fontSize: 11 }}>{doc ? `${doc.n.toLocaleString("en-US")} points · ${anomalies.length} anomalies` : ""}</span>
          {doc ? <Lane lo={doc.lo} hi={doc.hi} bins={doc.bins} n={doc.n} view={view} onZoom={setView} height={110} bands={finderBands} /> : <div style={{ height: 110, background: "var(--subtle)", borderRadius: 4 }} />}
          {r.S.map((d) => {
            const name = t.detectors[d];
            const sc = scores[name];
            return (
              <div key={name} style={{ display: "contents" }}>
                <span style={{ display: "flex", alignItems: "center", gap: 8, minWidth: 0, fontSize: 13, fontWeight: 600 }}><span className="swatch" style={{ background: color.get(d) }} /><span style={{ whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>{name}</span></span>
                {sc ? <Lane lo={sc.lo} hi={sc.hi} bins={sc.bins} n={sc.n} view={view} onZoom={setView} ymin={0} ymax={1} height={56} bands={takenBands(d)} stroke={color.get(d)} /> : <div style={{ height: 56, background: "var(--subtle)", borderRadius: 4 }} />}
              </div>
            );
          })}
        </div>
      </section>

      <div style={{ display: "flex", flexWrap: "wrap", gap: 16, alignItems: "flex-start" }}>
        <section className="panel" aria-label="Minimum set" style={{ flex: "1 1 380px", minWidth: 0 }}>
          <table className="table">
            <thead><tr><th>Detector</th><th className="r">Anomalies found</th></tr></thead>
            <tbody>
              {r.S.map((d) => (
                <tr key={d} style={{ cursor: "default" }}>
                  <td className="sans" style={{ fontWeight: 600 }}><span style={{ display: "inline-flex", alignItems: "center", gap: 8 }}><span className="swatch" style={{ background: color.get(d) }} />{t.detectors[d]}</span></td>
                  <td className="r mono">{takes(d)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
        <section className="panel" aria-label="Anomalies found by none" style={{ flex: "1 1 520px", minWidth: 0, maxHeight: 360, overflowY: "auto" }}>
          <table className="table">
            <thead><tr><th><span style={{ display: "inline-flex", alignItems: "center", gap: 8 }}><span className="swatch" style={{ background: NONE }} />Found by none</span></th><th className="r">Start</th><th className="r">Length</th><th>Best detector</th><th className="r">Local AutoVUS</th></tr></thead>
            <tbody>
              {r.none.map((a) => {
                const w = best(a), lab = anomalies[a];
                return (
                  <tr key={a} onClick={() => zoom(a)}>
                    <td className="mono" style={{ fontWeight: 600 }}>A{a + 1}</td>
                    <td className="r mono muted">{lab ? lab[0].toLocaleString("en-US") : ""}</td>
                    <td className="r mono muted">{lab ? (lab[1] - lab[0] + 1).toLocaleString("en-US") : ""}</td>
                    <td className="sans">{w < 0 ? "" : t.detectors[w]}</td>
                    <td className="r mono">{w < 0 ? "" : f2(M[w][a])}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </section>
      </div>
    </>
  );
}
