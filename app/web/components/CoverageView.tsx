"use client";
// Coverage: with each anomaly scored by the detector that is best on it,
// how good can a set of detectors be, and how few are needed. Datasets,
// then series; a series shows its best-coverage set, the best set of each
// size, and the set's lanes. "Show in Local" opens the series with the set.
// The second tab is the minimum set (MinimumSet.tsx).

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import { loadLocalTable, loadScoreDoc, loadSeriesDoc } from "@/lib/api";
import { href, type ScoreDoc, type SeriesDoc, type State } from "@/lib/model";
import { SET_COLORS, bestSets, datasetOf, oracle, seriesName, type CoverTable, type Oracle } from "@/lib/local";
import { useRemembered } from "@/lib/remember";
import { Lane, type Band } from "./Lane";
import { MinimumSet } from "./MinimumSet";

const f2 = (v: number) => v.toFixed(2);

export function CoverageView({ state }: { state: State }) {
  const b = state.benchmark;
  const [t, setT] = useState<CoverTable | null>(null);
  const [error, setError] = useState(false);
  const [dataset, setDataset] = useRemembered<string | null>(`coverage:${b}:dataset`, null);
  const [series, setSeries] = useRemembered<number | null>(`coverage:${b}:series`, null);
  const [tab, setTab] = useRemembered<"best" | "min">(`coverage:${b}:tab`, "best");

  useEffect(() => {
    let live = true;
    loadLocalTable<CoverTable>(b, "cover").then((x) => { if (live) { if (x) setT(x); else setError(true); } });
    return () => { live = false; };
  }, [b]);

  // Per series: the detector x anomaly matrix and its oracle, computed once.
  const per = useMemo(() => (t ? t.series.map((_, i) => {
    const K = t.K[i], flat = t.M[i];
    const M = t.detectors.map((_, d) => Array.from({ length: K }, (_, a) => (flat[d * K + a] < -900 ? NaN : flat[d * K + a] / 100)));
    return { M, o: oracle(M) };
  }) : []), [t]);
  const oracleOf = (i: number) => per[i].o;
  const Ms = useMemo(() => per.map((p) => p.M), [per]);

  const byDs = useMemo(() => {
    const m: Record<string, number[]> = {};
    t?.series.forEach((s, i) => { if (t.K[i] > 1) (m[datasetOf(s)] ??= []).push(i); });
    return m;
  }, [t]);

  const summarize = (idx: number[]) => {
    let cov = 0, single = 0, n = 0;
    for (const i of idx) { const o = oracleOf(i); cov += o.cov; single += o.alone[o.single]; n += o.set.length; }
    const m = Math.max(1, idx.length);
    return { cov: cov / m, single: single / m, n: n / m, count: idx.length };
  };

  const crumbs = (
    <div style={{ display: "flex", flexWrap: "wrap", alignItems: "center", justifyContent: "space-between", gap: 12 }}>
    <nav aria-label="Scope" style={{ display: "flex", flexWrap: "wrap", alignItems: "center", gap: 8, fontSize: 14 }}>
      <button type="button" onClick={() => { setDataset(null); setSeries(null); }} style={{ padding: 0, border: 0, background: "transparent", cursor: "pointer", fontSize: 14, fontWeight: 600, color: dataset ? "var(--accent)" : "var(--ink)" }}>All datasets</button>
      {dataset && (<><span className="muted" aria-hidden="true">›</span>
        <button type="button" onClick={() => setSeries(null)} style={{ padding: 0, border: 0, background: "transparent", cursor: "pointer", fontSize: 14, fontWeight: 600, color: series !== null ? "var(--accent)" : "var(--ink)" }}>{dataset}</button></>)}
      {t && series !== null && (<><span className="muted" aria-hidden="true">›</span><span className="mono" style={{ fontSize: 13 }}>{seriesName(t.series[series])}</span></>)}
    </nav>
    <div className="seg" role="group" aria-label="View" style={{ display: "flex" }}>
      <button type="button" aria-pressed={tab === "best"} onClick={() => setTab("best")} style={{ padding: "0 14px" }}>Best coverage</button>
      <button type="button" aria-pressed={tab === "min"} onClick={() => setTab("min")} style={{ padding: "0 14px" }}>Minimum set</button>
    </div>
    </div>
  );

  if (!t || !per.length) {
    return (
      <main className="content">{crumbs}
        <div className="panel" style={{ minHeight: 300 }}>{error && <span className="muted">No local tables. Run <code>conda run -n autovus python app/export_local.py</code>.</span>}</div>
      </main>
    );
  }

  if (tab === "min") return <main className="content">{crumbs}<MinimumSet state={state} t={t} Ms={Ms} dataset={dataset} series={series} setDataset={setDataset} setSeries={setSeries} /></main>;

  if (series !== null) return <main className="content">{crumbs}<SeriesCoverage state={state} t={t} i={series} M={per[series].M} o={per[series].o} /></main>;

  const scope = dataset ? byDs[dataset] ?? [] : Object.values(byDs).flat();
  const total = summarize(scope);
  let children = dataset
    ? scope.map((i) => { const o = oracleOf(i); return { key: String(i), name: seriesName(t.series[i]), count: t.K[i], s1: o.alone[o.single], sk: o.cov, n: String(o.set.length), open: () => setSeries(i), mono: true }; })
    : Object.keys(byDs).map((name) => { const r = summarize(byDs[name]); return { key: name, name, count: r.count, s1: r.single, sk: r.cov, n: r.n.toFixed(1), open: () => setDataset(name), mono: false }; });
  children = children.sort((a, c) => (c.sk - c.s1) - (a.sk - a.s1));
  const mx = Math.max(1e-9, ...children.map((r) => r.sk - r.s1));

  return (
    <main className="content">
      {crumbs}
      <Headline cov={total.cov} covNote={`mean over ${total.count} time series · ${total.n.toFixed(1)} detectors`} single={total.single} singleNote="mean over the same series" />
      <section className="panel" aria-label={dataset ? "Time series" : "Datasets"} style={{ overflowX: "auto" }}>
        <table className="table">
          <thead><tr><th>{dataset ? "Time series" : "Dataset"}</th><th className="r">{dataset ? "Anomalies" : "Time series"}</th><th className="r">Best single</th><th className="r">Best coverage</th><th style={{ width: 220 }}>Gain</th><th className="r">Detectors</th></tr></thead>
          <tbody>
            {children.map((r) => (
              <tr key={r.key} onClick={r.open}>
                <td className={r.mono ? "mono" : "sans"} style={{ fontWeight: 600 }}>{r.name}</td>
                <td className="r mono muted">{r.count}</td>
                <td className="r mono muted">{f2(r.s1)}</td>
                <td className="r mono">{f2(r.sk)}</td>
                <td>
                  <span style={{ display: "flex", alignItems: "center", gap: 10 }}>
                    <span className="mono" style={{ width: 48, textAlign: "right" }}>+{f2(r.sk - r.s1)}</span>
                    <span style={{ flex: 1, height: 8, background: "var(--border-soft)", borderRadius: 2, position: "relative" }}><span style={{ position: "absolute", left: 0, top: 0, bottom: 0, width: `${((r.sk - r.s1) / mx) * 100}%`, background: "var(--exact)", borderRadius: 2 }} /></span>
                  </span>
                </td>
                <td className="r mono">{r.n}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
    </main>
  );
}

function Headline({ cov, covNote, single, singleNote, action }: { cov: number; covNote: string; single: number; singleNote: string; action?: React.ReactNode }) {
  const item = (label: string, value: string, note: string, color: string) => (
    <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
      <span className="muted" style={{ fontSize: 12, fontWeight: 600 }}>{label}</span>
      <span className="mono" style={{ fontSize: 34, lineHeight: 1.1, color }}>{value}</span>
      <span className="muted" style={{ fontSize: 13 }}>{note}</span>
    </div>
  );
  return (
    <section className="panel" aria-label="Best coverage" style={{ flexDirection: "row", flexWrap: "wrap", alignItems: "flex-end", gap: 32, padding: "16px 20px" }}>
      {item("Best coverage", f2(cov), covNote, "var(--accent)")}
      {item("Best single", f2(single), singleNote, "var(--ink)")}
      {item("Gain", `+${f2(cov - single)}`, " ", "var(--exact)")}
      {action}
    </section>
  );
}

function SeriesCoverage({ state, t, i, M, o }: { state: State; t: CoverTable; i: number; M: number[][]; o: Oracle }) {
  const file = t.series[i];
  const color = new Map(o.set.map((d, j) => [d, SET_COLORS[j % SET_COLORS.length]]));
  const N = o.set.length;
  const trade = useMemo(() => (N > 1 ? bestSets(M, Math.min(N - 1, 6)) : []), [M, N]);
  const [doc, setDoc] = useState<SeriesDoc | null>(null);
  const [scores, setScores] = useState<Record<string, ScoreDoc>>({});
  const [view, setView] = useState<[number, number] | null>(null);

  useEffect(() => {
    let live = true;
    setDoc(null); setScores({}); setView(null);
    loadSeriesDoc(state.benchmark, file).then((d) => { if (live) setDoc(d); });
    for (const d of o.set) loadScoreDoc(state.benchmark, file, t.detectors[d]).then((s) => { if (live && s) setScores((m) => ({ ...m, [t.detectors[d]]: s })); });
    return () => { live = false; };
  }, [state.benchmark, file, o, t]);

  const anomalies = doc?.labels ?? [];
  const winnerBands: Band[] = anomalies.map(([s, e], a) => ({ from: s, to: e + 1, color: color.get(o.winner[a]) ?? "var(--label)", opacity: 0.55, key: `w${a}` }));
  const wonBands = (d: number): Band[] => anomalies.flatMap(([s, e], a) => (o.winner[a] === d ? [{ from: s, to: e + 1, color: color.get(d)!, opacity: 0.25, key: `b${a}` }] : []));
  const setNames = o.set.map((d) => t.detectors[d]);
  const local = href("/local", state, { s: file, det: setNames.join(",") });

  return (
    <>
      <Headline cov={o.cov} covNote={`${N} ${N === 1 ? "detector" : "detectors"}`} single={o.alone[o.single]} singleNote={t.detectors[o.single]}
        action={<Link href={local} className="btn btn-primary" style={{ marginLeft: "auto", alignSelf: "center", display: "inline-flex", alignItems: "center", height: 36, textDecoration: "none" }}>Show in Local</Link>} />

      <section className="panel" aria-label="The series and the best-coverage detectors">
        <div style={{ display: "grid", gridTemplateColumns: "150px minmax(0, 1fr)", gap: "8px 16px", alignItems: "center" }}>
          <span className="muted mono" style={{ fontSize: 11 }}>{doc ? `${doc.n.toLocaleString("en-US")} points · ${anomalies.length} anomalies` : ""}</span>
          {doc ? <Lane lo={doc.lo} hi={doc.hi} bins={doc.bins} n={doc.n} view={view} onZoom={setView} height={110} bands={winnerBands} /> : <div style={{ height: 110, background: "var(--subtle)", borderRadius: 4 }} />}
          {o.set.map((d) => {
            const name = t.detectors[d];
            const sc = scores[name];
            return (
              <div key={name} style={{ display: "contents" }}>
                <span style={{ display: "flex", alignItems: "center", gap: 8, minWidth: 0, fontSize: 13, fontWeight: 600 }}><span className="swatch" style={{ background: color.get(d) }} /><span style={{ whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>{name}</span></span>
                {sc ? <Lane lo={sc.lo} hi={sc.hi} bins={sc.bins} n={sc.n} view={view} onZoom={setView} ymin={0} ymax={1} height={56} bands={wonBands(d)} stroke={color.get(d)} /> : <div style={{ height: 56, background: "var(--subtle)", borderRadius: 4 }} />}
              </div>
            );
          })}
        </div>
      </section>

      <div style={{ display: "flex", flexWrap: "wrap", gap: 16, alignItems: "flex-start" }}>
        <section className="panel" aria-label="Best coverage set" style={{ flex: "1 1 380px", minWidth: 0 }}>
          <table className="table">
            <thead><tr><th>Detector</th><th className="r">Anomalies</th><th className="r">Alone</th></tr></thead>
            <tbody>
              {o.set.map((d) => (
                <tr key={d} style={{ cursor: "default" }}>
                  <td className="sans" style={{ fontWeight: 600 }}><span style={{ display: "inline-flex", alignItems: "center", gap: 8 }}><span className="swatch" style={{ background: color.get(d) }} />{t.detectors[d]}</span></td>
                  <td className="r mono">{o.wins.get(d)}</td>
                  <td className="r mono muted">{f2(o.alone[d])}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
        <section className="panel" aria-label="Fewer detectors" style={{ flex: "1 1 520px", minWidth: 0 }}>
          <table className="table">
            <thead><tr><th style={{ width: 40 }}>k</th><th className="r" style={{ width: 80 }}>Coverage</th><th>Detectors</th></tr></thead>
            <tbody>
              {[...trade.map((r) => ({ k: r.k, cov: r.cov, S: r.S, full: false })), { k: N, cov: o.cov, S: o.set, full: true }].map((r) => (
                <tr key={r.k} style={{ cursor: "default", background: r.full ? "var(--border-soft)" : undefined }}>
                  <td className="mono">{r.k}</td>
                  <td className="r mono">{f2(r.cov)}</td>
                  <td className="sans" style={{ whiteSpace: "normal" }}>
                    <span style={{ display: "flex", flexWrap: "wrap", gap: "4px 12px" }}>
                      {r.S.map((d) => <span key={d} style={{ display: "inline-flex", alignItems: "center", gap: 5, fontWeight: r.full ? 600 : 400 }}><span className="swatch" style={{ width: 9, height: 9, background: color.get(d) ?? "var(--border)" }} />{t.detectors[d]}</span>)}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      </div>
    </>
  );
}
