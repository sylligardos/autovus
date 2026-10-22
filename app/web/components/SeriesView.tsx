"use client";
// The Series page: one series, its label, one lane per detector with the
// buffer AutoVUS selected on every anomaly, a hover card, and a drawer with
// the curves of the clicked anomaly.

import Link from "next/link";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { loadCurves, loadNotes, loadRank, loadScoreDoc, loadSelect, loadSeriesDoc, saveNotes, type Notes } from "@/lib/api";
import { TYPE_COLOR, f3, fmt, fsec, href, shortFile, type Anomaly, type CurvesDoc, type RankRow, type ScoreDoc, type SelectDoc, type SeriesDoc, type SeriesMeta, type State, type Summary } from "@/lib/model";
import { Lane, type Band, type Zone } from "./Lane";
import { LineChart } from "./LineChart";

const PINS_KEY = "autovus.pins";
const MAX_TOP = 8;
type Hover = { detector: string; a: Anomaly; left: number } | null;

function readPins(): string[] { try { return JSON.parse(localStorage.getItem(PINS_KEY) ?? "[]"); } catch { return []; } }
function writePins(p: string[]) { try { localStorage.setItem(PINS_KEY, JSON.stringify(p)); } catch { /* blocked */ } }

export function SeriesView({ state, summary, initialFile, initialPins, initialTop, seriesByDataset }: {
  state: State; summary: Summary; initialFile: string | null; initialPins: string[]; initialTop: number | null;
  seriesByDataset: Record<string, SeriesMeta[]>;
}) {
  const b = state.benchmark;
  const datasets = Object.keys(seriesByDataset);
  const firstFile = seriesByDataset[datasets[0]]?.[0]?.file ?? "";
  const [file, setFile] = useState(initialFile ?? firstFile);
  const upload = file.startsWith("upload:");
  const dataset = useMemo(() => datasets.find((d) => seriesByDataset[d].some((s) => s.file === file)) ?? datasets[0], [datasets, seriesByDataset, file]);
  const [L, setL] = useState(256);
  const [sides, setSides] = useState<"per_side" | "symmetric">("per_side");
  const [gpu, setGpu] = useState(false);
  const [topN, setTopN] = useState(initialTop ?? 3);
  const [pins, setPins] = useState<string[]>(initialPins);
  const [doc, setDoc] = useState<SeriesDoc | null>(null);
  const [rank, setRank] = useState<RankRow[]>([]);
  const [scores, setScores] = useState<Record<string, ScoreDoc>>({});
  const [selects, setSelects] = useState<Record<string, SelectDoc>>({});
  const [hover, setHover] = useState<Hover>(null);
  const [picked, setPicked] = useState<{ detector: string; k: number } | null>(null);
  const [curves, setCurves] = useState<CurvesDoc | null>(null);
  const [notes, setNotes] = useState<Notes>({});
  const [draft, setDraft] = useState("");
  const [error, setError] = useState<string | null>(null);
  const loaded = useRef(false);

  // Pins: the URL wins over the browser's memory.
  useEffect(() => {
    if (!initialPins.length) setPins(readPins());
    loadNotes().then((n) => setNotes(n ?? {}));
    loaded.current = true;
  }, [initialPins]);
  useEffect(() => { if (loaded.current) writePins(pins); }, [pins]);
  useEffect(() => { setDraft(notes[file]?.note ?? ""); }, [file, notes]);

  // The URL follows the series and the pins.
  useEffect(() => {
    const q = new URLSearchParams(window.location.search);
    q.set("s", file); q.delete("pin"); q.delete("top");
    window.history.replaceState(null, "", `${window.location.pathname}?${q}`);
  }, [file]);

  // The series, then the ranking of every detector on it.
  useEffect(() => {
    let live = true;
    setDoc(null); setScores({}); setSelects({}); setPicked(null); setCurves(null); setHover(null); setError(null);
    loadSeriesDoc(b, file).then((d) => { if (live) { if (d) setDoc(d); else setError("The Python side did not answer."); } });
    if (upload) setRank([{ detector: "yours", autovus: null, adj_vus: null, vus_pr: null, auc_pr: null }]);
    else loadRank(b, file, L, sides).then((r) => { if (live) setRank(r ?? []); });
    return () => { live = false; };
  }, [b, file, L, sides, upload]);

  const shown = useMemo(() => {
    const top = rank.slice(0, topN).map((r) => r.detector);
    const set = new Map<string, boolean>();
    for (const d of top) set.set(d, false);
    for (const d of pins) if (!upload) set.set(d, true);
    if (upload) set.set("yours", false);
    const order = new Map(rank.map((r, i) => [r.detector, i]));
    return Array.from(set.entries()).sort((a, c) => (order.get(a[0]) ?? 1e9) - (order.get(c[0]) ?? 1e9));
  }, [rank, topN, pins, upload]);

  // Scores and selections of the shown detectors, once each.
  useEffect(() => {
    let live = true;
    for (const [det] of shown) {
      if (!scores[det]) loadScoreDoc(b, file, det).then((s) => { if (live && s) setScores((m) => ({ ...m, [det]: s })); });
      if (!selects[det]) loadSelect(b, file, det, L, sides, gpu).then((s) => { if (live && s) setSelects((m) => ({ ...m, [det]: s })); });
    }
    return () => { live = false; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [shown, b, file, L, sides, gpu]);
  useEffect(() => { setSelects({}); setPicked(null); setCurves(null); }, [L, sides, gpu]);

  // The drawer.
  useEffect(() => {
    if (!picked) { setCurves(null); return; }
    let live = true;
    loadCurves(b, file, picked.detector, picked.k, L, sides, gpu).then((c) => { if (live) setCurves(c); });
    return () => { live = false; };
  }, [picked, b, file, L, sides, gpu]);

  const seconds = Object.values(selects).reduce((s, x) => s + x.seconds, 0);
  const n = doc?.n ?? 1;
  const labelBands: Band[] = useMemo(() => (doc?.labels ?? []).map(([s, e], i) => ({ from: s, to: e + 1, color: "var(--label)", opacity: 0.45, key: `l${i}` })), [doc]);
  const step = useCallback((dir: 1 | -1) => {
    const list = seriesByDataset[dataset] ?? [];
    const i = list.findIndex((s) => s.file === file);
    const next = list[(i + dir + list.length) % list.length];
    if (next) setFile(next.file);
  }, [seriesByDataset, dataset, file]);
  const sample = useCallback(() => {
    const all = datasets.flatMap((d) => seriesByDataset[d]);
    setFile(all[Math.floor(Math.random() * all.length)].file);
  }, [datasets, seriesByDataset]);

  const picksel = picked ? selects[picked.detector] : null;
  const pickA = picksel?.anomalies[picked!.k];

  return (
    <>
      <aside className="rail" aria-label="Controls">
        <div className="field">
          <span>Series</span>
          <select className="select" aria-label="Dataset" value={dataset} onChange={(e) => setFile(seriesByDataset[e.target.value][0].file)} disabled={upload}>
            {datasets.map((d) => <option key={d} value={d}>{d}</option>)}
          </select>
          <select className="select mono" aria-label="Series" value={upload ? "" : file} onChange={(e) => setFile(e.target.value)} disabled={upload} style={{ fontSize: 12 }}>
            {upload && <option value="">{file}</option>}
            {(seriesByDataset[dataset] ?? []).map((s) => <option key={s.file} value={s.file}>{s.file.replace(/\.csv$/, "")}</option>)}
          </select>
          <div style={{ display: "flex", gap: 6 }}>
            <button type="button" className="btn" style={{ flex: 1 }} onClick={() => step(-1)} disabled={upload}>Prev</button>
            <button type="button" className="btn" style={{ flex: 1 }} onClick={() => step(1)} disabled={upload}>Next</button>
            <button type="button" className="btn btn-primary" style={{ flex: 2 }} onClick={sample}>Sample</button>
          </div>
        </div>
        <div className="field">
          <span>Buffer</span>
          <span className="seg" role="group" aria-label="Buffer sides">
            <button type="button" aria-pressed={sides === "per_side"} onClick={() => setSides("per_side")}>Per side</button>
            <button type="button" aria-pressed={sides === "symmetric"} onClick={() => setSides("symmetric")}>Symmetric</button>
          </span>
          <label className="muted" style={{ display: "flex", flexDirection: "column", gap: 4, fontSize: 12, marginTop: 4 }}>
            <span style={{ display: "flex", justifyContent: "space-between" }}>L, largest buffer <span className="mono" style={{ color: "var(--ink)" }}>{L}</span></span>
            <input type="range" min={16} max={512} step={16} value={L} onChange={(e) => setL(Number(e.target.value))} />
          </label>
        </div>
        <div className="field">
          <span>Engine</span>
          <span className="seg" role="group" aria-label="Engine">
            <button type="button" aria-pressed={!gpu} onClick={() => setGpu(false)}>FastVUS CPU</button>
            <button type="button" aria-pressed={gpu} onClick={() => setGpu(true)} disabled={!summary.gpu} title={summary.gpu ? "" : "no GPU on this machine"}>GPU</button>
          </span>
        </div>
        <div className="field" style={{ paddingTop: 12, borderTop: "1px solid var(--border)" }}>
          <span>Computed</span>
          <span className="mono" style={{ fontSize: 22 }}>{Object.keys(selects).length ? fsec(seconds) : "…"}</span>
          <span className="muted" style={{ fontSize: 12 }}>{gpu ? "FastVUS GPU" : "FastVUS CPU"}, {shown.length} detectors, {doc?.labels.length ?? "…"} anomalies</span>
        </div>
        <div className="field muted" style={{ paddingTop: 12, borderTop: "1px solid var(--border)", fontSize: 12, gap: 6 }}>
          <span>Types</span>
          {(["exact", "early", "late", "missed"] as const).map((t) => <span key={t} style={{ display: "flex", alignItems: "center", gap: 8 }}><span className="swatch" style={{ background: TYPE_COLOR[t] }} />{t}</span>)}
          <span style={{ display: "flex", alignItems: "center", gap: 8 }}><span className="swatch" style={{ background: "var(--label)" }} />label</span>
        </div>
      </aside>

      <main className="content">
        <div style={{ display: "flex", alignItems: "baseline", gap: 14, flexWrap: "wrap" }}>
          <h1 style={{ fontSize: 22, fontWeight: 700, letterSpacing: "-0.01em" }}>{upload ? "Your data" : file.replace(/\.csv$/, "")}</h1>
          {doc && <span className="muted" style={{ fontSize: 13 }}>{fmt(doc.n)} points · {doc.labels.length} anomalies</span>}
          {error && <span style={{ color: "var(--late)", fontSize: 13 }}>{error}</span>}
        </div>

        <section className="panel" aria-label="The series">
          {doc ? <Lane lo={doc.lo} hi={doc.hi} bins={doc.bins} n={doc.n} height={120} bands={labelBands} /> : <div style={{ height: 120 }} />}
        </section>

        <section className="panel" aria-label="Detector lanes">
          <div className="panel-head">
            <h2>Detectors on this series</h2>
            <span className="muted" style={{ fontSize: 12 }}>top {topN} by AutoVUS, plus pinned</span>
          </div>
          {shown.map(([det, pinned]) => {
            const sel = selects[det];
            const zones: Zone[] = [];
            const bands: Band[] = [...labelBands];
            if (sel) for (const a of sel.anomalies) {
              const color = TYPE_COLOR[a.type];
              if (a.type === "missed") bands.push({ from: a.start, to: a.end + 1, color: "var(--missed)", opacity: 0.55, key: `m${a.k}` });
              if (a.b_left > 0) bands.push({ from: a.start - a.b_left, to: a.start, color, opacity: 0.4, key: `bl${a.k}` });
              if (a.b_right > 0) bands.push({ from: a.end + 1, to: a.end + 1 + a.b_right, color, opacity: 0.4, key: `br${a.k}` });
              zones.push({
                from: a.start - a.b_left, to: a.end + 1 + a.b_right, color, key: `z${a.k}`,
                selected: picked?.detector === det && picked.k === a.k,
                onEnter: () => setHover({ detector: det, a, left: ((a.start + a.end + 1) / 2 / n) * 100 }),
                onLeave: () => setHover((h) => (h && h.detector === det && h.a.k === a.k ? null : h)),
                onClick: () => setPicked({ detector: det, k: a.k }),
              });
            }
            const r = rank.find((x) => x.detector === det);
            const pos = rank.findIndex((x) => x.detector === det);
            const sc = scores[det];
            return (
              <div key={det} className="lane-row">
                <div className="lane-name">
                  <span style={{ display: "flex", alignItems: "center", gap: 6, fontWeight: 600 }}>
                    {pinned && <svg width="9" height="9" viewBox="0 0 10 10" aria-hidden="true"><path d="M5 0 L10 5 L5 10 L0 5 Z" fill="var(--accent)" /></svg>}
                    {det}
                    {!upload && <button type="button" className="unpin" aria-label={pinned ? `Unpin ${det}` : `Pin ${det}`} title={pinned ? "Unpin" : "Pin"}
                      onClick={() => setPins((p) => (pinned ? p.filter((x) => x !== det) : [...p, det]))}>{pinned ? "×" : "+"}</button>}
                  </span>
                  <span className="mono muted" style={{ fontSize: 11 }}>
                    {sel ? `AutoVUS ${f3(sel.autovus)} · Adj VUS ${f3(sel.skill_full)} · VUS ${f3(sel.vus_full)}` : r && r.autovus !== null ? `AutoVUS ${f3(r.autovus)} · Adj VUS ${f3(r.adj_vus)}` : "…"}{pos >= 0 && !upload ? ` · #${pos + 1}` : ""}
                  </span>
                </div>
                {sc ? (
                  <Lane lo={sc.lo} hi={sc.hi} bins={sc.bins} n={sc.n} ymin={0} ymax={1} bands={bands} zones={zones}>
                    {hover && hover.detector === det && (
                      <div className="hover-card" style={{ left: `${Math.min(Math.max(hover.left, 12), 88)}%` }}>
                        <span><b>{det}</b> on anomaly {hover.a.k + 1}, [{fmt(hover.a.start)}, {fmt(hover.a.end)}]</span>
                        <span className="mono"><span style={{ color: TYPE_COLOR[hover.a.type], fontWeight: 600 }}>{hover.a.type}</span> · b* = ({hover.a.b_left}, {hover.a.b_right})</span>
                      </div>
                    )}
                  </Lane>
                ) : <div style={{ height: 56, background: "var(--subtle)", borderRadius: 4 }} />}
              </div>
            );
          })}
          {!upload && (
            <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap", paddingTop: 8, borderTop: "1px solid var(--border)" }}>
              <label className="muted" style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 12 }}>Pin a detector
                <select className="select" style={{ height: 30, fontSize: 12 }} value="" onChange={(e) => { if (e.target.value) setPins((p) => (p.includes(e.target.value) ? p : [...p, e.target.value])); }}>
                  <option value="">…</option>
                  {[...summary.detectors, ...summary.references].filter((d) => !shown.some(([s]) => s === d)).map((d) => <option key={d} value={d}>{d}</option>)}
                </select>
              </label>
              <label className="muted" style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 12 }}>Top
                <input type="number" className="input" style={{ width: 56, height: 30, fontSize: 12 }} min={0} max={MAX_TOP} value={topN} onChange={(e) => setTopN(Math.max(0, Math.min(MAX_TOP, Number(e.target.value) || 0)))} />
              </label>
            </div>
          )}
        </section>

        {picked && pickA && (
          <section className="panel panel-accent" aria-label="Selected anomaly">
            <div className="panel-head">
              <h2>Why this buffer</h2>
              <span className="mono" style={{ fontSize: 12 }}>
                anomaly {pickA.k + 1} · {picked.detector} · [{fmt(pickA.start)}, {fmt(pickA.end)}] · <span style={{ color: TYPE_COLOR[pickA.type], fontWeight: 600 }}>{pickA.type}</span> · b* = ({pickA.b_left}, {pickA.b_right})
              </span>
              <button type="button" className="btn" style={{ marginLeft: "auto", height: 28 }} onClick={() => setPicked(null)}>Close</button>
            </div>
            {curves ? <Drawer curves={curves} a={pickA} L={L} /> : <div className="muted" style={{ fontSize: 12 }}>…</div>}
          </section>
        )}

        <section className="panel" aria-label="Notes">
          <label htmlFor="note" style={{ fontSize: 13, fontWeight: 700, letterSpacing: "0.02em", textTransform: "uppercase" }}>Notes on this series</label>
          <textarea id="note" rows={2} className="input" style={{ height: "auto", padding: "10px 12px", resize: "vertical", background: "var(--subtle)" }} value={draft} onChange={(e) => setDraft(e.target.value)} />
          <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
            <button type="button" className="btn btn-primary" onClick={() => { const next = { ...notes, [file]: { note: draft, updated: new Date().toISOString() } }; if (!draft.trim()) delete next[file]; setNotes(next); saveNotes(next); }}>Save note</button>
            <button type="button" className="btn" onClick={() => navigator.clipboard?.writeText(window.location.href)}>Copy link to this view</button>
            {notes[file] && <span className="muted" style={{ fontSize: 12 }}>saved {notes[file].updated.slice(0, 10)}</span>}
            <Link href={href("/cases", state)} style={{ marginLeft: "auto", fontSize: 12, fontWeight: 600, color: "var(--accent)" }}>Cases</Link>
          </div>
        </section>
      </main>
    </>
  );
}

function Drawer({ curves, a, L }: { curves: CurvesDoc; a: Anomaly; L: number }) {
  const z = curves.zoom;
  const bands: Band[] = [{ from: a.start - z.from, to: a.end + 1 - z.from, color: "var(--label)", opacity: 0.45 }];
  const color = TYPE_COLOR[a.type];
  if (a.b_left > 0) bands.push({ from: a.start - a.b_left - z.from, to: a.start - z.from, color, opacity: 0.35 });
  if (a.b_right > 0) bands.push({ from: a.end + 1 - z.from, to: a.end + 1 + a.b_right - z.from, color, opacity: 0.35 });
  const sides = Object.entries(curves.sides);
  return (
    <div style={{ display: "grid", gridTemplateColumns: "minmax(0, 5fr) minmax(0, 3fr) minmax(0, 3fr)", gap: 16 }}>
      <figure style={{ margin: 0, display: "flex", flexDirection: "column", gap: 6 }}>
        <figcaption className="muted" style={{ fontSize: 12 }}><b style={{ color: "var(--ink)" }}>(i) the anomaly</b> score, label, the selected buffer · points {fmt(z.from)} to {fmt(z.to)}</figcaption>
        <Lane lo={z.lo} hi={z.hi} bins={z.bins} n={z.to - z.from} height={150 * sides.length + 6 * (sides.length - 1)} ymin={0} ymax={1} bands={bands} />
      </figure>
      <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
        {sides.map(([side, s]) => (
          <LineChart key={side} x={curves.buffers} title={`(ii) ${side === "both" ? "both sides" : `${side} side`} · detector C(b) in ink, random G(b) in grey`} height={150}
            lines={[{ y: s.random_curve, color: "var(--missed)" }, { y: s.vus_curve, color: "var(--ink)" }]} mark={s.b} left="b = 0" right={String(L)} />
        ))}
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
        {sides.map(([side, s]) => (
          <LineChart key={side} x={curves.buffers} title={`(iii) ${side === "both" ? "both sides" : `${side} side`} · the rule, first argmax is b* = ${s.b}`} height={150}
            lines={[{ y: s.rule_curve, color: "var(--accent)", width: 1.8 }]} mark={s.b} dot={{ line: 0, i: s.idx }} left="b = 0" right={String(L)} />
        ))}
      </div>
    </div>
  );
}
